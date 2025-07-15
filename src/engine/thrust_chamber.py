import hydra
from cantera.cti2yaml import ideal_gas
from omegaconf import DictConfig
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.columns import Columns
from typing import Literal
import cantera as ct
from rocketcea.cea_obj import CEA_Obj
import rocketisp
import numpy as np
console = Console()

from pint import UnitRegistry, DimensionalityError
ureg = UnitRegistry()
Q_ = ureg.Quantity

# ── add near the top of the file ──────────────────────────────────────────────
DISPLAY_UNITS = {
    "thrust":      "kN",    # kilo-newton
    "throat_dia":  "mm",    # millimetre
    "mass_flow":   "kg/s",  # kilogram per second
}


def fmt_qty(q, criterion: str | None = None) -> str:
    """
    Return a compact human-readable string for *q*.
    If *criterion* is given and listed in DISPLAY_UNITS it is converted
    before formatting.
    """
    try:
        target = DISPLAY_UNITS.get(criterion)
        return f"{q.to(target) if target else q:~P}"
    except Exception:
        # Fallback: show whatever Pint gives us
        return f"{q:~P}"


def make_qty(node, *, to_si: bool = True) -> Q_:
    """
    Convert an OmegaConf node with keys
        value: <number>
        unit : <str>
    into a pint.Quantity.

    If *to_si* is True (default) the quantity is expressed in
    SI base units via `.to_base_units()`.  Offset units such as
    degC / degF are handled explicitly.
    """
    qty = Q_(node.value, node.unit)

    if not to_si:
        return qty

    # Most units convert straight to base units.
    try:
        return qty.to_base_units()

    # Offset units (°C, °F, …) need manual mapping.
    except DimensionalityError:
        offset_map = {
            "degC": "kelvin",
            "°C":   "kelvin",
            "degF": "kelvin",
            "°F":   "kelvin",
        }
        target = offset_map.get(str(qty.units))
        return qty.to(target) if target else qty


class ThrustChamber:
    """
    Simple model that receives quantities already converted
    from the {value, unit} representation.
    """

    def __init__(self, cfg: DictConfig, design_criteria: Literal["thrust", "throat_dia", "mass_flow"]) -> None:
        self.cfg = cfg
        
        self.design_criteria = design_criteria
        self.design_point    = make_qty(cfg.engine.design_criteria[design_criteria])
        
        self.fuel     = cfg.engine.fuel
        self.oxidizer = cfg.engine.oxidizer
        self.ROF      = cfg.engine.inlet_conditions.ROF

        self.p_inlet = make_qty(cfg.engine.inlet_conditions.pressure)
        self.T_inlet = make_qty(cfg.engine.inlet_conditions.temperature)
        
        self.p_outlet = make_qty(cfg.engine.outlet_conditions.pressure)
        
        self.gas = self.hp_combustion()

    def hp_combustion(self):
        """
        Adiabatic combustion of the fuel and oxidizer.
        """
        gas_adiabatic = ct.Solution("gri30.yaml")

        # Strip Pint units → floats expected by Cantera
        T = self.T_inlet.m_as("K")   # kelvin
        P = self.p_inlet.m_as("Pa")  # pascal

        gas_adiabatic.TPY = T, P, f"{self.fuel}:1, {self.oxidizer}:{self.ROF}"
        gas_adiabatic.equilibrate("HP")
        return gas_adiabatic
    
    def ideal_performance(self):
        """
        Calculate the thrust and mass flow rate.
        """
        
        gamma     = self.gas.cp / self.gas.cv
        R         = ct.gas_constant / self.gas.mean_molecular_weight
        P_outlet  = self.p_outlet.m_as("Pa")
        P_0       = self.gas.P
        # Characteristic Velocity
        c_star    = np.sqrt((R * self.gas.T / gamma) * ((gamma + 1) / 2) ** ((gamma + 1) / (gamma - 1)))
        # Thurst coefficient
        c_f = np.sqrt((2 * gamma ** 2) / (gamma - 1) * (2 / (gamma + 1)) ** ((gamma + 1) / (gamma - 1)) * (1 - (P_outlet / P_0) ** ((gamma - 1) / gamma)))
        # Effective exhaust
        c = c_star * c_f
        # Isp ideal
        Isp_ideal = c / 9.81
        
        if self.design_criteria == "thrust":
            thrust = self.design_point.m_as("N")
            mass_flow = (thrust / (c_star * c_f)) ## kg/s
            At = c_star * mass_flow / P_0 # .m_as("kg/s")
            
            return mass_flow * Q_(1, "kg/s"), At * Q_(1, "m^2"), Isp_ideal * Q_(1, "s"), c * Q_(1, "m/s")
        
        elif self.design_criteria == "throat_dia":
            throat_dia = self.design_point.m_as("m")
            At = np.pi * (throat_dia / 2) ** 2
            mass_flow = (At * P_0 / np.sqrt(self.gas.T)) * np.sqrt(gamma/R) * ((gamma + 1) / 2)**((gamma + 1) / 2*(gamma - 1))
            
            thrust = mass_flow * c_star * c_f
            return mass_flow * Q_(1, "kg/s"), thrust * Q_(1, "N"), Isp_ideal * Q_(1, "s"), c * Q_(1, "m/s")
            
        elif self.design_criteria == "mass_flow":
            mass_flow = self.design_point.m_as("kg/s")
            thrust = mass_flow * c_star * c_f
            At = mass_flow * np.sqrt(self.gas.T) / P_0 / np.sqrt(gamma/R) * ((gamma + 1) / 2)**((gamma + 1) / 2*(gamma - 1))
            throat_dia = 2 * np.sqrt(At / np.pi)
            
            return mass_flow * Q_(1, "kg/s"), thrust * Q_(1, "N"), Isp_ideal * Q_(1, "s"), c * Q_(1, "m/s")
        
        return None
    
    def nozzle(self, n_points=200, show_plot=True):
        """Analyze nozzle performance across a range of pressure conditions.

        Computes flow properties (area ratio, Mach number, temperature, pressure)
        through a converging-diverging nozzle assuming isentropic flow.
        Uses plotly for interactive visualization.

        Args:
            n_points: Number of data points to calculate
            show_plot: Whether to display the plotly visualization

        Returns:
            Tuple of (data, fig) where data is a numpy array of calculated values
            and fig is the plotly figure object
        """
        import numpy as np
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        # Use the class's gas object with its properties
        gas = self.gas

        # Store initial conditions
        s0 = gas.s
        h0 = gas.h
        p0 = gas.P
        T0 = gas.T

        # For a normalized analysis, mass flow rate is arbitrary
        mdot = 1.0

        # Initialize data array: [area_ratio, mach_number, temp_ratio, press_ratio]
        data = np.zeros((n_points, 4))

        # Calculate properties at different pressure points
        for i, p in enumerate(np.linspace(0.01 * p0, 0.99 * p0, n_points)):
            # Set the state using constant entropy (isentropic) and varying pressure
            gas.SP = s0, p

            # Calculate velocity from conservation of energy
            v = np.sqrt(2.0 * (h0 - gas.h))  # h + V^2/2 = h0

            # Calculate area from continuity equation
            area = mdot / (gas.density * v)  # rho*v*A = constant

            # Calculate Mach number
            Ma = v / gas.sound_speed

            # Store computed values
            data[i, :] = [area, Ma, gas.T / T0, p / p0]

        # Normalize by the minimum area (nozzle throat)
        data[:, 0] /= min(data[:, 0])

        # Create plotly figure with dual y-axes
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add pressure ratio trace
        fig.add_trace(
            go.Scatter(
                x=data[:, 1],  # Mach number
                y=data[:, 3],  # Pressure ratio
                name="P/P₀",
                line=dict(color="red", width=2)
            ),
            secondary_y=False
        )

        # Add temperature ratio trace
        fig.add_trace(
            go.Scatter(
                x=data[:, 1],  # Mach number
                y=data[:, 2],  # Temperature ratio
                name="T/T₀",
                line=dict(color="green", width=2)
            ),
            secondary_y=False
        )

        # Add area ratio trace on secondary y-axis
        fig.add_trace(
            go.Scatter(
                x=data[:, 1],  # Mach number
                y=data[:, 0],  # Area ratio
                name="A/A*",
                line=dict(color="blue", width=2)
            ),
            secondary_y=True
        )

        # Update layout and axes
        fig.update_layout(
            title="Nozzle Flow Properties",
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=60, r=60, t=80, b=60)
        )

        fig.update_xaxes(title_text="Mach Number")
        fig.update_yaxes(title_text="Temperature / Pressure Ratio", range=[0, 1.05], secondary_y=False)
        fig.update_yaxes(title_text="Area Ratio", secondary_y=True)

        # Restore gas to original state
        gas.SP = s0, p0

        if show_plot:
            fig.show()

        return data, fig
    
    def results(self):
        table = Table(title="[bold green]Thrust Chamber Summary & Results[/bold green]", box=None)
        
        table.add_column("Parameter", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")
        
        table.add_row("Design Criteria", f"{self.design_criteria}, {fmt_qty(self.design_point, self.design_criteria)}")
        table.add_row("Fuel", f"{self.fuel}")
        table.add_row("Oxidizer", f"{self.oxidizer}")
        table.add_row("Inlet Pressure", f"{self.p_inlet.to('bar'):~P}")
        table.add_row("Inlet Temperature", f"{self.T_inlet:~P}")
        table.add_row("ROF", f"{self.ROF}")
        table.add_row("-"*30)
        table.add_row("Adiabatic Temperature", f"{self.gas.T} K")
        
        console.print(table)
        
@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    chamber = ThrustChamber(cfg, "throat_dia")
    chamber.results()
    massflow, thrust, Isp_ideal, effective_exh = chamber.ideal_performance()
    
    print(f"Mass Flow Rate: {fmt_qty(massflow)}")
    print(f"Thrust: {fmt_qty(thrust)}")
    print(f"Isp Ideal: {fmt_qty(Isp_ideal)}")
    chamber.nozzle()
    

if __name__ == "__main__":
    main()
    from rocketcea.cea_obj_w_units import CEA_Obj
    
    C = CEA_Obj(oxName='LOX', fuelName='CH4', pressure_units='MPa')
    
    IspAmb, mode = C.estimate_Ambient_Isp(Pc=10, MR=3.1, eps=8, Pamb=1)
    IspVac = C.get_Isp(Pc=30, MR=4, eps=1000)
    print("Isp RocketCEA: ",IspAmb)
    print("Isp CEA: ",IspVac)

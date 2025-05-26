import cantera as ct
from dataclasses import dataclass
from pathlib import Path
import src
import os


@dataclass(slots=True)
class CombustionResults:
    pass


class Combustion:
    def __init__(self):
        # Load the configuration data
        self.config_params = src.config_data

        # Set the propellants for the combustion calculations
        self.fuel = self.config_params["propellants"]["fuel"]["identifier"][0]
        self.oxidiser = self.config_params["propellants"]["oxidiser"]["identifier"][0]

        # Set the combustion conditions
        self.comb_pressure = float(self.config_params["conditions"]["combustion_pressure"])
        self.initial_temp = float(self.config_params["conditions"]["initial_temperature"])

        self.OF_ratio = float(self.config_params["conditions"]["of_ratio"])
        self.mixture_fraction = 1 / (1 + self.OF_ratio)

        # Set the path to the mechanism file

        project_root = Path(src.__file__).resolve().parent.parent  # <repo>/src/..
        self.mechanism_path = project_root / "data" / "custom_gri30_highT.yaml"

        # 2. Fail fast if the file is missing.
        if not self.mechanism_path.is_file():
            raise FileNotFoundError(f"Mechanism file not found: {self.mechanism_path}")

    def hp_combustion(self):
        # Initialize the gas object with the custom GRI 3.0 mechanism
        gas_hp = ct.Solution(self.mechanism_path)

        # Set the temperature and pressure
        gas_hp.TP = self.initial_temp, self.comb_pressure

        # Set a specific mixture fraction
        gas_hp.set_mixture_fraction(self.mixture_fraction, self.fuel, self.oxidiser, basis="mass")

        gas_hp.name = "HP Combustion Gas"
        gas_hp.equilibrate("HP")

        return gas_hp

    def free_flame(self):
        gas_ff = ct.Solution(self.mechanism_path)
        gas_ff.TP = self.initial_temp, self.comb_pressure

        gas_ff.set_mixture_fraction(self.mixture_fraction, self.fuel, self.oxidiser, basis="mass")

        ff = ct.FreeFlame(gas_ff, width=0.1)
        ff.show()

        ff.solve(loglevel=2, refine_grid=True, auto=True)
        # ff.solve(loglevel=1, stage=2)

        ff.show()
        print(f"mixture-averaged flamespeed = {ff.velocity[0]:7f} m/s")

        return ff

    def core_flame(self):
        pass

    def core_reactor(self):
        pass


if __name__ == "__main__":
    # Example usage

    from rich import print
    from rich.table import Table  # Import Table

    # Instantiate and run the combustion calculation
    combustion_instance = Combustion()
    gas = combustion_instance.hp_combustion()
    # gas = combustion_instance.free_flame()

    # --- Create the Rich Table ---
    table = Table(
        title="Combustion Inputs and Results",
        show_header=True,
        header_style="bold magenta",
    )

    # Define columns
    table.add_column("Parameter", style="dim", width=25)
    table.add_column("Value", justify="right")
    table.add_column("Units", justify="left", style="italic")

    # --- Add Input Rows ---
    table.add_row("[bold]Inputs[/bold]", "", "")  # Section Header
    table.add_row("Fuel", combustion_instance.fuel, "")
    table.add_row("Oxidiser", combustion_instance.oxidiser, "")
    table.add_row("Initial Temperature", f"{combustion_instance.initial_temp:.2f}", "K")
    table.add_row(
        "Combustion Pressure", f"{combustion_instance.comb_pressure / 1e5:.2f}", "bar"
    )  # Convert Pa to bar for display
    table.add_row("O/F Ratio", f"{combustion_instance.OF_ratio:.3f}", "")
    table.add_row("Mixture Fraction (fuel)", f"{combustion_instance.mixture_fraction:.4f}", "")
    table.add_row("Mechanism", os.path.basename(combustion_instance.mechanism_path), "")  # Show just the filename

    # --- Add Output Rows ---
    table.add_section()  # Separator
    table.add_row("[bold]Results (Equilibrium)[/bold]", "", "")  # Section Header
    table.add_row("Temperature", f"{gas.T:.2f}", "K")
    table.add_row("Pressure", f"{gas.P / 1e5:.2f}", "bar")  # Convert Pa to bar for display
    table.add_row("Density", f"{gas.density:.4f}", "kg/m³")
    table.add_row("Mean Molecular Weight", f"{gas.mean_molecular_weight:.4f}", "kg/kmol")
    table.add_row("Specific Enthalpy (mass)", f"{gas.enthalpy_mass / 1e6:.4f}", "MJ/kg")  # Convert J/kg to MJ/kg
    table.add_row("Specific Entropy (mass)", f"{gas.entropy_mass / 1e3:.4f}", "kJ/kg·K")  # Convert J/kg K to kJ/kg K
    table.add_row("Specific Heat Cp (mass)", f"{gas.cp_mass / 1e3:.4f}", "kJ/kg·K")  # Convert J/kg K to kJ/kg K
    table.add_row("Specific Heat Cv (mass)", f"{gas.cv_mass / 1e3:.4f}", "kJ/kg·K")  # Convert J/kg K to kJ/kg K
    table.add_row("Gamma (Cp/Cv)", f"{gas.cp_mass / gas.cv_mass:.4f}", "")
    table.add_row("Viscosity", f"{gas.viscosity:.3e}", "Pa·s")
    table.add_row("Thermal Conductivity", f"{gas.thermal_conductivity:.3f}", "W/m·K")

    # Print the table
    print(table)

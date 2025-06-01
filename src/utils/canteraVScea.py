import cantera as ct
import numpy as np
from typing import List, Dict
from pathlib import Path
import src

class ThrustChamber:
    def __init__(self):
        # Load the configuration data
        self.config_params = src.config_data

        # Set the propellants for the combustion calculations
        self.fuel       = self.config_params["propellants"]["fuel"]["identifier"][0]
        self.oxidiser   = self.config_params["propellants"]["oxidiser"]["identifier"][0]

        # Set the combustion conditions
        self.comb_pressure  = float(self.config_params["conditions"]["combustion_pressure"])
        self.initial_temp   = float(self.config_params["conditions"]["initial_temperature"])

        self.OF_ratio           = float(self.config_params["conditions"]["of_ratio"])
        self.mixture_fraction   = 1 / (1 + self.OF_ratio)

        # Set the path to the mechanism file

        project_root        = Path(src.__file__).resolve().parent.parent  # <repo>/src/..
        self.mechanism_path = project_root / "data" / "custom_gri30_highT.yaml"

        # 2. Fail fast if the file is missing.
        if not self.mechanism_path.is_file():
            raise FileNotFoundError(f"Mechanism file not found: {self.mechanism_path}")
    # ─────────────────────────────────────────────────────────────────────
    #  PUBLIC DRIVER
    # ─────────────────────────────────────────────────────────────────────
    def solve_stationary_flow(self, *, N_segments: int = 60, atol=1e-4, rtol=1e-6):
        """
        High-level entry: build geometry, run shifting-equilibrium network,
        populate `self.results` as a dict of numpy arrays.
        """
        self._build_geometry(N_segments)
        self._initialise_gas()
        self._instantiate_reactors()
        self._iterate_shifting_eq(atol=atol, rtol=rtol)
        self._collect_results()
    
    # ─────────────────────────────────────────────────────────────────────
    #  STEP 1 – Geometry
    # ─────────────────────────────────────────────────────────────────────
    def _build_geometry(
        self,
        N: int,
        *,
        grid_mode: str = "uniform",  # "uniform" | "mach"
        dM: float = 0.05,
    ):
        """
        Create the 1-D geometry arrays.

        grid_mode="uniform" : equidistant Δx  (legacy)
        grid_mode="mach"    : equidistant ΔMach based on a frozen-flow guess
        """
        # cfg = self.config_params["geometry"]
        # D_throat = float(cfg["D_throat_m"])
        # eps = float(cfg["expansion_ratio"])
        # L_total = float(cfg["length_m"])
        
        D_throat = 22.0e-3  # m
        eps = 4.0  # expansion ratio
        L_total = 0.2  # m
        

        if grid_mode == "uniform":
            self.x = np.linspace(0.0, L_total, N + 1)

        elif grid_mode == "mach":
            #
            # 1. cheap isentropic map  M(x)  assuming linear-cone contour
            #
            gamma = 1.25  # rough average
            # axial reference grid just for the surrogate solution
            x_ref = np.linspace(0.0, L_total, 6 * N)  # fine, inexpensive
            D_ref = np.interp(x_ref, [0, L_total], [D_throat, D_throat * np.sqrt(eps)])
            A_ref = 0.25 * np.pi * D_ref**2
            Arat = A_ref / A_ref.min()  # A / A*

            # Isentropic relation  (γ+1)/(γ-1)  exponent form
            term = (Arat ** ((gamma - 1) / gamma) - 1) * 2 / (gamma - 1)
            M_ref = np.sqrt(np.maximum(term, 0.0))  # sub + sup branch

            #
            # 2. select nodes so that ΔM ≈ dM everywhere
            #
            x_nodes = [x_ref[0]]
            last_M = M_ref[0]
            for x_i, M_i in zip(x_ref[1:], M_ref[1:]):
                if abs(M_i - last_M) >= dM:
                    x_nodes.append(x_i)
                    last_M = M_i
            if x_nodes[-1] < x_ref[-1]:
                x_nodes.append(x_ref[-1])  # ensure nozzle exit

            self.x = np.asarray(x_nodes)

        else:
            raise ValueError(f"Unknown grid_mode '{grid_mode}'")

        # Common post-processing for both modes
        self.D = np.interp(self.x, [0, L_total], [D_throat, D_throat * np.sqrt(eps)])
        self.A = 0.25 * np.pi * self.D**2

    # ─────────────────────────────────────────────────────────────────────
    #  STEP 2 – Gas object & inlet state
    # ─────────────────────────────────────────────────────────────────────
    def _initialise_gas(self):
        self.gas = ct.Solution("self.mechanism_path")
        # mass fractions from OF + mixture fraction
        Y_fuel   = self.mixture_fraction
        Y_ox     = 1 - Y_fuel
        comp     = {self.fuel: Y_fuel, self.oxidiser: Y_ox}
        self.gas.TPX = self.initial_temp, self.comb_pressure, comp
        self.h0  = self.gas.enthalpy_mass
        self.rho0 = self.gas.density
        self.a0   = self.gas.sound_speed

    # ─────────────────────────────────────────────────────────────────────
    #  STEP 3 – Reactors
    # ─────────────────────────────────────────────────────────────────────
    def _instantiate_reactors(self):
        self._segments: List[ct.Reactor] = []
        for i in range(len(self.x) - 1):
            V = self.A[i] * (self.x[i + 1] - self.x[i])
            r = ct.IdealGasReactor(self.gas, energy='on', volume=V)
            self._segments.append(r)

        # Valve network (mass-flow couplers)
        self._valves: List[ct.Valve] = []
        for up, down in zip(self._segments[:-1], self._segments[1:]):
            v = ct.Valve(upstream=up, downstream=down)
            v.valve_coeff = 1.0   # placeholder, will be updated
            self._valves.append(v)

        self.sim = ct.ReactorNet(self._segments)

    # ─────────────────────────────────────────────────────────────────────
    #  STEP 4–5 – Shifting-equilibrium outer loop
    # ─────────────────────────────────────────────────────────────────────
    def _iterate_shifting_eq(self, *, atol, rtol, max_iter=50):
        P_old = np.full(len(self._segments), self.comb_pressure)
        for k in range(max_iter):
            # 1) Chemistry relaxation
            self.sim.advance_to_steady_state()
            # 2) Momentum-based ΔP
            self._update_valves_from_momentum()
            # 3) Convergence check
            P_new = np.array([seg.thermo.P for seg in self._segments])
            if np.max(np.abs(P_new - P_old) / P_old) < rtol:
                break
            P_old = P_new.copy()
        else:
            raise RuntimeError("Shifting-equilibrium did not converge")

    def _update_valves_from_momentum(self):
        """Recompute mass-flow in each throat to satisfy ρ u A continuity."""
        # First segment uses guessed mdot from config or choked condition
        mdot_up = float(self.config_params["conditions"]["m_dot_kg_s"])
        for i, v in enumerate(self._valves):
            seg = self._segments[i]
            rho = seg.thermo.density
            u   = mdot_up / (rho * self.A[i])            # update velocity
            # quasi-1D momentum equation: ΔP_mom = ρ u² (A2/A1 – 1)
            A_down = self.A[i + 1]
            delta_p = rho * u * u * (A_down / self.A[i] - 1.0)
            P_target = seg.thermo.P - delta_p
            # choose valve coefficient such that downstream P ≈ target
            v.valve_coeff = max(1e-6, (seg.thermo.P - P_target) / 1e5)
            mdot_up = mdot_up  # steady 1-D assumption – constant mdot

    # ─────────────────────────────────────────────────────────────────────
    #  STEP 6 – Results
    # ─────────────────────────────────────────────────────────────────────
    def _collect_results(self):
        self.results: Dict[str, np.ndarray] = {
            "x_m":        self.x[:-1],                     # node centre
            "P_Pa":       np.array([r.thermo.P for r in self._segments]),
            "T_K":        np.array([r.thermo.T for r in self._segments]),
            "rho_kg_m3":  np.array([r.thermo.density for r in self._segments]),
            "u_m_s":      self._velocity_profile(),
            "Mach":       self._mach_profile(),
            "species_X":  np.vstack([r.thermo.X for r in self._segments]),
        }

    def _velocity_profile(self) -> np.ndarray:
        mdot = float(self.config_params["conditions"]["m_dot_kg_s"])
        return mdot / (self.results["rho_kg_m3"] * self.A[:-1])

    def _mach_profile(self) -> np.ndarray:
        a = np.array([r.thermo.sound_speed for r in self._segments])
        return self.results["u_m_s"] / a


if __name__ == "__main__":
    # Example usage
    tc = ThrustChamber()
    # 80 segments with roughly 0.05 Mach increment
    tc._build_geometry(N=80, grid_mode="mach", dM=0.05)
    tc.solve_stationary_flow(atol=1e-4, rtol=1e-6)

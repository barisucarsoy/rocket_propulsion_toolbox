from pathlib import Path
import src
import cantera as ct

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
        # Load the config
        
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

    def counterflow_diffusion_flame(self):
        pass

    def free_flame(self):
        pass
    
    def chamber_geometry(self):
        pass

    def reactor_network(self):
        pass

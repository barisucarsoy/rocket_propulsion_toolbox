from src.engine.combustion import Combustion
import src
import numpy as np


class Performance:
    def __init__(self):
        self.adiabatic_gas = Combustion().hp_combustion()  # Cantera gas object of the combustion gas
        self.config_params = src.config_data

        # Set the propellants for the combustion calculations
        self.mdot_total = float(self.config_params["design_point"]["mass_flow_rate"])  # kg/s
        self.combustion_pressure = float(self.config_params["conditions"]["combustion_pressure"])  # Pa
        self.exit_pressure = float(self.config_params["conditions"]["exit_pressure"])  # Pa
        self.d_throat = float(self.config_params["design_point"]["throat_diameter"])
        self.A_throat = np.pi * (self.d_throat / 2) ** 2  # m^2

        # Set the combustion conditions
        self.T0 = self.adiabatic_gas.T
        self.mw = self.adiabatic_gas.mean_molecular_weight  # kg/kmol
        self.Ru = 8.31446261815324  # J/(mol*K)
        self.R = self.Ru / self.mw * 1000  # J/(kg*K)

        self.gamma = self.adiabatic_gas.cp / self.adiabatic_gas.cv

        print("Adiabatic Combustion Gas Properties:")
        print(f"Temperature: {self.T0:.2f} K")
        print(f"Pressure: {self.combustion_pressure:.2f} Pa")
        print(f"Specific Heat Ratio (gamma): {self.gamma:.2f}")
        print(f"Mass Flow Rate: {self.mdot_total:.2f} kg/s")
        print(f"Exit Pressure: {self.exit_pressure:.2f} Pa")
        print(f"Combustion Pressure: {self.combustion_pressure:.2f} Pa")
        print(f"Specific Gas Constant: {self.R:.2f} J/(kg*K)")

    def ideal_performance(self):
        """Calculate the ideal performance of the combustion gas"""

        print("\nIdeal Performance Calculations:")

        # Throat diameter
        characteristic_velocity = np.sqrt(
            ((self.R * self.T0) / self.gamma) * ((self.gamma + 1) / 2) ** ((self.gamma + 1) / (self.gamma - 1))
        )
        print(f"Characteristic Velocity: {characteristic_velocity:.2f} m/s")

        thrust_coefficient = np.sqrt(
            (2 * (self.gamma**2) / (self.gamma - 1))
            * ((2 / (self.gamma + 1)) ** ((self.gamma + 1) / (self.gamma - 1)))
            * (1 - (self.exit_pressure / self.combustion_pressure) ** ((self.gamma - 1) / self.gamma))
        )
        print(f"Thrust Coefficient: {thrust_coefficient:.2f}")

        Thrust = thrust_coefficient * self.mdot_total * characteristic_velocity
        print(f"Thrust: {Thrust:.2f} N")

        Isp = thrust_coefficient * characteristic_velocity / 9.81
        print(f"Isp: {Isp:.2f} s")

        exhaust_velocity = characteristic_velocity * thrust_coefficient
        print(f"Exhaust Velocity: {exhaust_velocity:.2f} m/s")


if __name__ == "__main__":
    # Example usage
    performance_instance = Performance().ideal_performance()

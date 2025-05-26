import cantera as ct

# 1. Setup Gas and Interfaces
gas = ct.Solution("gri30.yaml")

# 2. Define fuel and oxidiser streams
fuel_inlet = ct.Solution(thermo="IdealGas", species=gas.species())
fuel_inlet.TPX = 300.0, 60.0 * ct.one_atm, "CH4:1"

oxidiser_inlet = ct.Solution(thermo="IdealGas", species=gas.species())
oxidiser_inlet.TPX = 90.0, 60.0 * ct.one_atm, "O2:1"  # GOX at cryogenic temp

# 3. Create the 1-D counter-flow diffusion flame object
f = ct.CounterflowDiffusionFlame(gas, width=0.02)

# Copy inlet states
f.fuel_inlet.T = fuel_inlet.T
f.fuel_inlet.X = fuel_inlet.X
f.fuel_inlet.mdot = 0.5  # kg / m² s (example value)

f.oxidizer_inlet.T = oxidiser_inlet.T
f.oxidizer_inlet.X = oxidiser_inlet.X
f.oxidizer_inlet.mdot = 0.5  # kg / m² s

# 4. Set the strain rate and solve
f.solve(loglevel=1, auto=True)

# 5. The solution is now available in f.T, f.Y, f.Z, ...

f.show()

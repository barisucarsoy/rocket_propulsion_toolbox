import cantera as ct
import numpy as np
import math

print(f"Cantera version: {ct.__version__}")

# --- 1. Preliminaries and Parameters ---

# Gas mixture and reaction mechanism
gas = ct.Solution('gri30.yaml')  # GRI-Mech 3.0 for CH4 combustion

# Nozzle Geometry
D_in_mm = 55.0  # Inlet diameter [mm]
D_out_mm = 22.0 # Outlet diameter [mm]
N_segments = 300 # Number of reactor segments
L_segment_mm = 0.1 # Width of each segment [mm]

D_in = D_in_mm / 1000.0  # m
D_out = D_out_mm / 1000.0 # m
L_segment = L_segment_mm / 1000.0 # m
L_nozzle = N_segments * L_segment # Total nozzle length [m]

# Calculate areas for each segment (using area at the midpoint of the segment)
segment_axial_positions_m = np.linspace(L_segment / 2.0, L_nozzle - L_segment / 2.0, N_segments)
diameters_m = D_in - (D_in - D_out) * (segment_axial_positions_m / L_nozzle)
areas_m2 = math.pi * (diameters_m**2) / 4.0

# Inlet Conditions
P_inlet_bar = 12
T_inlet_K = 300.0  # Assuming pre-combustion temperature
OF_ratio = 3.1   # Oxygen-to-Fuel mass ratio

# Assumed mass flow rate (kg/s) - THIS IS A CRITICAL ASSUMPTION
# You might need to adjust this based on expected nozzle performance or a more detailed analysis
m_dot_kg_s = 0.2 # Example value, adjust as needed

# --- 2. Initial State Calculation ---

# CH4 + 2O2 -> CO2 + 2H2O
# MW_CH4 = 16.043 g/mol, MW_O2 = 31.998 g/mol
MW_CH4 = gas.molecular_weights[gas.species_index('CH4')]
MW_O2 = gas.molecular_weights[gas.species_index('O2')]

# Moles of O2 per mole of CH4
# (n_O2 * MW_O2) / (n_CH4 * MW_CH4) = OF_ratio
# n_O2/n_CH4 = OF_ratio * (MW_CH4 / MW_O2)
n_O2_per_n_CH4 = OF_ratio * (MW_CH4 / MW_O2)

initial_comp_dict = {'CH4': 1.0, 'O2': n_O2_per_n_CH4}
# gri30.yaml does not have Ar by default in the species list if not added.
# If you want a diluent like N2 (from air for example), it should be added here.
# For pure CH4/O2, this is fine. If O2 comes from air, add N2: n_O2_per_n_CH4 * (0.79/0.21)

gas.TPX = T_inlet_K, P_inlet_bar * 1e5, initial_comp_dict
gas.equilibrate('HP')  # High-pressure equilibrium to set initial state
initial_density = gas.density
initial_enthalpy_mass = gas.enthalpy_mass
initial_cp_mass = gas.cp_mass
initial_sound_speed = gas.sound_speed

print("--- Inlet Conditions ---")
print(f"Pressure: {gas.P/1e5:.2f} bar")
print(f"Temperature: {gas.T:.2f} K")
print(f"Density: {initial_density:.3f} kg/m^3")
print(f"Sound Speed: {initial_sound_speed:.2f} m/s")
print("Composition (mole fractions):")
for i, species_name in enumerate(gas.species_names):
    if gas.X[i] > 1e-9:
        print(f"  {species_name}: {gas.X[i]:.4f}")
print(f"Mass flow rate (assumed): {m_dot_kg_s:.3f} kg/s")
print(f"Calculated n_O2 / n_CH4 mole ratio: {n_O2_per_n_CH4:.4f}")
stoich_O2_per_CH4 = 2.0
print(f"Stoichiometric n_O2 / n_CH4 mole ratio: {stoich_O2_per_CH4:.4f}")
if n_O2_per_n_CH4 < stoich_O2_per_CH4:
    print("Mixture is FUEL-RICH")
elif n_O2_per_n_CH4 > stoich_O2_per_CH4:
    print("Mixture is FUEL-LEAN")
else:
    print("Mixture is STOICHIOMETRIC")
print("------------------------")

# --- 3. Reactor Simulation Loop (Marching downstream) ---

# Store results
results_axial_pos = [0.0] # Start with inlet at x=0
results_P_Pa = [gas.P]
results_T_K = [gas.T]
results_rho_kg_m3 = [gas.density]
results_u_m_s = [m_dot_kg_s / (gas.density * areas_m2[0])] # Approx inlet velocity to first segment
results_Mach = [results_u_m_s[0] / gas.sound_speed]
results_species_X = [gas.X.copy()] # Store full composition vector

current_P_Pa = gas.P
current_T_K = gas.T
current_comp = gas.X.copy()
current_enthalpy_mass = gas.enthalpy_mass # For energy balance if needed, reactors handle it

print("\n--- Simulating Nozzle Segments ---")
print(f"{'Seg':>3s} | {'X (mm)':>7s} | {'Area (cm2)':>10s} | {'P (bar)':>7s} | {'T (K)':>7s} | {'Rho (kg/m3)':>11s} | {'u (m/s)':>7s} | {'Mach':>5s} | {'CH4':>7s} | {'O2':>7s} | {'CO2':>7s} | {'H2O':>7s}")
print("-" * 130)

# Initial print for inlet conditions relative to first segment
print(f"{'In':>3s} | {0.0:>7.2f} | {areas_m2[0]*1e4:>10.3f} | {current_P_Pa/1e5:>7.2f} | {current_T_K:>7.1f} | {gas.density:>11.3f} | {results_u_m_s[0]:>7.1f} | {results_Mach[0]:>5.3f} | {current_comp[gas.species_index('CH4')]:>7.4f} | {current_comp[gas.species_index('O2')]:>7.4f} | {current_comp[gas.species_index('CO2')]:>7.4f} | {current_comp[gas.species_index('H2O')]:>7.4f}")


for i in range(N_segments):
    segment_area = areas_m2[i]
    segment_volume = segment_area * L_segment
    
    # Set gas state for the beginning of this segment
    gas.TPX = current_T_K, current_P_Pa, current_comp
    rho_upstream = gas.density
    sound_speed_upstream = gas.sound_speed
    
    if rho_upstream < 1e-6: # Avoid division by zero if density is too low
        print(f"Segment {i+1}: Density too low ({rho_upstream:.2e}). Stopping.")
        break
    
    u_upstream = m_dot_kg_s / (rho_upstream * segment_area)
    
    if u_upstream <= 1e-3: # Avoid extremely long residence times if velocity is tiny
        print(f"Segment {i+1}: Upstream velocity too low ({u_upstream:.2e} m/s). Pressure likely too high or m_dot too low. Stopping.")
        break
    
    residence_time_s = L_segment / u_upstream
    
    # Create a reactor for the current segment
    reactor = ct.IdealGasReactor(contents=gas, name=f"segment_{i+1}")
    reactor.volume = segment_volume
    
    # Create a reactor network for this single reactor segment
    # (Needed to advance the reactor state)
    sim_network = ct.ReactorNet([reactor])
    
    # Advance the reactor state over the residence time
    sim_network.advance(reactor.thermo.T + residence_time_s)
    
    # Get outlet state from the reactor
    T_downstream_reactor = reactor.T
    P_reactor_outlet = reactor.thermo.P # Reactor pressure is constant during integration at fixed volume
    rho_downstream_reactor = reactor.thermo.density
    X_downstream_reactor = reactor.thermo.X.copy()
    sound_speed_downstream_reactor = reactor.thermo.sound_speed
    
    # Momentum Equation Application (Simplified for constant area segment `i`)
    # P_downstream_segment = P_upstream_segment - (m_dot/A_segment)^2 * (1/rho_downstream_reactor - 1/rho_upstream)
    # Here, P_upstream_segment is current_P_Pa (inlet P to this segment)
    # P_downstream_segment will be the inlet P for the *next* segment.
    
    # Pressure change due to momentum. Note: This is a strong simplification.
    # If (1/rho_downstream - 1/rho_upstream) is negative (density increases, e.g. cooling), pressure would increase.
    # If positive (density decreases, e.g. heating/expansion), pressure would decrease.
    delta_P_momentum = (m_dot_kg_s**2 / segment_area**2) * (1.0/rho_downstream_reactor - 1.0/rho_upstream)
    P_downstream_segment_outlet = current_P_Pa - delta_P_momentum
    
    # Update state for the next segment's inlet
    current_P_Pa = P_downstream_segment_outlet
    current_T_K = T_downstream_reactor # Temperature changes due to reactions
    current_comp = X_downstream_reactor
    
    # Calculate velocity and Mach number at segment outlet (using outlet density and pressure)
    # For consistency, use density at the new P, T state
    gas.TPX = current_T_K, current_P_Pa, current_comp # Update gas object to downstream state
    rho_final_segment = gas.density # Density at the actual outlet P, T
    u_downstream_segment = m_dot_kg_s / (rho_final_segment * segment_area)
    mach_downstream_segment = u_downstream_segment / gas.sound_speed
    
    # Store results (axial position is end of segment)
    current_axial_pos_m = (i + 1) * L_segment
    results_axial_pos.append(current_axial_pos_m)
    results_P_Pa.append(current_P_Pa)
    results_T_K.append(current_T_K)
    results_rho_kg_m3.append(rho_final_segment)
    results_u_m_s.append(u_downstream_segment)
    results_Mach.append(mach_downstream_segment)
    results_species_X.append(current_comp)
    
    # Print segment outlet info
    print(f"{i+1:>3d} | {current_axial_pos_m*1000:>7.2f} | {segment_area*1e4:>10.3f} | {current_P_Pa/1e5:>7.2f} | {current_T_K:>7.1f} | {rho_final_segment:>11.3f} | {u_downstream_segment:>7.1f} | {mach_downstream_segment:>5.3f} | {current_comp[gas.species_index('CH4')]:>7.4f} | {current_comp[gas.species_index('O2')]:>7.4f} | {current_comp[gas.species_index('CO2')]:>7.4f} | {current_comp[gas.species_index('H2O')]:>7.4f}")
    
    # Sanity checks
    if current_P_Pa <= 0:
        print(f"Segment {i+1}: Pressure became non-positive ({current_P_Pa:.2e} Pa). Stopping.")
        break
    if current_T_K <= 0:
        print(f"Segment {i+1}: Temperature became non-positive ({current_T_K:.2e} K). Stopping.")
        break
    if mach_downstream_segment >= 1.0:
        print(f"Segment {i+1}: Flow became sonic (Mach = {mach_downstream_segment:.3f}). Further calculations with this model might be less accurate.")
        # For a converging nozzle, M=1 at exit is expected if choked.
        if i == N_segments -1:
            print("Choked flow at nozzle exit predicted.")
        # else: # If sonic internally, this simplified model has limitations.
        # print("Sonic condition reached internally. Model limitations apply.")


print("-" * 130)
print("Simulation finished.")

# --- 4. Post-processing and Output (Example: Plotting) ---
try:
    import matplotlib.pyplot as plt
    
    fig, axs = plt.subplots(4, 1, sharex=True, figsize=(10, 12))
    
    # Pressure
    axs[0].plot(np.array(results_axial_pos) * 1000, np.array(results_P_Pa) / 1e5)
    axs[0].set_ylabel('Pressure (bar)')
    axs[0].grid(True)
    
    # Temperature
    axs[1].plot(np.array(results_axial_pos) * 1000, results_T_K)
    axs[1].set_ylabel('Temperature (K)')
    axs[1].grid(True)
    
    # Velocity and Mach Number
    ax_vel = axs[2]
    ax_mach = ax_vel.twinx()
    
    p1, = ax_vel.plot(np.array(results_axial_pos) * 1000, results_u_m_s, color='blue', label='Velocity')
    ax_vel.set_ylabel('Velocity (m/s)')
    ax_vel.yaxis.label.set_color(p1.get_color())
    ax_vel.tick_params(axis='y', colors=p1.get_color())
    
    p2, = ax_mach.plot(np.array(results_axial_pos) * 1000, results_Mach, color='red', label='Mach No.')
    ax_mach.set_ylabel('Mach Number')
    ax_mach.yaxis.label.set_color(p2.get_color())
    ax_mach.tick_params(axis='y', colors=p2.get_color())
    ax_mach.grid(True, linestyle='--')
    
    lines = [p1, p2]
    ax_vel.legend(lines, [l.get_label() for l in lines])
    
    
    # Key Species (CH4, O2, CO2, H2O)
    idx_ch4 = gas.species_index('CH4')
    idx_o2 = gas.species_index('O2')
    idx_co2 = gas.species_index('CO2')
    idx_h2o = gas.species_index('H2O')
    
    species_data = np.array(results_species_X)
    axs[3].plot(np.array(results_axial_pos) * 1000, species_data[:, idx_ch4], label='CH4')
    axs[3].plot(np.array(results_axial_pos) * 1000, species_data[:, idx_o2], label='O2')
    axs[3].plot(np.array(results_axial_pos) * 1000, species_data[:, idx_co2], label='CO2')
    axs[3].plot(np.array(results_axial_pos) * 1000, species_data[:, idx_h2o], label='H2O')
    axs[3].set_ylabel('Mole Fraction')
    axs[3].set_xlabel('Axial Position (mm)')
    axs[3].legend()
    axs[3].grid(True)
    axs[3].set_ylim(0,1)
    
    
    fig.suptitle(f'Nozzle Simulation (Din={D_in_mm}mm, Dout={D_out_mm}mm, OF={OF_ratio}, m_dot={m_dot_kg_s}kg/s)', fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

except ImportError:
    print("\nMatplotlib not found. Skipping plots. Install with 'pip install matplotlib'")
except Exception as e:
    print(f"\nError during plotting: {e}")

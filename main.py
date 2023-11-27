import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from determination_of_max_velocity_point import generate_max_velocity_point_graph

# Function to calculate omega_n from settling time (ts) and zeta
def calculate_omega_n(zeta, ts):
    if zeta == 1:
        return 6 / ts
    elif zeta == 0.7:
        return 5.7 / ts
    else:
        raise ValueError("Invalid value of zeta. Please use either 0.7 or 1.")

# Function to calculate maximum velocity (v_max) based on zeta, x_max, and omega_n
def calculate_max_velocity(zeta, x_max, omega_n):
    if zeta == 0.7:
        return 0.4668 * x_max * omega_n
    elif zeta == 1:
        return 0.3678 * x_max * omega_n
    else:
        raise ValueError("Invalid value of zeta. Please use either 0.7 or 1.")

# Function to calculate minimum pa/ps based on b and ra
def calculate_pa_ps_min(b, ra):
    discriminant = b**2 * ra**2 - 4 * b + 1 + 5 * b**2 - 2 * b**3 - 2 * b * ra**2 + ra**2
    numerator = b * ra + np.sqrt(discriminant) * ra
    denominator = 1 - 2 * b + b**2 + ra**2
    return numerator / denominator

# Function to calculate vess/C based on parameters
def calculate_vess_over_C(pa_ps, ps, p0, b, ra, mu, Fl):
    term1 = ps * p0 * np.sqrt(1 - ((pa_ps - b) / (1 - b))**2)
    term2 = (pa_ps - (p0 / ps) * (1 - ra) - mu - ((p0 / ps) / (b / ra + np.sqrt((b**2 - 2 * b + 1) / (ra**2) + 1 + pa_ps ** -1 * (-2 * b + pa_ps ** -1 * (2 * b - 1))))))
    return term1 * term2 / (pa_ps * Fl)

# Function to calculate (pa/ps)Av based on (pa/ps)vmax
def calculate_pa_ps_Av(pa_ps_vmax):
    return (1 + pa_ps_vmax) / 2

# Function to calculate C based on (pa/ps)Av, Aa, vmax, p0, and b
def calculate_C(pa_ps_Av, Aa, vmax, p0, b):
    numerator = pa_ps_Av * Aa * vmax
    denominator = p0 * np.sqrt(1 - ((pa_ps_Av - b) / (1 - b))**2)
    return numerator / denominator

# Function to calculate p0/pB based on ps/pa_max, b, and ra
def calculate_p0_over_pB(pa_ps_max, b, ra):
    term1 = b + np.sqrt(b**2 - 2 * b + 1 + ra**2 + (2 * b * ra**2 - ra**2)*(pa_ps_max)**-2 - 2 * b * ra**2*(pa_ps_max**-1))
    return term1

# Function to calculate Lr based on pb/p0, p0/ps, ra, p0/ps, pa/ps at vmax
def calculate_Lr(pb_p0, ps, p0, ra, pa_ps_vmax):
    return pa_ps_vmax - (p0/ps) * (1 - ra) - (p0/ps) * ra * (pb_p0) ** -1

# Function to calculate Aa based on Fl, ps, Lr_opt, and mu
def calculate_Aa(Fl, ps, Lr_opt, mu):
    return Fl / (ps * (Lr_opt - mu))

# Set parameters
zeta_value = 0.7
x_max = 0.0102
b_value = 0.3
ra_value = 1
mu_value = 0.01
Fl_value = 26.16e3
ps_value = 10e5
p0_value = 1e5
settling_time = 1.25

# Step 1: Calculate omega_n from settling time (ts) and zeta
omega_n = calculate_omega_n(zeta_value, settling_time)

# Step 2: Calculate vmax from omega_n and x_max
vmax = calculate_max_velocity(zeta_value, x_max, omega_n)

# Step 3: Calculate pa/ps_min from b and ra
pa_ps_min = calculate_pa_ps_min(b_value, ra_value)

# Step 4: Plot vess/C vs. pa/ps and find pa/ps_vmax
pa_ps_values = np.linspace(pa_ps_min, 1, 100)
vess_over_C_values = [calculate_vess_over_C(pa_ps, ps_value, p0_value, b_value, ra_value, mu_value, Fl_value) for pa_ps in pa_ps_values]

pa_ps_vmax_index = np.argmax(vess_over_C_values)
pa_ps_vmax = pa_ps_values[pa_ps_vmax_index]

# Step 5: Calculate p0/pB
pb_p0_value = calculate_p0_over_pB(pa_ps_vmax, b_value, ra_value)

# Step 6: Calculate Lr_opt
Lr_opt_value = calculate_Lr(pb_p0_value, ps_value, p0_value, ra_value, pa_ps_vmax)

# Step 7: Calculate Aa
Aa_result = calculate_Aa(Fl_value, ps_value, Lr_opt_value, mu_value)

# Step 8: Calculate (pa/ps)Av from (pa/ps)vmax
pa_ps_Av = calculate_pa_ps_Av(pa_ps_vmax)

# Step 9: Calculate C from (pa/ps)Av, Aa, vmax, p0, and b
C_value = calculate_C(pa_ps_Av, Aa_result, vmax, p0_value, b_value)

# Print results
print("Results:")
print("Settling Time (ts):", settling_time)
print("Calculated omega_n:", omega_n)
print("Maximum Velocity (v_max):", vmax)
print("pa/ps_min:", pa_ps_min)
print("pa/ps_vmax:", pa_ps_vmax)
print("(pa/ps)Av:", pa_ps_Av)
print("C:", C_value)
print("p0/pB:", pb_p0_value)
print("Lr_opt:", Lr_opt_value)
print("Aa:", Aa_result)

# Plot vess/C vs. pa/ps
plt.figure(figsize=(8, 6))
plt.plot(pa_ps_values, vess_over_C_values, label=r'$\frac{v_{\mathrm{ess}}}{C}$ vs. $\frac{p_A}{p_S}$')
plt.scatter([pa_ps_vmax], [vess_over_C_values[pa_ps_vmax_index]], color='red', marker='o', label=r'Maximum Value at $\frac{p_A}{p_S}_{\mathrm{vmax}}$')
plt.xlabel(r'$\frac{p_A}{p_S}$')
plt.ylabel(r'$\frac{v_{\mathrm{ess}}}{C}$')
plt.title(r'$\frac{v_{\mathrm{ess}}}{C}$ vs. $\frac{p_A}{p_S}$')
plt.legend()
plt.grid(True)
plt.show()


# Calculate p0/pB for the entire range of pa/ps_values
pb_p0_values = [calculate_p0_over_pB(pa_ps, b_value, ra_value) for pa_ps in pa_ps_values]

generate_max_velocity_point_graph(pa_ps_values, pa_ps_vmax, pb_p0_values, vess_over_C_values, pa_ps_vmax_index)

from scipy.optimize import differential_evolution

# Initialize lists to store information during optimization
iteration_data = []

def objective_function(ps, *args):
    # Unpack the arguments
    target_C, p0, ra, b, Fl, mu, xmax, wn, zeta = args
    
    # Calculate pa/ps_vmax
    pa_ps_min = calculate_pa_ps_min(b, ra)
    pa_ps_values = np.linspace(pa_ps_min, 1, 100)
    vess_over_C_values = [calculate_vess_over_C(pa_ps, ps, p0, b, ra, mu, Fl) for pa_ps in pa_ps_values]
    pa_ps_vmax_index = np.argmax(vess_over_C_values)
    pa_ps_vmax = pa_ps_values[pa_ps_vmax_index]

    # Calculate C
    pa_ps_Av = calculate_pa_ps_Av(pa_ps_vmax)
    pb_p0_value = calculate_p0_over_pB(pa_ps_vmax, b, ra)
    Lr_opt_value = calculate_Lr(pb_p0_value, ps, p0, ra, pa_ps_vmax)
    vmax = calculate_max_velocity(zeta, xmax, wn)
    Aa_result = calculate_Aa(Fl, ps, Lr_opt_value, mu)

    # Calculate C
    C = calculate_C(pa_ps_Av, Aa_result, vmax, p0, b)

    # Constraint check: C must not be < 75% or > 125% of the original C
    if C < 0.75 * target_C or C > 1.25 * target_C:
        penalty = float('inf')  # Penalize this solution heavily
    else:
        # Incentivize higher ps and lower area
        ps_incentive = (ps - ps_min) / (ps_max - ps_min)
        area_penalty = Aa_result / Aa_initial
    
        # Combine into a single penalty function
        penalty = abs(C - target_C) - ps_incentive + area_penalty
    
    # Store information for visualization
    iteration_data.append({
        'ps': ps,
        'Aa': Aa_result,
        'C': C,
    })

    return penalty

# Set initial parameters for optimization
ps_min = 1.1e5  # Lower bound for supply pressure (1.1 bar)
ps_max = 40e5   # Upper bound for supply pressure (40 bar)
Aa_initial = Aa_result  # Initial area for normalization

# Arguments to pass to the objective function
target_C = C_value
args = (target_C, p0_value, ra_value, b_value, Fl_value, mu_value, x_max, omega_n, zeta_value)

# Perform the differential evolution algorithm
# Perform the optimization
result = differential_evolution(
    objective_function,
    bounds=[(ps_min, ps_max)],
    args=args,
    maxiter=1000,
    tol=1e-6
)

# Process the optimization result
if result.success:
    optimal_ps = result.x[0]
    optimal_Aa = calculate_Aa(Fl_value, optimal_ps, Lr_opt_value, mu_value)
    print(f"Found optimal ps: {optimal_ps / 1e5} bar")
    print(f"Corresponding optimal actuator area Aa: {optimal_Aa} square meters")
else:
    print("Optimization did not converge.")

# Visualize the optimization process, showing only the top 100 iterations
top_iterations = 100
iteration_numbers = list(range(1, len(iteration_data) + 1))

# Trim data to the top iterations
trimmed_iteration_data = iteration_data[:top_iterations]

# Plot all variables in one subplot with manual layout adjustments
plt.figure(figsize=(15, 12))

# Plot supply pressures over iterations
plt.subplot(3, 1, 1)
plt.plot(iteration_numbers[:top_iterations], [iteration['ps'] for iteration in trimmed_iteration_data], marker='o', linestyle='-', color='b', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Supply Pressure (ps)')
plt.title('Supply Pressure Evolution')

# Plot actuator area over iterations with logarithmic scale
plt.subplot(3, 1, 2)
plt.plot(iteration_numbers[:top_iterations], [iteration['Aa'] for iteration in trimmed_iteration_data], marker='o', linestyle='-', color='g', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('Actuator Area (Aa)')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('Actuator Area Evolution')

# Plot C over iterations with logarithmic scale
plt.subplot(3, 1, 3)
plt.plot(iteration_numbers[:top_iterations], [iteration['C'] for iteration in trimmed_iteration_data], marker='o', linestyle='-', color='r', alpha=0.7)
plt.xlabel('Iteration')
plt.ylabel('C')
plt.yscale('log')  # Set y-axis to logarithmic scale
plt.title('C Evolution')

# Adjust the layout
plt.subplots_adjust(hspace=1)  # Adjust the vertical spacing between subplots

# Show the figure
plt.show()

# Define the number of steps for each range
num_steps = 10

# Define the ranges
ps_values = np.linspace(1.1e5, 20e5, num_steps)
Fl_values = np.linspace(1e3, 50e3, num_steps)
settling_times = np.linspace(0.25, 5, num_steps)

# Initialize arrays to store results
C_results = np.zeros((num_steps, num_steps, num_steps))
Aa_results = np.zeros((num_steps, num_steps, num_steps))

pa_ps_min = calculate_pa_ps_min(b_value, ra_value)
pa_ps_values = np.linspace(pa_ps_min, 1, 100)

# Nested loops to vary each parameter
for i, ps in enumerate(ps_values):
    for j, Fl in enumerate(Fl_values):
        for k, st in enumerate(settling_times):
            omega_n = calculate_omega_n(zeta_value, st)
            vmax = calculate_max_velocity(zeta_value, x_max, omega_n)
            vess_over_C_values = [calculate_vess_over_C(pa_ps, ps, p0_value, b_value, ra_value, mu_value, Fl) for pa_ps in pa_ps_values]
            pa_ps_vmax_index = np.argmax(vess_over_C_values)
            pa_ps_vmax = pa_ps_values[pa_ps_vmax_index]
            pb_p0_value = calculate_p0_over_pB(pa_ps_vmax, b_value, ra_value)
            Lr_opt_value = calculate_Lr(pb_p0_value, ps, p0_value, ra_value, pa_ps_vmax)
            Aa_result = calculate_Aa(Fl, ps, Lr_opt_value, mu_value)
            pa_ps_Av = calculate_pa_ps_Av(pa_ps_vmax)
            C_value = calculate_C(pa_ps_Av, Aa_result, vmax, p0_value, b_value)
            C_results[i, j, k] = C_value
            Aa_results[i, j, k] = Aa_result

# Plotting function for subplots
plt.figure(figsize=(12, 18))

# Plot for Aa_result and C_value as a function of ps_value
plt.subplot(3, 2, 1)
plt.semilogy(ps_values, Aa_results[:, 0, 0], label='Aa vs ps')
plt.xlabel('Supply Pressure (ps)')
plt.ylabel('Aa Result')
plt.title('Aa vs Supply Pressure')
plt.grid(True)

plt.subplot(3, 2, 2)
plt.semilogy(ps_values, C_results[:, 0, 0], label='C vs ps')
plt.xlabel('Supply Pressure (ps)')
plt.ylabel('C Value')
plt.title('C vs Supply Pressure')
plt.grid(True)

# Plot for Aa_result and C_value as a function of Fl_value
plt.subplot(3, 2, 3)
plt.semilogy(Fl_values, Aa_results[0, :, 0], label='Aa vs Fl')
plt.xlabel('Load Force (Fl)')
plt.ylabel('Aa Result')
plt.title('Aa vs Load Force')
plt.grid(True)

plt.subplot(3, 2, 4)
plt.semilogy(Fl_values, C_results[0, :, 0], label='C vs Fl')
plt.xlabel('Load Force (Fl)')
plt.ylabel('C Value')
plt.title('C vs Load Force')
plt.grid(True)

# Plot for Aa_result and C_value as a function of settling_time
plt.subplot(3, 2, 5)
plt.semilogy(settling_times, Aa_results[0, 0, :], label='Aa vs ts')
plt.xlabel('Settling Time (ts)')
plt.ylabel('Aa Result')
plt.title('Aa vs Settling Time')
plt.grid(True)

plt.subplot(3, 2, 6)
plt.semilogy(settling_times, C_results[0, 0, :], label='C vs ts')
plt.xlabel('Settling Time (ts)')
plt.ylabel('C Value')
plt.title('C vs Settling Time')
plt.grid(True)

# Adjust the layout
plt.subplots_adjust(top=0.9, bottom=0.085, hspace=0.5)

plt.show()

# Function to compute outputs
def compute_outputs(ps, ts, Fl):
    # Calculate omega_n
    omega_n = calculate_omega_n(zeta_value, ts)

    # Calculate pa/ps_vmax
    pa_ps_min = calculate_pa_ps_min(b_value, ra_value)
    pa_ps_values = np.linspace(pa_ps_min, 1, 100)
    vess_over_C_values = [calculate_vess_over_C(pa_ps, ps, p0_value, b_value, ra_value, mu_value, Fl) for pa_ps in pa_ps_values]
    pa_ps_vmax_index = np.argmax(vess_over_C_values)
    pa_ps_vmax = pa_ps_values[pa_ps_vmax_index]

    # Calculate vmax, pb_p0, Lr_opt, Aa, and C
    vmax = calculate_max_velocity(zeta_value, x_max, omega_n)
    pb_p0 = calculate_p0_over_pB(pa_ps_vmax, b_value, ra_value)
    Lr_opt = calculate_Lr(pb_p0, ps, p0_value, ra_value, pa_ps_vmax)
    Aa = calculate_Aa(Fl, ps, Lr_opt, mu_value)
    C = calculate_C(pa_ps_Av, Aa, vmax, p0_value, b_value)

    return Aa, C

# Define the sets of values and the perturbation percentage
ts_values = [1.25, 2.5, 5]
Fl_values = [13e3, 25e3, 40e3]
ps_values = [5e5, 8e5, 10e5]
delta_percentage = 0.25

# Initialize a list to store the results
results = []

# Your function definitions: compute_outputs, etc.

# Iterate through each set of values
for ts in ts_values:
    for Fl in Fl_values:
        for ps in ps_values:
            # Baseline calculations
            baseline_Aa, baseline_C = compute_outputs(ps, ts, Fl)

            # Apply perturbations and calculate sensitivities
            for param, value in [('ts', ts), ('Fl', Fl), ('ps', ps)]:
                delta = value * delta_percentage
                perturbed_Aa, perturbed_C = compute_outputs(
                    ps + (delta if param == 'ps' else 0),
                    ts + (delta if param == 'ts' else 0),
                    Fl + (delta if param == 'Fl' else 0)
                )

                sensitivity_Aa = (perturbed_Aa - baseline_Aa) / delta
                sensitivity_C = (perturbed_C - baseline_C) / delta

                # Store the results with adjusted values and units
                results.append({
                    'Tested Variable': param,
                    'Delta Value': delta,
                    'ps Value [bar]': ps / 1e5,  # Convert Pascals to bar
                    'ts Value [s]': ts,  # Seconds
                    'Fl Value [kN]': Fl / 1e3,  # Convert Newtons to kiloNewtons
                    'Sensitivity to C': sensitivity_C,
                    'Sensitivity to A': sensitivity_Aa
                })

# Convert the results list to a DataFrame
sensitivity_df = pd.DataFrame(results)

print(sensitivity_df)

sensitivity_df.to_csv('data/sensitivity_analysis_results.csv', index=False)

import numpy as np
import matplotlib.pyplot as plt
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

# Visualize the optimization process
iteration_numbers = list(range(1, len(iteration_data) + 1))

# Plot all variables in one subplot with manual layout adjustments
plt.figure(figsize=(15, 12))

# Plot supply pressures over iterations
plt.subplot(3, 1, 1)
plt.plot(iteration_numbers, [iteration['ps'] for iteration in iteration_data], marker='o')
plt.xlabel('Iteration')
plt.ylabel('Supply Pressure (ps)')
plt.title('Supply Pressure Evolution')

# Plot actuator area over iterations
plt.subplot(3, 1, 2)
plt.plot(iteration_numbers, [iteration['Aa'] for iteration in iteration_data], marker='o')
plt.xlabel('Iteration')
plt.ylabel('Actuator Area (Aa)')
plt.title('Actuator Area Evolution')

# Plot C over iterations
plt.subplot(3, 1, 3)
plt.plot(iteration_numbers, [iteration['C'] for iteration in iteration_data], marker='o')
plt.xlabel('Iteration')
plt.ylabel('C')
plt.title('C Evolution')

# Adjust the layout
plt.subplots_adjust(hspace=1)  # Adjust the vertical spacing between subplots

# Show the figure
plt.show()

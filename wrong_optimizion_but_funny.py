# Function to find the optimal supply pressure (ps)
def find_optimal_ps(target_C, acceptable_range, p0, ra, b, Fl, mu, xmax, wn, zeta):
    # Initialize lists to store the iteration values
    ps_values = []
    area_values = []

    # Binary search to find the optimal ps
    ps_min = 1.1e5
    ps_max = 15e5
    epsilon = 1e-6  # Tolerance for convergence
    
    while ps_max - ps_min > epsilon:
        mid_ps = (ps_min + ps_max) / 2

        ps_values.append(mid_ps)
        
        # Calculate pa/ps_vmax
        pa_ps_min = calculate_pa_ps_min(b, ra)
        pa_ps_values = np.linspace(pa_ps_min, 1, 100)
        vess_over_C_values = [calculate_vess_over_C(pa_ps, mid_ps, p0, b, ra, mu, Fl) for pa_ps in pa_ps_values]
        pa_ps_vmax_index = np.argmax(vess_over_C_values)
        pa_ps_vmax = pa_ps_values[pa_ps_vmax_index]

        # Calculate C
        pa_ps_Av = calculate_pa_ps_Av(pa_ps_vmax)
        pb_p0_value = calculate_p0_over_pB(pa_ps_vmax, b, ra)
        Lr_opt_value = calculate_Lr(pb_p0_value, mid_ps, p0, ra, pa_ps_vmax)
        vmax = calculate_max_velocity(zeta, xmax, wn)
        Aa_result = calculate_Aa(Fl, mid_ps, Lr_opt_value, mu)

        area_values.append(Aa_result)

        C = calculate_C(pa_ps_Av, Aa_result, vmax, p0, b)
        
        # Check if C is within the acceptable range
        if C < target_C - acceptable_range:
            ps_min = mid_ps
        elif C > target_C + acceptable_range:
            ps_max = mid_ps
        else:
            break
    return mid_ps, Aa_result, ps_values, area_values

# Specify the acceptable range for C
target_C = C_value
acceptable_range = 0.15 * target_C

# Find the optimal ps
optimal_ps, optimal_area, ps_iterations, area_iterations = find_optimal_ps(target_C, acceptable_range, p0_value, ra_value, b_value, Fl_value, mu_value, x_max, omega_n, zeta_value)

# Plot the results
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(ps_iterations, 'b-o', label='Supply Pressure (ps)')
plt.xlabel('Iteration')
plt.ylabel('Supply Pressure (ps)')
plt.title('Convergence of Supply Pressure')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(area_iterations, 'r-o', label='Actuator Area (Aa)')
plt.xlabel('Iteration')
plt.ylabel('Actuator Area (Aa)')
plt.title('Convergence of Actuator Area')
plt.legend()

plt.tight_layout()
plt.show()


print("Optimal pressure:", optimal_ps)
print("Optimal area:", optimal_area)

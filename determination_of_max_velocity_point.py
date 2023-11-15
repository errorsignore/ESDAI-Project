import numpy as np
import matplotlib.pyplot as plt

def generate_max_velocity_point_graph(pa_ps_values, pa_ps_vmax, pb_p0_values, vess_over_C_values, pa_ps_vmax_index):
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot Op curve on the left y-axis
    color = 'tab:blue'
    ax1.set_xlabel(r'$\frac{p_A}{p_S}$')
    ax1.set_ylabel(r'$\frac{p_0}{p_B}$', color=color)
    op_curve, = ax1.plot(pa_ps_values, pb_p0_values, color=color, label='Op curve')
    ax1.tick_params(axis='y', labelcolor=color)

    # Create a point at the maximum pa/ps value
    ax1.scatter(pa_ps_vmax, pb_p0_values[pa_ps_vmax_index], color='red', marker='o')
    ax1.text(pa_ps_vmax, pb_p0_values[pa_ps_vmax_index], f'Max\n({pa_ps_vmax:.2f}, {pb_p0_values[pa_ps_vmax_index]:.2f})', fontsize=9,
             verticalalignment='bottom', horizontalalignment='right')

    # Instantiate a second axes that shares the same x-axis
    ax2 = ax1.twinx()  

    # Plot Velocity/C curve on the right y-axis
    color = 'tab:green'
    ax2.set_ylabel(r'$\frac{v_{\mathrm{ess}}}{C}$', color=color)
    velocity_curve, = ax2.plot(pa_ps_values, vess_over_C_values, color=color, label='Velocity/C curve')
    ax2.tick_params(axis='y', labelcolor=color)

    # List of Line2D objects representing the plotted data
    curves = [op_curve, velocity_curve]

    # Manually adjust the legend position to be outside the upper left of the plot area
    ax1.legend(curves, [curve.get_label() for curve in curves], loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)

    # Display the title
    plt.title('Determination of the Maximum Velocity Point')

    # Adjust layout for better spacing
    plt.tight_layout()  
    plt.show()

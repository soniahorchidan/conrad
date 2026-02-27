import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
from scipy.optimize import fsolve

# --- Configuration ---
ETA = 0.8  # eta (risk budget)

# Fixed lambda values (eta line intersects CDF at these points)
lambda_1 = 0.82
lambda_2 = 0.77
lambda_3 = 0.91

# Generate x-axis (Score space)
x = np.linspace(0, 1, 1000)

# --- Define Distributions ---
# Solve for Beta(a, b) params so CDF(lambda) = eta for each hop.
# Fix b and solve for a to get different curve shapes (uncertain -> certain).
def solve_beta_a(lam, eta_val, b_fixed=2):
    """Find a such that Beta(a, b_fixed).cdf(lam) = eta_val."""
    def eq(a):
        return beta.cdf(lam, a, b_fixed) - eta_val
    return fsolve(eq, 3.0)[0]

a1 = solve_beta_a(lambda_1, ETA)
a2 = solve_beta_a(lambda_2, ETA)
a3 = solve_beta_a(lambda_3, ETA)

# Hop 1: CDF crosses eta at lambda_1=0.82
cdf_hop1 = beta.cdf(x, a1, 2)

# Hop 2: CDF crosses eta at lambda_2=0.77
cdf_hop2 = beta.cdf(x, a2, 2)

# Hop 3: CDF crosses eta at lambda_3=0.91
cdf_hop3 = beta.cdf(x, a3, 2)

# --- Plotting Function ---
def plot_cdf_figure(x_data, y_data, threshold_val, global_val, title, color_line, filename):
    fig, ax = plt.subplots(figsize=(3, 1.5)) 
    
    # Plot CDF
    ax.plot(x_data, y_data, color=color_line, lw=3)
    ax.fill_between(x_data, y_data, alpha=0.1, color=color_line)

    # Eta line (Horizontal) - intersects CDF at (threshold_val, global_val)
    ax.axhline(y=global_val, color='gray', linestyle='--', linewidth=1.5)

    # Derived Threshold (Vertical)
    ax.axvline(x=threshold_val, ymax=global_val, color='black', linestyle=':', linewidth=2)

    # Intersection Dot - where eta line meets CDF (on the curve by construction)
    ax.plot(threshold_val, global_val, 'o', color='black', markersize=6)
    
    # Annotations
    ax.text(0.05, global_val + 0.04, r'$\eta = ' + str(global_val) + '$', 
            fontsize=10, color='gray', fontweight='bold')
    
    # Extract hop number safely
    if "." in filename:
        hop_num = filename.split('.')[0][-1]
    else:
        hop_num = filename[-1]

    # Dynamic text positioning
    text_offset = -0.4 if threshold_val > 0.75 else 0.05
    ax.annotate(r'$\lambda_{} = {:.2f}$'.format(hop_num, threshold_val), 
                xy=(threshold_val, 0), xytext=(threshold_val + text_offset, 0.2),
                fontsize=9, fontweight='bold', color='black',
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

    # Styling
    # ax.set_title(title, fontsize=11, pad=10)
    ax.set_ylabel("CDF", fontsize=10)
    ax.set_xlabel("Score $s$", fontsize=10)
    
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

# --- Generate Figures ---

# Figure 1: Uncertain (Loose)
plot_cdf_figure(x, cdf_hop1, lambda_1, ETA,
    "Hop 1: Uncertain Model\n(Gradual CDF $\\to$ Loose $\\lambda$)", 
    '#ff7f0e', "hop1.png") # Orange for caution

# Figure 2: Intermediate
plot_cdf_figure(x, cdf_hop2, lambda_2, ETA,
    "Hop 2: Moderate Confidence\n(Steeper CDF $\\to$ Medium $\\lambda$)", 
    '#1f77b4', "hop2.png") # Blue for standard

# Figure 3: Strict (Certain)
plot_cdf_figure(x, cdf_hop3, lambda_3, ETA,
    "Hop 3: High Confidence\n(Sharp CDF $\\to$ Strict $\\lambda$)", 
    '#2ca02c', "hop3.png") # Green for good
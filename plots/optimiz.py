import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# --- Configuration ---
GLOBAL_LAMBDA = 0.8  # The global quantile (Risk Budget proxy)

# Generate x-axis (Score space)
x = np.linspace(0, 1, 1000)

# --- Define Distributions (Updated Logic) ---

# Hop 1: Uncertain / Noisy
# Beta(2, 2) is symmetric and spread out.
# The CDF rises gradually, meaning to cover 80% of data, you need a lower threshold.
cdf_hop1 = beta.cdf(x, 2, 2)

# Hop 2: Bit more certain
# Beta(5, 2) shifts the mass to the right (mean ~0.71).
# The model is confident, but not perfect.
cdf_hop2 = beta.cdf(x, 5, 2)

# Hop 3: Very Certain
# Beta(20, 2) is extremely peaked near 1.0.
# The CDF stays flat until the very end, pushing the threshold very high.
cdf_hop3 = beta.cdf(x, 20, 2)

# --- Helper to find Lambda ---
def get_lambda(cdf_arr, global_val):
    idx = np.searchsorted(cdf_arr, global_val)
    return x[idx]

lambda_1 = get_lambda(cdf_hop1, GLOBAL_LAMBDA)
lambda_2 = get_lambda(cdf_hop2, GLOBAL_LAMBDA)
lambda_3 = get_lambda(cdf_hop3, GLOBAL_LAMBDA)

# --- Plotting Function ---
def plot_cdf_figure(x_data, y_data, threshold_val, global_val, title, color_line, filename):
    fig, ax = plt.subplots(figsize=(3, 1.5)) 
    
    # Plot CDF
    ax.plot(x_data, y_data, color=color_line, lw=3)
    ax.fill_between(x_data, y_data, alpha=0.1, color=color_line)

    # Global Line (Horizontal)
    ax.axhline(y=global_val, color='gray', linestyle='--', linewidth=1.5)
    
    # Derived Threshold (Vertical)
    ax.axvline(x=threshold_val, ymax=global_val, color='black', linestyle=':', linewidth=2)
    
    # Intersection Dot
    ax.plot(threshold_val, global_val, 'o', color='black', markersize=6)
    
    # Annotations
    ax.text(0.05, global_val + 0.04, r'$\Lambda = ' + str(global_val) + '$', 
            fontsize=10, color='gray', fontweight='bold')
    
    # Extract hop number safely
    if "." in filename:
        hop_num = filename.split('.')[0][-1]
    else:
        hop_num = filename[-1]

    # Dynamic text positioning
    text_offset = -0.4 if threshold_val > 0.85 else 0.05
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
plot_cdf_figure(x, cdf_hop1, lambda_1, GLOBAL_LAMBDA,
    "Hop 1: Uncertain Model\n(Gradual CDF $\\to$ Loose $\\lambda$)", 
    '#ff7f0e', "hop1.png") # Orange for caution

# Figure 2: Intermediate
plot_cdf_figure(x, cdf_hop2, lambda_2, GLOBAL_LAMBDA,
    "Hop 2: Moderate Confidence\n(Steeper CDF $\\to$ Medium $\\lambda$)", 
    '#1f77b4', "hop2.png") # Blue for standard

# Figure 3: Strict (Certain)
plot_cdf_figure(x, cdf_hop3, lambda_3, GLOBAL_LAMBDA,
    "Hop 3: High Confidence\n(Sharp CDF $\\to$ Strict $\\lambda$)", 
    '#2ca02c', "hop3.png") # Green for good
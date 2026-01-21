import matplotlib.pyplot as plt
import numpy as np

# Configuration for the 3x3 plot
datasets = ["FB15k-237", "NELL-995", "YAGO3-10"]
sparsities = ["5% Missing Data", "20% Missing Data", "40% Missing Data"]
confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]

# --- FB15k-237 Data (Row 1) ---
fb_5_3p = [0.5175, 0.6193, 0.7177, 0.8085, 0.908]
fb_5_2u = [0.5109, 0.6144, 0.7131, 0.8135, 0.9118]
fb_5_2ip = [0.4989, 0.6277, 0.7117, 0.7998, 0.9117]

fb_20_3p = [0.5164, 0.5941, 0.717, 0.8047, 0.9052]
fb_20_2u = [0.5165, 0.6243, 0.7219, 0.8089, 0.8972]
fb_20_2ip = [0.506, 0.5704, 0.6769, 0.7893, 0.8951]

fb_40_3p = [0.5091, 0.6086, 0.7406, 0.871] # 4 points
fb_40_2u = [0.5021, 0.5985, 0.6909, 0.8036, 0.9062]
fb_40_2ip = [0.5444, 0.6286] # OOM on 0.7

# --- NELL-995 Data (Row 2) ---
nell_5_3p = [0.5184, 0.6057, 0.7648, 0.8182] # OOM on 0.9
nell_5_2u = [0.5228, 0.6255, 0.7227, 0.8084, 0.9085]

nell_20_3p = [0.5223, 0.5936, 0.6925, 0.7906, 0.9017]
nell_20_2u = [0.5151, 0.6054, 0.7129, 0.815, 0.893]

nell_40_2u = [0.5208, 0.6207, 0.722, 0.8205, 0.9004]

# --- Plotting Setup ---
fig, axes = plt.subplots(3, 3, figsize=(8, 5.5), sharex=True, sharey=True)

# Helper to plot data safely
def plot_panel(ax, row_data, label, color, marker, confs=confidence_levels):
    if row_data:
        # Match lengths in case of missing points/OOMs
        ax.plot(confs[:len(row_data)], row_data, marker=marker, 
                label=label, color=color, linewidth=2.5, markersize=7)

for r in range(3): # Datasets
    for c in range(3): # Sparsities
        ax = axes[r, c]
        
        # Draw Ideal Line (y=x)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, x_line, linestyle='--', color='red', label='Ideal', alpha=0.8, zorder=1)
        
        # Column Titles (Top Row Only)
        if r == 0: ax.set_title(sparsities[c], fontsize=12)
        
        # Row Labels (Right side or Left side)
        if c == 0: ax.set_ylabel(f"{datasets[r]}\nEmpirical Recall", fontsize=12)
        if r == 2: ax.set_xlabel('Target Recall', fontsize=12)

        # Plot Logic
        if r == 0: # FB15k-237
            if c == 0:
                plot_panel(ax, fb_5_3p, '3p', '#1f77b4', 'o')
                plot_panel(ax, fb_5_2u, '2u', '#2ca02c', 's')
                plot_panel(ax, fb_5_2ip, '2ip', '#ff7f0e', '^')
            elif c == 1:
                plot_panel(ax, fb_20_3p, '3p', '#1f77b4', 'o')
                plot_panel(ax, fb_20_2u, '2u', '#2ca02c', 's')
                plot_panel(ax, fb_20_2ip, '2ip', '#ff7f0e', '^')
            elif c == 2:
                plot_panel(ax, fb_40_3p, '3p', '#1f77b4', 'o', [0.5, 0.6, 0.7, 0.8])
                plot_panel(ax, fb_40_2u, '2u', '#2ca02c', 's')
                plot_panel(ax, fb_40_2ip, '2ip', '#ff7f0e', '^', [0.5, 0.6])

        elif r == 1: # NELL-995
            if c == 0:
                plot_panel(ax, nell_5_3p, '3p', '#1f77b4', 'o', [0.5, 0.6, 0.7, 0.8])
                plot_panel(ax, nell_5_2u, '2u', '#2ca02c', 's')
            elif c == 1:
                plot_panel(ax, nell_20_3p, '3p', '#1f77b4', 'o')
                plot_panel(ax, nell_20_2u, '2u', '#2ca02c', 's')
            elif c == 2:
                plot_panel(ax, nell_40_2u, '2u', '#2ca02c', 's')
        
        elif r == 2: # YAGO (Empty for now)
            pass

        # Formatting
        ax.set_xlim(0.45, 0.95)
        ax.set_ylim(0.45, 0.95)
        ax.grid(True, alpha=0.6)
        if r == 0 and c == 0: 
            handles, labels = ax.get_legend_handles_labels()

plt.tight_layout()
# Create figure-level legend positioned at the top center
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.3, 1.035))
plt.savefig('conrad_validity_full.png', dpi=300, bbox_inches='tight')
# plt.show()
import matplotlib.pyplot as plt


# --- Configuration ---
alphas = [0.5, 0.4, 0.3, 0.2, 0.1]
sparsities = ["5% Missing", "20% Missing", "40% Missing"]
query_types = ["3p", "2u", "2ip"]
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
neural_markers = ['s', '^', 'D', 'v']  # square, triangle, diamond, inverted triangle


# fb15k-237
# --- Data Dictionary ---
# Data points are ordered by threshold (0.7, 0.8, 0.9, 0.99)
data = {
    "3p": {
        "conrad": [
            ([0.5175, 0.6193, 0.7177, 0.8085, 0.908], [0.668, 0.752, 0.828, 0.911, 0.960]),
            ([0.5164, 0.5941, 0.717, 0.8047, 0.9052], [0.75, 0.7948, 0.8841, 0.9169, 0.8709]),
            ([0.5091, 0.6086, 0.7406, 0.871], [0.8019, 0.8244, 0.6165, 0.3571])
        ],
        "neural": [
            ([0.9191, 0.9126, 0.8987, 0.8442], [0.9304, 0.9449, 0.9531, 0.9101]),
            ([0.6758, 0.6507, 0.6163, 0.5412], [0.8017, 0.8069, 0.7944, 0.7167]),
            ([0.3905, 0.3531, 0.3083, 0.2272], [0.6040, 0.5799, 0.5260, 0.3893])
        ],
        "symbolic": [(0.8668, 0.9310), (0.5832, 0.7780), (0.3153, 0.5730)]
    },
    "2u": {
        "conrad": [
            ([0.5109, 0.6144, 0.7131, 0.8135, 0.9118], [0.78, 0.86, 0.928, 0.97, 0.994]),
            ([0.5165, 0.6243, 0.7219, 0.8089, 0.8972], [0.823, 0.915, 0.966, 0.991, 0.999]),
            ([0.5021, 0.5985, 0.6909, 0.8036, 0.9062], [0.86, 0.93, 0.971, 0.9886, 0.7363])
        ],
        "neural": [
            ([0.9696, 0.9659, 0.9578, 0.9343], [0.9909, 0.9930, 0.9949, 0.9970]),
            ([0.8595, 0.8477, 0.8306, 0.7880], [0.9807, 0.9838, 0.9874, 0.9792]),
            ([0.6947, 0.6660, 0.6328, 0.5493], [0.9441, 0.9490, 0.9445, 0.9001])
        ],
        "symbolic": [(0.9534, 1.0000), (0.8049, 0.9900), (0.5835, 0.9210)]
    },
    "2ip": {
        "conrad": [
            ([0.4989, 0.6277, 0.7117, 0.7998, 0.9117], [0.587, 0.686, 0.769, 0.839, 0.933]), # 5% Added
            ([0.506, 0.5704, 0.6769, 0.7893, 0.8951], [0.63, 0.696, 0.792, 0.8738, 0.9021]), # 20%
            ([0.5444, 0.6286], [0.6197, 0.5986]) # 40% (Partial data before OOM)
        ],
        "neural": [
            ([0.9039, 0.8953, 0.8810, 0.8429], [0.9246, 0.9239, 0.9143, 0.8800]),
            ([0.6250, 0.5990, 0.5727, 0.5058], [0.7159, 0.6971, 0.6736, 0.6055]),
            ([0.3442, 0.3049, 0.2649, 0.1857], [0.4435, 0.4042, 0.3552, 0.2610])
        ],
        "symbolic": [(0.8748, 0.9100), (0.5511, 0.6520), (0.2724, 0.3870)]
    }
}

# --- Plotting ---
fig, axes = plt.subplots(3, 3, figsize=(10, 4.5))

for row, q_type in enumerate(query_types):
    for col, sparsity in enumerate(sparsities):
        ax = axes[row, col]
        
        # Plot Symbolic
        s_r, s_p = data[q_type]["symbolic"][col]
        ax.scatter(s_r, s_p, marker='X', color='C2', s=120, edgecolors='black', label='symbolic', zorder=5)

        # Plot Neural Points (different markers for each threshold)
        n_r, n_p = data[q_type]["neural"][col]
        for i, (r, p) in enumerate(zip(n_r, n_p)):
            label = f'neural (t={neural_thresholds[i]})' if row == 0 and col == 0 else None
            ax.scatter(r, p, marker=neural_markers[i], color='C1', s=50, 
                      edgecolors='black', linewidths=0.5, label=label, zorder=2)
        # Draw dashed line connecting neural points
        ax.plot(n_r, n_p, linestyle='--', color='C1', linewidth=1, alpha=0.5, zorder=3)

        # Plot Conrad Curve
        c_r, c_p = data[q_type]["conrad"][col]
        if c_r:
            ax.plot(c_r, c_p, marker='o', linestyle='-', color='C0', linewidth=2.5, label='conrad')
            for i, a in enumerate(alphas[:len(c_r)]):
                ax.annotate(f'α={a}', (c_r[i], c_p[i]), textcoords="offset points", 
                            xytext=(0, 10), fontsize=7, color='C0', fontweight='bold', ha='center')

        # Axis limits and Titles
        if col == 2:
            ax.set_xlim(0.1, 1)
            ax.set_ylim(0.2, 1.2)
        else:
            ax.set_xlim(0.45, 1)
            ax.set_ylim(0.5, 1.2)
            
        if row == 0: ax.set_title(sparsity, fontsize=12)
        if col == 0: ax.set_ylabel(f"{q_type.upper()}\nPrecision", fontsize=12)
        if row == 2: ax.set_xlabel("Empirical Recall", fontsize=12)
        
        ax.grid(True)
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
# Create figure-level legend positioned at the top center
# Group legend: symbolic, conrad, then neural thresholds
fig.legend(handles, labels, loc='upper center', ncol=7, fontsize=9, bbox_to_anchor=(0.5, 1.025))
plt.savefig('baselines_incompleteness_fb15k237.png', dpi=300, bbox_inches='tight')
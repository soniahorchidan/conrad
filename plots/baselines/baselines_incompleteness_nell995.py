import matplotlib.pyplot as plt

# --- Configuration & Data ---
alphas = [0.5, 0.4, 0.3, 0.2, 0.1]
sparsities = ["5% Missing", "20% Missing", "40% Missing"]
query_types = ["3p", "2u", "2ip"]
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
neural_markers = ['s', '^', 'D', 'v']  # square, triangle, diamond, inverted triangle

# Data from your NELL-995 experiments
data = {
    "3p": {
        "conrad": [
            ([0.5184, 0.6057, 0.7648, 0.8182], [0.563, 0.648, 0.815, 0.8449]), # 5%
            ([0.5223, 0.5936, 0.6925, 0.7906, 0.9017], [0.6019, 0.6568, 0.749, 0.6779, 0.7236]), # 20%
            ([], []) # 40% (OOM)
        ],
        "neural": [
            ([], []), # 5% (OOM)
            ([0.4023, 0.4022, 0.4022, 0.4000], [0.3636, 0.3637, 0.3642, 0.3666]), # 20%
            ([0.3256, 0.3253, 0.3244, 0.3238], [0.2817, 0.2820, 0.2828, 0.2855]) # 40%
        ],
        "symbolic": [(None, None), (0.3336, 0.4140), (0.2162, 0.3270)]
    },
    "2u": {
        "conrad": [
            ([0.5228, 0.6255, 0.7227, 0.8084, 0.9085], [0.819, 0.901, 0.951, 0.979, 0.995]), # 5%
            ([0.5151, 0.6054, 0.7129, 0.815, 0.893], [0.812, 0.884, 0.939, 0.9498, 0.7867]), # 20%
            ([0.5208, 0.6207, 0.722, 0.8205, 0.9004], [0.822, 0.8706, 0.8103, 0.7597, 0.4228]) # 40%
        ],
        "neural": [
            ([0.9693, 0.9687, 0.9687, 0.9618], [0.4166, 0.4343, 0.4609, 0.5316]), # 5%
            ([0.8708, 0.8693, 0.8653, 0.8490], [0.4273, 0.4431, 0.4688, 0.5317]), # 20%
            ([0.7198, 0.7136, 0.7085, 0.6893], [0.4620, 0.4720, 0.4936, 0.5432]) # 40%
        ],
        "symbolic": [(0.9561, 0.9970), (0.7970, 0.9730), (0.6073, 0.8800)]
    },
    "2ip": {
        "conrad": [([], []), ([], []), ([], [])], # TODO
        "neural": [
            ([0.9398, 0.9392, 0.9387, 0.9322], [0.4133, 0.4212, 0.4261, 0.4577]), # 5%
            ([0.7895, 0.7866, 0.7826, 0.7680], [0.3596, 0.3641, 0.3715, 0.3928]), # 20%
            ([0.5443, 0.5391, 0.5359, 0.5222], [0.2974, 0.2986, 0.3022, 0.3181]) # 40%
        ],
        "symbolic": [(0.8750, 0.8980), (0.5891, 0.6520), (0.2887, 0.3600)]
    }
}

# --- Plotting Logic (Matching FB15k Style) ---
fig, axes = plt.subplots(3, 3, figsize=(10, 4.5))

for row, q_type in enumerate(query_types):
    for col, sparsity in enumerate(sparsities):
        ax = axes[row, col]
        
        # Plot Symbolic
        s_r, s_p = data[q_type]["symbolic"][col]
        if s_r is not None:
            ax.scatter(s_r, s_p, marker='X', color='C2', s=120, edgecolors='black', label='symbolic', zorder=5)

        # Plot Neural Points (different markers for each threshold)
        n_r, n_p = data[q_type]["neural"][col]
        if n_r:
            for i, (r, p) in enumerate(zip(n_r, n_p)):
                label = f'neural (t={neural_thresholds[i]})' if row == 1 and col == 0 else None
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
            ax.set_xlim(0.1, 1.2); ax.set_ylim(0.2, 1.2)
        else:
            ax.set_xlim(0.2, 1.15); ax.set_ylim(0.3, 1.15)
            
        if row == 0: ax.set_title(sparsity, fontsize=12)
        if col == 0: ax.set_ylabel(f"{q_type.upper()}\nPrecision", fontsize=12)
        if row == 2: ax.set_xlabel("Empirical Recall", fontsize=12)
        
        ax.grid(True)
        if row == 1 and col == 0:
            handles, labels = ax.get_legend_handles_labels()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
# Create figure-level legend positioned at the top center
# Group legend: symbolic, conrad, then neural thresholds
fig.legend(handles, labels, loc='upper center', ncol=7, fontsize=9, bbox_to_anchor=(0.5, 1.025))
plt.savefig('baselines_incompleteness_nell995.png', dpi=300, bbox_inches='tight')
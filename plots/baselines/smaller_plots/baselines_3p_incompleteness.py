import matplotlib.pyplot as plt
import numpy as np

# --- Data Preparation ---
confidence = [0.5, 0.6, 0.7, 0.8, 0.9]
alphas = [0.5, 0.4, 0.3, 0.2, 0.1]
ultra_thresholds = [0.7, 0.8, 0.9, 0.99]

# 3p Results - Conrad
c_5_p, c_5_r = [0.668, 0.752, 0.828, 0.911, 0.960], [0.5175, 0.6193, 0.7177, 0.8085, 0.908]
c_20_p, c_20_r = [0.75, 0.7948, 0.8841, 0.9169, 0.8709], [0.5164, 0.5941, 0.717, 0.8047, 0.9052]
c_40_p, c_40_r = [0.8019, 0.8244, 0.6165, 0.3571], [0.5091, 0.6086, 0.7406, 0.871] # 4 points

# 3p Results - Full-Neural (Ultra)
n_5_p, n_5_r = [0.9304, 0.9449, 0.9531, 0.9101], [0.9191, 0.9126, 0.8987, 0.8442]
n_20_p, n_20_r = [0.8017, 0.8069, 0.7944, 0.7167], [0.6758, 0.6507, 0.6163, 0.5412]
n_40_p, n_40_r = [0.6040, 0.5799, 0.5260, 0.3893], [0.3905, 0.3531, 0.3083, 0.2272]

# 3p Results - Full-Symbolic (Neo4j)
s_5_p, s_5_r = 0.9310, 0.8668
s_20_p, s_20_r = 0.7780, 0.5832
s_40_p, s_40_r = 0.5730, 0.3153

# --- Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3))
sparsities = ["5% Missing", "20% Missing", "40% Missing"]

# Zip data for iteration
conrad_data = [(c_5_r, c_5_p), (c_20_r, c_20_p), (c_40_r, c_40_p)]
neural_data = [(n_5_r, n_5_p), (n_20_r, n_20_p), (n_40_r, n_40_p)]
symbolic_data = [(s_5_r, s_5_p), (s_20_r, s_20_p), (s_40_r, s_40_p)]

for i, ax in enumerate(axes):
    # 1. Plot Full-Symbolic (Scatter)
    ax.scatter(symbolic_data[i][0], symbolic_data[i][1], color='C2', s=120, 
                label='symbolic', zorder=5, edgecolor='black', marker='X')

    # 2. Plot Full-Neural (Dashed Line)
    ax.plot(neural_data[i][0], neural_data[i][1], marker='s', linestyle='--', 
            color='C1', label='neural')

    # 3. Plot Conrad (Solid Line)
    ax.plot(conrad_data[i][0], conrad_data[i][1], marker='o', linestyle='-', 
            color='C0', linewidth=3, label='conrad')

    # Annotations for Alpha (Ours)
    curr_r, curr_p = conrad_data[i]
    for j, a in enumerate(alphas[:len(curr_r)]):
        ax.annotate(f'α={a}', (curr_r[j], curr_p[j]), textcoords="offset points", 
                    xytext=(0, 10), fontsize=8, color='C0', fontweight='bold', ha='center')

    # Formatting
    ax.set_title(sparsities[i], fontsize=12)
    ax.set_xlabel('Empirical Recall', fontsize=12)
    if i == 0: ax.set_ylabel('Precision', fontsize=12)
    if i == 2:
        ax.set_xlim(0.15, 1.0)
        ax.set_ylim(0.3, 1.05)
    else:
        ax.set_xlim(0.45, 0.95)
        ax.set_ylim(0.65, 1.05)
    ax.grid(True, alpha=0.6)
    if i == 0: ax.legend(loc='lower right', frameon=True)

plt.tight_layout()
plt.savefig('baselines_3p_incompleteness.png', dpi=300)
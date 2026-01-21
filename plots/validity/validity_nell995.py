import matplotlib.pyplot as plt
import numpy as np

# Target Recall (1 - alpha)
targets = [0.5, 0.6, 0.7, 0.8, 0.9]

# NELL-995 Results
# 5% Missing
nell_5_2u = [0.5228, 0.6255, 0.7227, 0.8084, 0.9085]
nell_5_3p = [0.5184, 0.6057, 0.7648, 0.8182] # 0.9 is OOM

# 20% Missing
nell_20_2u = [0.5151, 0.6054, 0.7129, 0.815, 0.893]
nell_20_3p = [0.5223, 0.5936, 0.6925, 0.7906, 0.9017]

# 40% Missing
nell_40_2u = [0.5208, 0.6207, 0.722, 0.8205, 0.9004]
# nell_40_3p is OOM

fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)
titles = ["5% Missing", "20% Missing", "40% Missing"]

for i, ax in enumerate(axes):
    # Diagonal Safety Line
    x_line = np.linspace(0.4, 1.0, 100)
    ax.plot(x_line, x_line, linestyle='--', color='red', label='Ideal', linewidth=3)
    
    if i == 0: # 5%
        ax.plot(targets[:4], nell_5_3p, marker='o', label='3p', color='#1f77b4', linewidth=3)
        ax.plot(targets, nell_5_2u, marker='s', label='2u', color='#2ca02c', linewidth=3)
    elif i == 1: # 20%
        ax.plot(targets, nell_20_3p, marker='o', label='3p', color='#1f77b4', linewidth=3)
        ax.plot(targets, nell_20_2u, marker='s', label='2u', color='#2ca02c', linewidth=3)
    elif i == 2: # 40%
        ax.plot(targets, nell_40_2u, marker='s', label='2u', color='#2ca02c', linewidth=3)
        # 3p is OOM
        # ax.text(0.7, 0.5, "3p: OOM", color='grey', fontstyle='italic', ha='center')

    ax.set_title(titles[i], fontsize=12)
    ax.set_xlabel('Target Recall ($1-\\alpha$)', fontsize=12)
    if i == 0: ax.set_ylabel('Empirical Recall', fontsize=12)
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.45, 0.95)
    ax.grid(True, alpha=0.6)
    if i == 0: ax.legend()

plt.tight_layout()
plt.savefig('validity_nell995.png', dpi=300)
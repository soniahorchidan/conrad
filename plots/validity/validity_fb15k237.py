import matplotlib.pyplot as plt
import numpy as np

# Data from experiments
confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]

# 5% Missing
recall_5_3p = [0.5175, 0.6193, 0.7177, 0.8085, 0.908]
recall_5_2u = [0.5109, 0.6144, 0.7131, 0.8135, 0.9118]

# 20% Missing 
recall_20_3p = [0.5164, 0.5941, 0.717, 0.8047, 0.9052]
recall_20_2u = [0.5165, 0.6243, 0.7219, 0.8089, 0.8972]
recall_20_2ip = [0.506, 0.5704, 0.6769, 0.7893, 0.8951]

# 40% Missing
recall_40_3p = [0.5091, 0.6086, 0.7406, 0.871] # 4 points provided
conf_40_3p = [0.5, 0.6, 0.7, 0.8]
recall_40_2u = [0.5021, 0.5985, 0.6909, 0.8036, 0.9062]

fig, axes = plt.subplots(1, 3, figsize=(10, 2.5), sharey=True)
sparsities = ["5% Missing Data", "20% Missing Data", "40% Missing Data"]

for i, ax in enumerate(axes):
    # Safety Line
    x_line = np.linspace(0.4, 1.0, 100)
    ax.plot(x_line, x_line, linestyle='--', color='red', label='Ideal', linewidth=3)
    # ax.fill_between(x_line, x_line, 1.1, color='green', alpha=0.05)
    
    # Plotting Data
    if i == 0:
        ax.plot(confidence_levels, recall_5_3p, marker='o', label='3p', color='#1f77b4', linewidth=3)
        ax.plot(confidence_levels, recall_5_2u, marker='s', label='2u', color='#2ca02c', linewidth=3)
    elif i == 1:
        ax.plot(confidence_levels, recall_20_3p, marker='o', label='3p', color='#1f77b4', linewidth=3)
        ax.plot(confidence_levels, recall_20_2u, marker='s', label='2u', color='#2ca02c', linewidth=3)   
        ax.plot(confidence_levels, recall_20_2ip, marker='^', label='2ip', color='#ff7f0e', linewidth=3)
    elif i == 2:
        ax.plot(conf_40_3p, recall_40_3p, marker='o', label='3p', color='#1f77b4', linewidth=3)
        ax.plot(confidence_levels, recall_40_2u, marker='s', label='2u', color='#2ca02c', linewidth=3)

    ax.set_title(sparsities[i], fontsize=12)
    ax.set_xlabel('Target Recall ($1 - \\alpha$)', fontsize=12)
    if i == 0: ax.set_ylabel('Empirical Recall', fontsize=12)
    ax.set_xlim(0.45, 0.95)
    ax.set_ylim(0.45, 0.95)
    ax.grid(True, alpha=0.6)
    if i == 1: ax.legend(loc='lower right', ncol=2)

plt.tight_layout()
plt.savefig('validity_fb15k237.png', dpi=300)
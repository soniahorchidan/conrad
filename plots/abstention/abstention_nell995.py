import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Data for NELL-995 (3p Queries)
x_labels = ['0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(x_labels))
width = 0.25 

# --- 5% Missing Edges (NELL) ---
# CONRAD 3p abs_rate: [0.437, 0.352, 0.185, 0.148, OOM]
conrad_abs_5 = [0.437, 0.352, 0.185, 0.148, 0.0] 
# Neural Baseline 3p abs_rate: OOM
neural_abs_5 = [0.0] * 5
# Symbolic Baseline 3p abs_rate: N/A in provided logs, using 0 as placeholder or omit
symbolic_abs_5 = [0.0] * 5

# --- 20% Missing Edges (NELL) ---
# CONRAD 3p abs_rate: [0.395, 0.3279, 0.2318, 0.149, 0.072]
conrad_abs_20 = [0.395, 0.328, 0.232, 0.149, 0.072]
# Neural Baseline 3p abs_rate: [0.573, 0.573, 0.573, 0.573, 0.574]
neural_abs_20 = [0.573, 0.573, 0.573, 0.573, 0.574]
# Symbolic Baseline 3p abs_rate: 0.586
symbolic_abs_20 = [0.586] * 5

# --- 40% Missing Edges (NELL) ---
# CONRAD 3p: OOM
conrad_abs_40 = [0.0] * 5 
# Neural Baseline 3p abs_rate: [0.604, 0.604, 0.604, 0.604, 0.604]
neural_abs_40 = [0.604, 0.604, 0.604, 0.604, 0.604]
# Symbolic Baseline 3p abs_rate: 0.673
symbolic_abs_40 = [0.673] * 5

# --- Plotting ---
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 3), sharey=True)

# Plot 1: 5% Missing
ax1.bar(x - width, symbolic_abs_5, width, label='symbolic', color='#d9d9d9', hatch='//', edgecolor='grey')
ax1.bar(x, neural_abs_5, width, label='neural', color='#fdae61', edgecolor='grey', hatch='x')
ax1.bar(x + width, conrad_abs_5, width, label='conrad', color='#2c7bb6', edgecolor='black')
# ax1.text(x[-1]+width, 0.05, 'OOM', ha='center', color='red', fontsize=8, fontweight='bold')

# Plot 2: 20% Missing
ax2.bar(x - width, symbolic_abs_20, width, label='symbolic', color='#d9d9d9', hatch='//', edgecolor='grey')
ax2.bar(x, neural_abs_20, width, label='neural', color='#fdae61', edgecolor='grey', hatch='x')
ax2.bar(x + width, conrad_abs_20, width, label='conrad', color='#2c7bb6', edgecolor='black')

# Plot 3: 40% Missing
ax3.bar(x - width, symbolic_abs_40, width, label='symbolic', color='#d9d9d9', hatch='//', edgecolor='grey')
ax3.bar(x, neural_abs_40, width, label='neural', color='#fdae61', edgecolor='grey', hatch='x')
ax3.bar(x + width, conrad_abs_40, width, label='conrad', color='#2c7bb6', edgecolor='black')

# Formatting
ax1.set_title('5% Missing Data', fontsize=12)
ax2.set_title('20% Missing Data', fontsize=12)
ax3.set_title('40% Missing Data', fontsize=12)

for ax in [ax1, ax2, ax3]:
    ax.set_xlabel('Target Recall ($1 - \\alpha$)', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylim(0, 0.8) # NELL has higher baseline abstention than FB15k
    ax.grid(True, alpha=0.4, linestyle=':')

ax1.set_ylabel('Abstention Rate', fontsize=12)
ax1.legend(loc='upper right', frameon=True, fontsize='x-small')

plt.tight_layout()
plt.savefig('abstention_nell995.png', dpi=300)
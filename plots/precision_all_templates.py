import matplotlib.pyplot as plt
import numpy as np

# Data Setup
target_recall = ['0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(target_recall))
width = 0.25

# Abstention Rates (Updated)
abs_3hop = [0.25, 0.205, 0.113, 0.072, 0.041]
abs_2u = [0.177, 0.085, 0.034, 0.009, 0.001]
abs_2ip = [0.414, 0.297, 0.221, 0.143, 0.094]

# Precision (Non-Abstained) (Updated)
prec_non_abs_3hop = [1.0, 0.9998, 0.9968, 0.9881, 0.9081]
prec_non_abs_2u = [1.0, 1.0, 1.0, 1.0, 1.0]
prec_non_abs_2ip = [1.0, 1.0, 1.0, 1.0, 0.9965]

# Precision (Overall - includes abstained queries) (Updated)
prec_overall_3hop = [0.75, 0.7948, 0.8841, 0.9169, 0.8709]
prec_overall_2u = [0.823, 0.915, 0.966, 0.991, 0.999]
prec_overall_2ip = [0.586, 0.703, 0.779, 0.857, 0.9028]

fig, ax1 = plt.subplots(figsize=(5, 3.5))

# 1. Bar Chart for Abstention Rates (Primary Y-axis)
rects1 = ax1.bar(x - width, abs_3hop, width, label='3p abs. rate', color='#b3cde3', edgecolor='grey', alpha=0.7)
rects2 = ax1.bar(x, abs_2u, width, label='2u abs. rate', color='#ccebc5', edgecolor='grey', alpha=0.7)
rects3 = ax1.bar(x + width, abs_2ip, width, label='2ip abs. rate', color='#fed9a6', edgecolor='grey', alpha=0.7)

ax1.set_xlabel('Target Recall ($1 - \\alpha$)', fontsize=12)
ax1.set_ylabel('Abstention Rate', fontsize=12)
ax1.set_ylim(0, 0.8) # Adjust height for the merged legend
ax1.set_xticks(x)
ax1.set_xticklabels(target_recall)

# 2. Line Plot for Precision (Secondary Y-axis)
ax2 = ax1.twinx()

# Overall Precision (Solid Lines)
ax2.plot(x - width, prec_overall_3hop, marker='o', ls='-', color='#1f77b4', label='3p prec. (Overall)')
ax2.plot(x, prec_overall_2u, marker='s', ls='-', color='#2ca02c', label='2u prec. (Overall)')
ax2.plot(x + width, prec_overall_2ip, marker='^', ls='-', color='#ff7f0e', label='2ip prec. (Overall)')

# Non-Abstained Precision (Dashed Lines)
ax2.plot(x - width, prec_non_abs_3hop, marker='o', ls='--', color='#1f77b4', alpha=0.6, label='3p prec. (Non-Abs)')
ax2.plot(x, prec_non_abs_2u, marker='s', ls='--', color='#2ca02c', alpha=0.6, label='2u prec. (Non-Abs)')
ax2.plot(x + width, prec_non_abs_2ip, marker='^', ls='--', color='#ff7f0e', alpha=0.6, label='2ip prec. (Non-Abs)')

ax2.set_ylabel('Precision', fontsize=12)
ax2.set_ylim(0.5, 1.05) 

# --- Merge both legends ---
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()

# Combine handles and labels into a single legend with a small font to fit
ax1.legend(handles1 + handles2, labels1 + labels2, loc='upper center', 
           bbox_to_anchor=(0.5, 1.4), ncol=3, frameon=True, fontsize='small')

plt.grid(True, alpha=0.6)

fig.tight_layout()
plt.savefig('precision.png', dpi=300)
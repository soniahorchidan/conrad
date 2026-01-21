import matplotlib.pyplot as plt
import numpy as np

# 1. Setup Data
confidence = [0.5, 0.6, 0.7, 0.8, 0.9]

# Empirical Recall data from your experiments
recall_3hop = [0.5164, 0.5941, 0.717, 0.8047, 0.9052]
recall_2u = [0.5165, 0.6243, 0.7219, 0.8089, 0.8972]
recall_2iu = [0.4752, 0.5922, 0.6828, 0.7879, 0.8721]

plt.figure(figsize=(5, 3))

# 2. Draw Safety Boundary (y=x)
x_line = np.linspace(0.4, 1.0, 100)
plt.plot(x_line, x_line, linestyle='--', color='red', linewidth=2, label='Ideal')

# # 3. Shade regions to emphasize the "Broken" validity
# plt.fill_between(x_line, x_line, 1.1, color='green', alpha=0.08, label='Safe')
# plt.fill_between(x_line, 0, x_line, color='red', alpha=0.08, label='Violation')

# 4. Plot each query type as a separate line
plt.plot(confidence, recall_3hop, marker='o', markersize=8, label='3p', color='#1f77b4', linewidth=2)
plt.plot(confidence, recall_2u, marker='s', markersize=8, label='2u', color='#2ca02c', linewidth=2)
plt.plot(confidence, recall_2iu, marker='^', markersize=8, label='2ip', color='#ff7f0e', linewidth=2)

# # 5. Add specific markers for the 2iu violations
# for c, r in zip(confidence, recall_2iu):
#     if r < c:
#         plt.annotate('Violation', xy=(c, r), xytext=(c+0.01, r-0.04),
#                      arrowprops=dict(arrowstyle='->', color='black'),
#                      color='red', fontweight='bold', fontsize=9)

# 6. Formatting for VLDB standards
# plt.title('Empirical Validity Check (Target vs. Empirical Recall)', fontsize=14, fontweight='bold')
plt.xlabel('Target Recall ($1 - \\alpha$)', fontsize=12)
plt.ylabel('Empirical Recall', fontsize=12)
plt.xlim(0.45, 0.95)
plt.ylim(0.4, 1.0)
plt.xticks(confidence)
plt.grid(True, alpha=0.6)
plt.legend(loc='upper left', frameon=True)

plt.tight_layout()
plt.savefig('validity_violations_plot.png', dpi=300)
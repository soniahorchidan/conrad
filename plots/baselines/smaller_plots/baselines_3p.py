import matplotlib.pyplot as plt

# --- Data for conrad (3p) ---
# Target Confidence: [0.5, 0.6, 0.7, 0.8, 0.9]
ours_precision = [0.75, 0.7948, 0.8841, 0.9169, 0.8709]
ours_recall = [0.5164, 0.5941, 0.717, 0.8047, 0.9052]
alphas = [0.5, 0.4, 0.3, 0.2, 0.1]

# --- Updated Baseline Data ---
# Neo4j: Symbolic (Updated to R: 0.5832, P: 0.7780)
neo4j_rec, neo4j_prec = 0.5832, 0.7780

# Ultra: Static neural thresholds (Updated from table)
ultra_thresholds = [0.7, 0.8, 0.9, 0.95, 0.99]
ultra_rec = [0.6758, 0.6507, 0.6163, 0.5912, 0.5412]
ultra_prec = [0.8017, 0.8069, 0.7944, 0.7763, 0.7167]

plt.figure(figsize=(5, 4))

# Plot Neo4j
plt.scatter(neo4j_rec, neo4j_prec, color='C2', s=100, 
            label='full-symbolic', zorder=5, edgecolor='black')

# Plot Ultra line
plt.plot(ultra_rec, ultra_prec, marker='s', linestyle='--', color='C1', label='full-neural')

# Plot Our Pipeline (conrad)
plt.plot(ours_recall, ours_precision, marker='o', linestyle='-', color='C0', 
         linewidth=2.5, label='conrad')

# Label alpha levels for CQO (Ours)
for i, a in enumerate(alphas):
    plt.annotate(f'α={a}', (ours_recall[i], ours_precision[i]), 
                 textcoords="offset points", xytext=(-15,5), fontsize=9, color='C0', fontweight='bold')

# Label the Ultra thresholds
for i, t in enumerate(ultra_thresholds):
    plt.annotate(f't={t}', (ultra_rec[i], ultra_prec[i]), 
                 textcoords="offset points", xytext=(-5,-15), fontsize=9, color='C1', fontweight='bold')

# --- Final Formatting ---
plt.xlabel('Empirical Recall', fontsize=14)
plt.ylabel('Precision', fontsize=14)
plt.grid(True, alpha=0.6)
# Place legend in lower left as precision for ours is very high in the top right
plt.legend(loc='lower right', frameon=True, fontsize='small')
plt.xlim(0.46, 0.92)
plt.ylim(0.5, 1.05)

plt.tight_layout()
plt.savefig('baselines_3p.png', dpi=300)
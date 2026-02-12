# 

# ABOVE PLOT ALL

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Configuration for the 1x3 plot (20% missing data only)
datasets = ["FB15k-237", "NELL-995", "YAGO3-10"]
dataset_keys = ["fb15k-237", "nell-955", "yago310"]
sparsity_val = 20
confidence_levels = [0.6, 0.7, 0.8, 0.9]

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/benchmark_old/benchmark_new_one_exps")

def load_data(dataset, query_type, sparsity):
    """Load recall data from CSV file."""
    dir_name = f"conrad_bench_{dataset}_{query_type}Pipeline_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Filter to only the desired confidence levels
        df = df[df['confidence'].isin(confidence_levels)]
        # Extract confidence (target recall) and recall (empirical recall)
        target_recall = df['confidence'].tolist()
        empirical_recall = df['recall'].tolist()
        return target_recall, empirical_recall
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

# --- Plotting Setup ---
fig, axes = plt.subplots(1, 3, figsize=(5, 2), sharey=True)

# Helper to plot data safely
def plot_panel(ax, target_recall, empirical_recall, label, color, marker):
    if target_recall is not None and empirical_recall is not None:
        ax.plot(target_recall, empirical_recall, marker=marker, 
                label=label, color=color, linewidth=2, markersize=5)

for c in range(3):  # Datasets
    ax = axes[c]
    
    # Draw Ideal Line (y=x)
    x_line = np.linspace(0, 1, 100)
    ax.plot(x_line, x_line, linestyle='--', color='red', label='Ideal', alpha=0.8, zorder=1)
    
    # Per-panel title and axis labels
    ax.set_title(f"{datasets[c]}", fontsize=10)
    if c == 0:
        ax.set_ylabel('Empirical Recall', fontsize=10)
    ax.set_xlabel('Target Recall', fontsize=10)

    # Load and plot data for current dataset at 20% sparsity
    dataset_key = dataset_keys[c]
    
    # Load data for each query type
    target_3p, recall_3p = load_data(dataset_key, "ThreeHop", sparsity_val)
    target_2u, recall_2u = load_data(dataset_key, "TwoUnion", sparsity_val)
    target_2ip, recall_2ip = load_data(dataset_key, "TwoIntersectProject", sparsity_val)
    
    # Plot each query type
    plot_panel(ax, target_3p, recall_3p, '3p', '#1f77b4', 'o')
    plot_panel(ax, target_2u, recall_2u, '2u', '#2ca02c', 's')
    plot_panel(ax, target_2ip, recall_2ip, '2ip', '#ff7f0e', '^')

    # Formatting
    ax.set_xlim(0.55, 0.95)
    ax.set_ylim(0.55, 1)
    ax.set_xticks(confidence_levels)
    ax.grid(True, alpha=0.6)
    if c == 0: 
        handles, labels = ax.get_legend_handles_labels()

plt.tight_layout()
# Create figure-level legend positioned at the top center
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=9, bbox_to_anchor=(0.4, 1.08))
plt.savefig('conrad_validity_20pct.png', dpi=300, bbox_inches='tight')
print("Plot saved to conrad_validity_20pct.png")

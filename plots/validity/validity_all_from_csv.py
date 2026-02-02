import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path

# Configuration for the 3x3 plot
datasets = ["FB15k-237", "NELL-995", "YAGO3-10"]
dataset_keys = ["fb15k-237", "nell-995", "yago3-10"]
sparsities = ["5% Missing Data", "20% Missing Data", "40% Missing Data"]
sparsity_values = [5, 20, 40]
query_types = {
    "3p": "ThreeHopPipeline",
    "2u": "TwoUnionPipeline",
    "2ip": "TwoIntersectProjectPipeline"
}
confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/benchmark")

def load_data(dataset, query_type, sparsity):
    """Load recall data from CSV file."""
    dir_name = f"conrad_bench_{dataset}_{query_type}Pipeline_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Extract confidence (target recall) and recall (empirical recall)
        target_recall = df['confidence'].tolist()
        empirical_recall = df['recall'].tolist()
        return target_recall, empirical_recall
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

# --- Plotting Setup ---
fig, axes = plt.subplots(3, 3, figsize=(8, 5.5), sharex=True, sharey=True)

# Helper to plot data safely
def plot_panel(ax, target_recall, empirical_recall, label, color, marker):
    if target_recall is not None and empirical_recall is not None:
        ax.plot(target_recall, empirical_recall, marker=marker, 
                label=label, color=color, linewidth=2.5, markersize=7)

for r in range(3):  # Datasets
    for c in range(3):  # Sparsities
        ax = axes[r, c]
        
        # Draw Ideal Line (y=x)
        x_line = np.linspace(0, 1, 100)
        ax.plot(x_line, x_line, linestyle='--', color='red', label='Ideal', alpha=0.8, zorder=1)
        
        # Column Titles (Top Row Only)
        if r == 0: 
            ax.set_title(sparsities[c], fontsize=12)
        
        # Row Labels (Left side)
        if c == 0: 
            ax.set_ylabel(f"{datasets[r]}\nEmpirical Recall", fontsize=12)
        if r == 2: 
            ax.set_xlabel('Target Recall', fontsize=12)

        # Load and plot data for current dataset and sparsity
        dataset_key = dataset_keys[r]
        sparsity_val = sparsity_values[c]
        
        # Load data for each query type
        target_3p, recall_3p = load_data(dataset_key, "ThreeHop", sparsity_val)
        target_2u, recall_2u = load_data(dataset_key, "TwoUnion", sparsity_val)
        target_2ip, recall_2ip = load_data(dataset_key, "TwoIntersectProject", sparsity_val)
        
        # Plot each query type
        plot_panel(ax, target_3p, recall_3p, '3p', '#1f77b4', 'o')
        plot_panel(ax, target_2u, recall_2u, '2u', '#2ca02c', 's')
        plot_panel(ax, target_2ip, recall_2ip, '2ip', '#ff7f0e', '^')

        # Formatting
        ax.set_xlim(0.45, 0.95)
        ax.set_ylim(0.45, 0.95)
        ax.grid(True, alpha=0.6)
        if r == 0 and c == 0: 
            handles, labels = ax.get_legend_handles_labels()

plt.tight_layout()
# Create figure-level legend positioned at the top center
fig.legend(handles, labels, loc='upper center', ncol=4, fontsize=10, bbox_to_anchor=(0.3, 1.035))
plt.savefig('conrad_validity_full2.png', dpi=300, bbox_inches='tight')
print("Plot saved to conrad_validity_full.png")

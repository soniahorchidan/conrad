import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/plots_results")

# Regex to find: | pred: X, GT: Y
log_pattern = re.compile(r"pred: (\d+), GT: (\d+)")

def extract_differences_from_log(filepath):
    diffs = []
    if not Path(filepath).exists():
        return []
    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # Difference: |Pred| - |GT|
                diff = int(match.group(1)) - int(match.group(2))
                diffs.append(diff)
    return diffs

# --- Configuration ---
datasets = ["nell-955",]
dataset_labels = ["nell995"]
query_types = ["3p", "2ip"]
query_type_map = {
    "3p": "ThreeHopPipeline",
    # "2u": "TwoUnionPipeline",
    "2ip": "TwoIntersectProjectPipeline"
}
confidence_levels = ['0.6', '0.7', '0.8', '0.9']
sparsity = 20  # Using 20% missing data

fig, axes = plt.subplots(len(query_types), len(datasets), figsize=(5*len(datasets),2.5), sharey=True, squeeze=False)

for r, q_type in enumerate(query_types):
    for c, (dataset, d_label) in enumerate(zip(datasets, dataset_labels)):
        ax = axes[r, c]
        
        # Construct the directory path
        query_pipeline = query_type_map[q_type]
        dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
        dir_path = base_path / dir_name
        
        if dir_path.exists():
            print(f"Loading {d_label} {q_type} from {dir_name}")
            data_to_plot = [
                extract_differences_from_log(dir_path / f'benchmark_conf_{conf}.log') 
                for conf in confidence_levels
            ]
            
            if any(data_to_plot):
                bp = ax.boxplot(data_to_plot, vert=False, patch_artist=True, 
                                showfliers=True, whis=[5, 95], widths=0.4,
                                flierprops=dict(marker='o', markersize=4, color='gray', alpha=0.2),
                                medianprops=dict(color="black", linewidth=1.5))
                
                # Styling
                for patch in bp['boxes']:
                    patch.set_facecolor('none')
                    patch.set_edgecolor('C0')
                
                ax.set_xscale('symlog', linthresh=10)
                ax.axvline(0, color='red', linestyle='--', linewidth=1.5)
        else:
            print(f"Directory not found: {dir_name}")
        
        # Titles and labels
        if c == 0: ax.set_ylabel(f"{q_type}\nTarget Recall", fontsize=9)
        # if r == 0 and c == 0: ax.set_title(d_label, fontsize=12)
        if r == len(query_types) - 1: ax.set_xlabel("|Pred| - |GT|", fontsize=9)
        
        ax.grid(True, alpha=0.6)

# Set yticks and labels once for all subplots (since sharey=True)
for ax in axes.flat:
    ax.set_yticks(range(1, len(confidence_levels) + 1))
    ax.set_yticklabels(confidence_levels, fontsize=9)
    ax.tick_params(axis='x', labelsize=9)

# Place the legend in the upper left
handles = [plt.Line2D([0], [0], color='red', linestyle='--', label='Ground Truth')]
fig.legend(handles=handles, loc='upper left', fontsize=9, bbox_to_anchor=(0.1, 1.07))

plt.tight_layout()
plt.savefig('conrad_efficiency_all_from_csv.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to conrad_efficiency_all_from_csv.png")
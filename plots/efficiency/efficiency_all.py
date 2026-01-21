import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Regex to find: | pred: X, GT: Y
log_pattern = re.compile(r"pred: (\d+), GT: (\d+)")

def extract_differences_from_log(filepath):
    diffs = []
    if not os.path.exists(filepath):
        return []
    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # Difference: |Pred| - |GT|
                diff = int(match.group(1)) - int(match.group(2))
                diffs.append(diff)
    return diffs

# --- Configuration Mapping ---
datasets = ["fb15k237", "nell995", "yago310"]
query_types = ["3p", "2u", "2ip"]
labels = ['0.5', '0.6', '0.7', '0.8', '0.9']

paths = {
    "3p": {
        "fb15k237": "../../ml_engine/conrad_logs/crc_benchmark_results_20260106_093514",
        "nell995": "../../ml_engine/conrad_logs/crc_benchmark_results_20260109_121116"
    },
    "2u": {
        "fb15k237": "../../ml_engine/conrad_logs/crc_benchmark_results_20260106_115302",
        "nell995": "../../ml_engine/conrad_logs/crc_benchmark_results_20260109_132951"
    },
    "2ip": {
        "fb15k237": "../../ml_engine/crc_benchmark_results_20260120_154417"
    }
}

fig, axes = plt.subplots(3, 3, figsize=(15, 5), sharey=True)

for r, d_name in enumerate(datasets):
    for c, q_type in enumerate(query_types):
        ax = axes[r, c]
        path = paths.get(q_type, {}).get(d_name)
        
        if path:
            data_to_plot = [
                extract_differences_from_log(os.path.join(path, f'benchmark_conf_{l}.log')) 
                for l in labels
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
        
        # Titles and labels
        if r == 0: ax.set_title(q_type, fontsize=12)
        if c == 0: ax.set_ylabel(f"{d_name}\nTarget Recall", fontsize=11)
        if r == 2: ax.set_xlabel("|Pred| - |GT|", fontsize=11)
        
        ax.grid(True, alpha=0.6)

# Set yticks and labels once for all subplots (since sharey=True)
for ax in axes.flat:
    ax.set_yticks(range(1, len(labels) + 1))
    ax.set_yticklabels(labels)

# Place the legend in the upper left
handles = [plt.Line2D([0], [0], color='red', linestyle='--', label='Perfect Plan')]
fig.legend(handles=handles, loc='upper left', bbox_to_anchor=(0.03, 1), fontsize=10)

plt.tight_layout()
plt.savefig('conrad_efficiency_all.png', dpi=300, bbox_inches='tight')
# plt.show()
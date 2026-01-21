# import re
# import matplotlib.pyplot as plt
# import numpy as np
# import os

# # Regex to find: | pred: X, GT: Y
# log_pattern = re.compile(r"pred: (\d+), GT: (\d+)")

# def extract_differences_from_log(filepath):
#     diffs = []
#     if not os.path.exists(filepath):
#         return []
#     with open(filepath, 'r') as f:
#         for line in f:
#             match = log_pattern.search(line)
#             if match:
#                 # Calculate the difference: |Pred| - |GT|
#                 diff = int(match.group(1)) - int(match.group(2))
#                 diffs.append(diff)
#     return diffs

# # 3hop pipelines
# # fb15k237
# PATH = "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260106_093514"
# # nell995
# # PATH = "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260109_121116"

# # 2u pipelines
# # fb15k237
# # PATH = "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260106_115302"
# # nell995
# # PATH = "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260109_132951"

# # 2ip pipelines
# # PATH = "/home/sfhor/orb-dev/orb/ml_engine/crc_benchmark_results_20260106_155858"
# labels = ['0.5', '0.6', '0.7', '0.8', '0.9']

# # Prepare data as a list of lists for matplotlib
# data_to_plot = [
#     extract_differences_from_log(os.path.join(PATH, f'benchmark_conf_{l}.log')) 
#     for l in labels
# ]

# fig, ax = plt.subplots(figsize=(5, 2.5))

# # Create the boxplot
# # patch_artist=True allows us to fill the boxes with color
# # showfliers=False removes the outliers for a cleaner "system" view
# # bp = ax.boxplot(data_to_plot, 
# #                 patch_artist=True, 
# #                 showfliers=True,
# #                 whis=[5, 95],
# #                 widths=0.6,
# #                 flierprops=dict(marker='o', markersize=2, color='gray', alpha=0.2),
# #                 medianprops=dict(color="black", linewidth=1.5))

# # horizontal boxplot
# bp = ax.boxplot(data_to_plot, 
#                 vert=False,          
#                 patch_artist=True, 
#                 showfliers=True,
#                 whis=[5, 95],
#                 widths=0.4,
#                 flierprops=dict(marker='o', markersize=4, color='gray', alpha=0.2),
#                 medianprops=dict(color="black", linewidth=1.5))

# # Optional: Use symlog if you still have those massive outliers
# ax.set_xscale('symlog', linthresh=10) 

# # Styling the boxes
# # colors = ['C0', 'C0', 'C0', 'C0', 'C0']
# # for patch, color in zip(bp['boxes'], colors):
# #     patch.set_facecolor(color)

# for patch in bp['boxes']:
#     patch.set_facecolor('none')  # Removes the blue fill
#     patch.set_edgecolor('C0')

# # Change to a vertical line at 0
# ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Perfect Plan')

# # Swap labels and tick axes
# ax.set_xlabel('Set Size Difference (|Pred| - |GT|)', fontsize=11)
# ax.set_ylabel('Target Recall ($1 - \\alpha$)', fontsize=11)
# ax.set_yticklabels(['0.5', '0.6', '0.7', '0.8', '0.9'])

# # Grid now follows the x-axis (the values)
# ax.grid(True, alpha=0.6)
# ax.legend(fontsize=10)
# plt.tight_layout()
# plt.savefig('set_size_efficiency_3p_fb15k237.png', dpi=300)

import re
import matplotlib.pyplot as plt
import numpy as np
import os

# Regex to find: | pred: X, GT: Y
log_pattern = re.compile(r"pred: (\d+), GT: (\d+)")

def extract_differences_from_log(filepath):
    diffs = []
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return []
    with open(filepath, 'r') as f:
        for line in f:
            match = log_pattern.search(line)
            if match:
                # Calculate the difference: |Pred| - |GT|
                diff = int(match.group(1)) - int(match.group(2))
                diffs.append(diff)
    return diffs

# --- Configuration Mapping ---
# Added all your paths into a structured dictionary
configs = {
    "3p": {
        "fb15k237": "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260106_093514",
        "nell995": "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260109_121116"
    },
    "2u": {
        "fb15k237": "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260106_115302",
        "nell995": "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260109_132951"
    },
    "2ip": {
        "fb15k237": "/home/sfhor/orb-dev/orb/ml_engine/conrad_logs/crc_benchmark_results_20260106_155858"
    }
}

labels = ['0.5', '0.6', '0.7', '0.8', '0.9']

# --- Main Plotting Loop ---
for query_type, datasets in configs.items():
    for dataset_name, path in datasets.items():
        print(f"Processing: {query_type} - {dataset_name}...")
        
        data_to_plot = [
            extract_differences_from_log(os.path.join(path, f'benchmark_conf_{l}.log')) 
            for l in labels
        ]

        # Check if we actually have data to plot
        if not any(data_to_plot):
            continue

        fig, ax = plt.subplots(figsize=(4, 2))

        # Horizontal boxplot
        bp = ax.boxplot(data_to_plot, 
                        vert=False,          
                        patch_artist=True, 
                        showfliers=True,
                        whis=[5, 95],
                        widths=0.4,
                        flierprops=dict(marker='o', markersize=4, color='gray', alpha=0.2),
                        medianprops=dict(color="black", linewidth=1.5))

        # Use symlog for massive outliers
        ax.set_xscale('symlog', linthresh=10) 

        # Styling
        for patch in bp['boxes']:
            patch.set_facecolor('none')
            patch.set_edgecolor('C0')

        ax.axvline(0, color='red', linestyle='--', linewidth=1.5, label='Perfect Plan')
        
        # Labels and Titles
        ax.set_title(f"{query_type} ({dataset_name})", fontsize=12)
        ax.set_xlabel('Set Size Difference (|Pred| - |GT|)', fontsize=11)
        ax.set_ylabel('Target Recall ($1 - \\alpha$)', fontsize=11)
        ax.set_yticklabels(labels)

        ax.grid(True, alpha=0.6)
        ax.legend(fontsize=10)
        
        plt.tight_layout()
        
        # Save with a dynamic filename
        filename = f'set_size_efficiency_{query_type}_{dataset_name}.png'
        plt.savefig(filename, dpi=300)
        plt.close() # Close plot to free up memory

print("All plots generated successfully.")
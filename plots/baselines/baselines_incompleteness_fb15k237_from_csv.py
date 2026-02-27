import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Configuration ---
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
sparsities = ["5% Missing", "20% Missing", "40% Missing"]
sparsity_values = [5, 20, 40]
query_types = ["3p", "2u", "2ip"]
query_type_map = {
    "3p": "ThreeHopPipeline",
    "2u": "TwoUnionPipeline",
    "2ip": "TwoIntersectProjectPipeline"
}
conrad_markers = ['o', 'p', 'd', '8', '>']  # circle, pentagon, thin diamond, octagon, right triangle
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
neural_markers = ['s', '^', 'D', 'v']  # square, triangle, diamond, inverted triangle
hybrid_thresholds = [0.45, 0.5, 0.6, 0.7]
hybrid_markers = ['*', 'P', 'h', 'H']  # star, plus, hexagon1, hexagon2

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/plots_results")

def load_conrad_data(query_type, sparsity):
    """Load Conrad recall and precision data from CSV file."""
    dir_name = f"conrad_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Conrad path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        recall = df['recall'].tolist()
        precision = df['precision'].tolist()
        return recall, precision
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_neural_data(query_type, sparsity):
    """Load Neural baseline data for all thresholds from single CSV."""
    dir_name = f"neural_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Neural path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all neural rows (sorted by threshold)
        neural_rows = df[df['baseline'] == 'neural'].sort_values('threshold')
        recalls = neural_rows['recall'].tolist()
        precisions = neural_rows['precision'].tolist()
        return recalls, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_symbolic_data(query_type, sparsity):
    """Load Symbolic baseline data (single point)."""
    dir_name = f"symbolic_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Symbolic path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get the symbolic row
        symbolic_row = df[df['baseline'] == 'symbolic'].iloc[0]
        return symbolic_row['recall'], symbolic_row['precision']
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_hybrid_data(query_type, sparsity):
    """Load Hybrid baseline data (multiple threshold points)."""
    dir_name = f"hybrid_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Hybrid path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all hybrid rows
        hybrid_rows = df[df['baseline'] == 'hybrid']
        recalls = hybrid_rows['recall'].tolist()
        precisions = hybrid_rows['precision'].tolist()
        return recalls, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

# --- Data Dictionary ---
# Build data dictionary from CSV files
data = {}
for q_type in query_types:
    query_pipeline = query_type_map[q_type]
    data[q_type] = {
        "conrad": [],
        "neural": [],
        "symbolic": [],
        "hybrid": []
    }
    
    for sparsity in sparsity_values:
        # Load Conrad data
        c_r, c_p = load_conrad_data(query_pipeline, sparsity)
        data[q_type]["conrad"].append((c_r, c_p))
        print(f"Loaded conrad {q_type} {sparsity}%: {len(c_r) if c_r else 0} points")
        
        # Load Neural data
        n_r, n_p = load_neural_data(query_pipeline, sparsity)
        data[q_type]["neural"].append((n_r, n_p))
        print(f"Loaded neural {q_type} {sparsity}%: {len(n_r) if n_r else 0} points")
        
        # Load Symbolic data
        s_r, s_p = load_symbolic_data(query_pipeline, sparsity)
        data[q_type]["symbolic"].append((s_r, s_p))
        print(f"Loaded symbolic {q_type} {sparsity}%: {'yes' if s_r is not None else 'no'}")
        
        # Load Hybrid data
        h_r, h_p = load_hybrid_data(query_pipeline, sparsity)
        data[q_type]["hybrid"].append((h_r, h_p))
        print(f"Loaded hybrid {q_type} {sparsity}%: {len(h_r) if h_r else 0} points")

# --- Plotting ---
fig, axes = plt.subplots(3, 3, figsize=(10, 4.5))

for row, q_type in enumerate(query_types):
    for col, sparsity in enumerate(sparsities):
        ax = axes[row, col]
        
        # Plot Symbolic
        s_data = data[q_type]["symbolic"][col]
        if s_data[0] is not None and s_data[1] is not None:
            s_r, s_p = s_data
            ax.scatter(s_r, s_p, marker='X', color='C2', s=120, edgecolors='black', label='symbolic', zorder=5)

        # Plot Neural Points (different markers for each threshold)
        n_data = data[q_type]["neural"][col]
        if n_data[0] is not None and n_data[1] is not None:
            n_r, n_p = n_data[0], n_data[1]
            for i, (r, p) in enumerate(zip(n_r, n_p)):
                label = f'neural (t={neural_thresholds[i]})' if row == 0 and col == 0 else None
                ax.scatter(r, p, marker=neural_markers[i], color='C1', s=50, 
                          edgecolors='black', linewidths=1, label=label, zorder=2)
            # Draw dashed line connecting neural points
            ax.plot(n_r, n_p, linestyle='--', color='C1', linewidth=1, alpha=0.5, zorder=3)

        # Plot Hybrid Points (different markers for each threshold)
        h_data = data[q_type]["hybrid"][col]
        if h_data[0] is not None and h_data[1] is not None:
            h_r, h_p = h_data[0], h_data[1]
            for i, (r, p) in enumerate(zip(h_r, h_p)):
                marker_idx = min(i, len(hybrid_markers) - 1)  # Handle cases with fewer points
                label = f'hybrid (t={hybrid_thresholds[marker_idx]})' if row == 0 and col == 0 and i < len(hybrid_thresholds) else None
                ax.scatter(r, p, marker=hybrid_markers[marker_idx], color='C3', s=60, 
                          edgecolors='black', linewidths=1, label=label, zorder=4)
            # Draw dashed line connecting hybrid points
            ax.plot(h_r, h_p, linestyle='--', color='C3', linewidth=1, alpha=0.5, zorder=3)

        # Plot Conrad Points (different markers for each alpha)
        c_data = data[q_type]["conrad"][col]
        if c_data[0] is not None and c_data[1] is not None:
            c_r, c_p = c_data[0], c_data[1]
            for i, (r, p) in enumerate(zip(c_r, c_p)):
                marker_idx = min(i, len(conrad_markers) - 1)
                label = f'conrad (α={alphas[i]})' if row == 0 and col == 0 and i < len(alphas) else None
                ax.scatter(r, p, marker=conrad_markers[marker_idx], color='C0', s=60,
                          edgecolors='black', linewidths=1, label=label, zorder=6)
            # Draw line connecting conrad points
            ax.plot(c_r, c_p, linestyle='-', color='C0', linewidth=2.5, zorder=5)

        # Axis limits and Titles
        # if col == 2:
        #     ax.set_xlim(0.1, 1)
        #     ax.set_ylim(0.2, 1.2)
        # else:
        ax.set_xlim(0.1, 1.2)
        ax.set_ylim(0.1, 1.2)

        # ax.set_yscale('logit')
        # ax.set_ylim(0.1, 0.999)
            
            
        if row == 0: ax.set_title(sparsity, fontsize=12)
        if col == 0: ax.set_ylabel(f"{q_type.upper()}\nPrecision", fontsize=12)
        if row == 2: ax.set_xlabel("Empirical Recall", fontsize=12)
        
        ax.grid(True)
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
# Create figure-level legend positioned at the top center
# Group legend: symbolic, conrad, hybrid thresholds, neural thresholds
fig.legend(handles, labels, loc='upper center', ncol=7, fontsize=7.5, bbox_to_anchor=(0.5, 1.06))
plt.savefig('baselines_incompleteness_fb15k237_from_csv.png', dpi=300, bbox_inches='tight')
print("Plot saved to baselines_incompleteness_fb15k237_from_csv.png")


# # Plotting bar charts:

# import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# import numpy as np
# import pandas as pd
# from pathlib import Path

# # --- Configuration ---
# alphas = [0.6, 0.7, 0.8, 0.9]
# sparsities = ["5% Missing", "20% Missing", "40% Missing"]
# sparsity_values = [5, 20, 40]
# query_types = ["3p", "2u", "2ip"]
# query_type_map = {
#     "3p": "ThreeHopPipeline",
#     "2u": "TwoUnionPipeline",
#     "2ip": "TwoIntersectProjectPipeline"
# }
# neural_thresholds = [0.7, 0.8, 0.9, 0.99]
# hybrid_thresholds = [0.45, 0.5, 0.6, 0.7]

# # Target recall buckets: each is (label, lower_bound, upper_bound)
# recall_buckets = [
#     ("≤50%",  0.0,  0.55),
#     ("60%",   0.55, 0.65),
#     ("70%",   0.65, 0.75),
#     ("80%",   0.75, 0.85),
#     ("≥90%",  0.85, 1.50),
# ]

# # Methods config: (name, color, hatch)
# methods_config = [
#     ("Conrad",   "C0", None),
#     ("Symbolic", "C2", "//"),
#     ("Neural",   "C1", ".."),
#     ("Hybrid",   "C3", "xx"),
# ]

# # Base path to benchmark results
# base_path = Path("/data/sonia/conrad/artifacts/plots_results")


# def load_conrad_data(query_type, sparsity):
#     """Load Conrad recall and precision data from CSV file."""
#     dir_name = f"conrad_bench_fb15k-237_{query_type}_{sparsity}"
#     csv_path = base_path / dir_name / "results_summary.csv"
#     if not csv_path.exists():
#         return []
#     try:
#         df = pd.read_csv(csv_path)
#         return list(zip(df['recall'].tolist(), df['precision'].tolist()))
#     except Exception as e:
#         print(f"Error loading {csv_path}: {e}")
#         return []


# def load_neural_data(query_type, sparsity):
#     """Load Neural baseline data for all thresholds from single CSV."""
#     dir_name = f"neural_bench_fb15k-237_{query_type}_{sparsity}"
#     csv_path = base_path / dir_name / "baseline_results_summary.csv"
#     if not csv_path.exists():
#         return []
#     try:
#         df = pd.read_csv(csv_path)
#         neural_rows = df[df['baseline'] == 'neural'].sort_values('threshold')
#         return list(zip(neural_rows['recall'].tolist(), neural_rows['precision'].tolist()))
#     except Exception as e:
#         print(f"Error loading {csv_path}: {e}")
#         return []


# def load_symbolic_data(query_type, sparsity):
#     """Load Symbolic baseline data (single point)."""
#     dir_name = f"symbolic_bench_fb15k-237_{query_type}_{sparsity}"
#     csv_path = base_path / dir_name / "baseline_results_summary.csv"
#     if not csv_path.exists():
#         return []
#     try:
#         df = pd.read_csv(csv_path)
#         symbolic_row = df[df['baseline'] == 'symbolic'].iloc[0]
#         return [(symbolic_row['recall'], symbolic_row['precision'])]
#     except Exception as e:
#         print(f"Error loading {csv_path}: {e}")
#         return []


# def load_hybrid_data(query_type, sparsity):
#     """Load Hybrid baseline data (multiple threshold points)."""
#     dir_name = f"hybrid_bench_fb15k-237_{query_type}_{sparsity}"
#     csv_path = base_path / dir_name / "baseline_results_summary.csv"
#     if not csv_path.exists():
#         return []
#     try:
#         df = pd.read_csv(csv_path)
#         hybrid_rows = df[df['baseline'] == 'hybrid']
#         return list(zip(hybrid_rows['recall'].tolist(), hybrid_rows['precision'].tolist()))
#     except Exception as e:
#         print(f"Error loading {csv_path}: {e}")
#         return []


# def best_precision_in_bucket(points, lo, hi):
#     """From a list of (recall, precision) points, return the best precision
#     for a point whose recall falls in [lo, hi). Returns None if no point qualifies."""
#     candidates = [(r, p) for r, p in points if lo <= r < hi]
#     if not candidates:
#         return None
#     # Return the one with highest precision
#     return max(candidates, key=lambda x: x[1])[1]


# # --- Load all data ---
# data = {}
# for q_type in query_types:
#     qp = query_type_map[q_type]
#     data[q_type] = {"Conrad": [], "Neural": [], "Symbolic": [], "Hybrid": []}
#     for sp in sparsity_values:
#         data[q_type]["Conrad"].append(load_conrad_data(qp, sp))
#         data[q_type]["Neural"].append(load_neural_data(qp, sp))
#         data[q_type]["Symbolic"].append(load_symbolic_data(qp, sp))
#         data[q_type]["Hybrid"].append(load_hybrid_data(qp, sp))
#         print(f"Loaded {q_type} {sp}%:  Conrad={len(data[q_type]['Conrad'][-1])}  "
#               f"Neural={len(data[q_type]['Neural'][-1])}  "
#               f"Symbolic={len(data[q_type]['Symbolic'][-1])}  "
#               f"Hybrid={len(data[q_type]['Hybrid'][-1])}")

# # --- Plotting ---
# n_buckets = len(recall_buckets)
# n_methods = len(methods_config)
# bar_width = 0.18
# bucket_labels = [b[0] for b in recall_buckets]

# fig, axes = plt.subplots(3, 3, figsize=(14, 9))

# for row, q_type in enumerate(query_types):
#     for col, (sparsity_label, sp_val) in enumerate(zip(sparsities, sparsity_values)):
#         ax = axes[row, col]
#         x = np.arange(n_buckets)

#         for m_idx, (method_name, color, hatch) in enumerate(methods_config):
#             points = data[q_type][method_name][col]
#             precisions = []
#             for _, lo, hi in recall_buckets:
#                 p = best_precision_in_bucket(points, lo, hi)
#                 precisions.append(p if p is not None else 0)

#             # Track which bars are actually present (non-zero)
#             mask = [p > 0 for p in precisions]
#             offsets = x + (m_idx - n_methods / 2 + 0.5) * bar_width

#             bars = ax.bar(
#                 offsets, precisions, bar_width,
#                 color=color, edgecolor='black', linewidth=0.6,
#                 hatch=hatch, alpha=0.85,
#                 label=method_name if (row == 0 and col == 0) else None,
#             )

#             # Grey-out / hide bars with no data (value == 0)
#             for bar_obj, has_data in zip(bars, mask):
#                 if not has_data:
#                     bar_obj.set_height(0)

#             # Add precision value on top of each bar
#             for bar_obj, p_val, has in zip(bars, precisions, mask):
#                 if has:
#                     ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
#                             bar_obj.get_height() + 0.01, f'{p_val:.2f}',
#                             ha='center', va='bottom', fontsize=5.5, rotation=90)

#         ax.set_xticks(x)
#         ax.set_xticklabels(bucket_labels, fontsize=8)
#         ax.set_ylim(0, 1.15)
#         ax.set_yticks(np.arange(0, 1.1, 0.2))

#         if row == 0:
#             ax.set_title(sparsity_label, fontsize=12, fontweight='bold')
#         if col == 0:
#             ax.set_ylabel(f"{q_type.upper()}\nPrecision", fontsize=11)
#         if row == 2:
#             ax.set_xlabel("Recall Bucket", fontsize=11)

#         ax.grid(axis='y', linestyle='--', alpha=0.4)
#         ax.set_axisbelow(True)

# # Collect legend handles from the first subplot
# handles, labels = axes[0, 0].get_legend_handles_labels()

# plt.subplots_adjust(wspace=0.25, hspace=0.35)
# fig.legend(handles, labels, loc='upper center', ncol=n_methods,
#            fontsize=10, bbox_to_anchor=(0.5, 1.03), frameon=True)
# plt.savefig('baselines_incompleteness_fb15k237_from_csv_bars.png', dpi=300, bbox_inches='tight')
# print("Plot saved to baselines_incompleteness_fb15k237_from_csv_bars.png")

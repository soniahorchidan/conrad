"""
1x3 figure: empirical recall vs target (with CI band, ideal line) per query template (3p, 2u, 2ip).
Columns share x-axis. Data from artifacts/final_results/benchmark_ultraqueryt_normalized.
Prints max precision per template and the target recall where it is achieved.
"""

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Configuration: 3 query templates (columns)
query_types = ["ThreeHop", "TwoUnion", "TwoIntersectProject"]
query_labels = ["3p", "2u", "2ip"]
query_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
query_markers = ['o', 's', '^']
confidence_levels = [0.6, 0.7, 0.8, 0.9]
dataset_key = "nell-955"
sparsity_val = 20

base_path = Path("/data/sonia/conrad/artifacts/final_results/benchmark_ultraqueryt_normalized")

# CI for recall band
ci_level = 0.95
z_score = stats.norm.ppf((1 + ci_level) / 2)


def compute_confidence_band(x_values, num_queries, z_score=z_score):
    """Confidence band around diagonal for mean recall estimate (same as validity script)."""
    if num_queries <= 0:
        return None, None
    x_array = np.array(x_values)
    se = np.sqrt(x_array * (1 - x_array) / num_queries)
    lower_band = np.maximum(0, x_array - z_score * se)
    upper_band = np.minimum(1, x_array + z_score * se)
    return lower_band, upper_band


def load_data(query_type, sparsity):
    """Load target recall, empirical recall, precision, and num_queries from CSV."""
    dir_name = f"conrad_bench_{dataset_key}_{query_type}Pipeline_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    if not csv_path.exists():
        return None, None, None, None
    try:
        df = pd.read_csv(csv_path)
        df = df[df['confidence'].isin(confidence_levels)]
        target_recall = df['confidence'].tolist()
        empirical_recall = df['recall'].tolist()
        precision = df['precision'].tolist()
        num_queries = df['num_queries'].tolist()
        return target_recall, empirical_recall, precision, num_queries
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None, None


# --- 1x3 grid: columns share x ---
# Proportions and font sizes to match validity_all_from_csv_with_ci.py
fig, axes = plt.subplots(1, 3, figsize=(5, 1.7), sharex=True, sharey=True)
legend_added = False
all_points = []  # (target_recall, empirical_recall, dataset_name, query_label) for deviation analysis
max_precision_per_template = []  # (query_label, max_precision, target_at_max)

for c in range(3):
    query_type = query_types[c]
    query_label = query_labels[c]
    query_color = query_colors[c]
    query_marker = query_markers[c]

    target_recall, empirical_recall, precision_list, num_queries = load_data(
        query_type, sparsity_val
    )

    # Max precision and setup where achieved
    if precision_list is not None and target_recall is not None and len(precision_list) > 0:
        i_max = int(np.argmax(precision_list))
        max_precision_per_template.append((
            query_label,
            precision_list[i_max],
            target_recall[i_max],
        ))

    # Recall vs target (validity-style)
    ax = axes[c]
    if num_queries and len(num_queries) > 0:
        n = num_queries[0]
        x_band = np.linspace(0.55, 0.95, 200)
        lower_band, upper_band = compute_confidence_band(x_band, n, z_score)
        if lower_band is not None:
            label_band = f'{int(ci_level*100)}% CI band' if not legend_added else None
            ax.fill_between(x_band, lower_band, upper_band,
                            alpha=0.2, color='gray', zorder=0, label=label_band)
            if label_band:
                legend_added = True

    x_line = np.linspace(0.55, 0.95, 100)
    ideal_label = 'Ideal' if c == 0 else None
    ax.plot(x_line, x_line, linestyle='--', color='red', label=ideal_label,
            linewidth=2, zorder=2)

    if target_recall is not None and empirical_recall is not None:
        ax.plot(target_recall, empirical_recall, marker=query_marker,
                color=query_color, linewidth=2.5, markersize=5, zorder=3)
        for t, e in zip(target_recall, empirical_recall):
            all_points.append((t, e, "NELL-955", query_label))

    ax.set_xlim(0.55, 0.95)
    ax.set_ylim(0.55, 1)
    ax.set_xticks(confidence_levels)
    ax.grid(True, alpha=0.6)
    # Title left empty; query type shown in figure legend
    if c == 0:
        ax.set_ylabel('Empirical Recall', fontsize=9)
    ax.set_xlabel('Target Recall', fontsize=9)

# --- Highest deviation from ideal (y = x) ---
upward = [(e - t, (t, e, ds, ql)) for t, e, ds, ql in all_points if e > t]
downward = [(t - e, (t, e, ds, ql)) for t, e, ds, ql in all_points if e < t]
max_up = max(upward, key=lambda x: x[0]) if upward else None
max_down = max(downward, key=lambda x: x[0]) if downward else None

print("\n--- Deviation from ideal (empirical recall vs target recall) ---")
if max_up is not None:
    dev, (t, e, ds, ql) = max_up
    print(f"Highest upward deviation:   {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{ds}, {ql}]")
else:
    print("Highest upward deviation:   (none; no point above the diagonal)")
if max_down is not None:
    dev, (t, e, ds, ql) = max_down
    print(f"Highest downward deviation: {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{ds}, {ql}]")
else:
    print("Highest downward deviation: (none; no point below the diagonal)")
print()

# --- Maximum precision per query template ---
print("--- Maximum precision per query template ---")
for ql, max_prec, target_at_max in max_precision_per_template:
    print(f"  {ql}: max precision = {max_prec:.4f}  at target recall = {target_at_max}")
print()

# Figure-level legend: Ideal, CI band, then query types (3p, 2u, 2ip)
handles, labels = axes[0].get_legend_handles_labels()
# Proxy artists for query types (line + marker, same as in plot)
for ql, color, marker in zip(query_labels, query_colors, query_markers):
    handles.append(Line2D([0], [0], color=color, marker=marker, linestyle='-',
                         linewidth=2.5, markersize=5, label=ql))
    labels.append(ql)
# Reorder so "Ideal" comes first
ideal_idx = next((i for i, label in enumerate(labels) if label == 'Ideal'), None)
if ideal_idx is not None and ideal_idx != 0:
    handles = [handles[ideal_idx]] + [h for i, h in enumerate(handles) if i != ideal_idx]
    labels = [labels[ideal_idx]] + [l for i, l in enumerate(labels) if i != ideal_idx]
fig.legend(handles, labels, loc='upper left', ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.05, 1.1))

plt.tight_layout()

out_dir = Path(__file__).resolve().parent
plt.savefig(out_dir / "recall_precision_ultraqueryt.pdf", dpi=300, bbox_inches='tight')
plt.savefig(out_dir / "recall_precision_ultraqueryt.png", dpi=300, bbox_inches='tight')
print(f"Saved to {out_dir / 'recall_precision_ultraqueryt.pdf'} and .png")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Configuration for the 1x8 plot (20% sparsity, all datasets overlapped per topology)
# Columns: paper order — 2p, 3p, 2i, 3i, ip, pi, 2u, up
# Each subplot shows all 3 datasets overlapped, grouped by query topology.
datasets = [
    # (dataset_name, dataset_key, color, marker)
    ("FB15k-237", "fb15k-237", '#1f77b4', 'o'),
    ("NELL-995",  "nell-955",  '#ff7f0e', 's'),
    ("YAGO3-10",  "yago310",   '#2ca02c', '^'),
]

# Per-template config: (query_type, query_label, base_path)
ORIGINAL_BASE = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
NEW_BASE = Path("/data/sonia/conrad/artifacts/benchmark_revision")

templates = [
    # Ordered as in the original paper: 2p, 3p, 2i, 3i, ip, pi, 2u, up
    ("TwoHop",             "2p", NEW_BASE),
    ("ThreeHop",           "3p", ORIGINAL_BASE),
    ("TwoIntersect",       "2i", NEW_BASE),
    ("ThreeIntersect",     "3i", NEW_BASE),
    ("TwoIntersectProject","ip", ORIGINAL_BASE),
    ("ProjectIntersect",   "pi", NEW_BASE),
    ("TwoUnion",           "2u", ORIGINAL_BASE),
    ("UnionProject",       "up", NEW_BASE),
]

sparsity_val = 20
confidence_levels = [0.6, 0.7, 0.8, 0.9]

# Transparency for the data lines so overlapping datasets remain visible
LINE_ALPHA = 0.8

# Confidence level for CI (e.g., 0.95 for 95% CI)
ci_level = 0.95
z_score = stats.norm.ppf((1 + ci_level) / 2)  # z-score for 95% CI (≈1.96)

def compute_confidence_band(x_values, num_queries, z_score=z_score):
    if num_queries <= 0:
        return None, None
    x_array = np.array(x_values)
    se = np.sqrt(x_array * (1 - x_array) / num_queries)
    lower_band = np.maximum(0, x_array - z_score * se)
    upper_band = np.minimum(1, x_array + z_score * se)
    return lower_band, upper_band

def load_data(base_path, dataset, query_type, sparsity):
    """Load recall data from CSV file. Returns (None, None, None) if absent."""
    dir_name = f"conrad_bench_{dataset}_{query_type}Pipeline_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"

    if not csv_path.exists():
        return None, None, None

    try:
        df = pd.read_csv(csv_path)
        df = df[df['confidence'].isin(confidence_levels)]
        target_recall = df['confidence'].tolist()
        empirical_recall = df['recall'].tolist()
        num_queries = df['num_queries'].tolist()
        return target_recall, empirical_recall, num_queries
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

# --- Plotting Setup ---
# 1x8 grid: columns = query templates, all datasets overlapped in each subplot
n_cols = len(templates)
fig, axes = plt.subplots(1, n_cols, figsize=(13, 2.1), sharex=True, sharey=True)

band_legend_added = False

# Collect all (target, empirical) with metadata for deviation analysis
all_points = []  # list of (target_recall, empirical_recall, dataset_name, query_label)

for c, (query_type, query_label, base_path) in enumerate(templates):  # Columns: topologies
    ax = axes[c]

    # Single gray CI band per panel, using the tighter CI (largest n across the
    # overlapped datasets). Band widths differ by <=0.01 across datasets, so a single
    # uniform band keeps the figure clean without misrepresenting the tolerance.
    x_band = np.linspace(0.55, 0.95, 200)
    panel_n = max(
        (nq[0] for _, dataset_key, _, _ in datasets
         for nq in [load_data(base_path, dataset_key, query_type, sparsity_val)[2]]
         if nq is not None and len(nq) > 0),
        default=None,
    )
    if panel_n is not None:
        lower_band, upper_band = compute_confidence_band(x_band, panel_n, z_score)
        if lower_band is not None and upper_band is not None:
            label_band = f'{int(ci_level*100)}% CI band' if not band_legend_added else None
            ax.fill_between(x_band, lower_band, upper_band,
                            alpha=0.2, color='gray', zorder=0, label=label_band)
            if label_band:
                band_legend_added = True

    # Draw Ideal Line (y=x) on top of the band — always show it, even with no data
    x_line = np.linspace(0.55, 0.95, 100)
    ideal_label = 'Ideal' if c == 0 else None
    ax.plot(x_line, x_line, linestyle='--', color='red', label=ideal_label, linewidth=1.5, zorder=2)

    # Plot each dataset overlapped
    for dataset_name, dataset_key, ds_color, ds_marker in datasets:
        target_recall, empirical_recall, _ = load_data(
            base_path, dataset_key, query_type, sparsity_val
        )
        if target_recall is not None and empirical_recall is not None:
            ax.plot(target_recall, empirical_recall, marker=ds_marker,
                    label=None, color=ds_color, linewidth=2.0, markersize=6,
                    alpha=LINE_ALPHA, zorder=3)
            for t, e in zip(target_recall, empirical_recall):
                all_points.append((t, e, dataset_name, query_label))

    # Formatting
    ax.set_xlim(0.55, 0.95)
    ax.set_ylim(0.55, 1)
    ax.set_xticks(confidence_levels)
    ax.grid(True, alpha=0.6)
    ax.set_title(query_label, fontsize=11)
    ax.set_xlabel('Target Recall', fontsize=10)

    # Y-axis label on the first subplot only
    if c == 0:
        ax.set_ylabel('Empirical Recall', fontsize=10)

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

plt.tight_layout()

# Figure-level legend: Ideal, CI band, then one entry per dataset
handles, labels = axes[0].get_legend_handles_labels()
for dataset_name, _, ds_color, ds_marker in datasets:
    handles.append(Line2D([0], [0], color=ds_color, marker=ds_marker, linestyle='-',
                          linewidth=2.0, markersize=6, alpha=LINE_ALPHA, label=dataset_name))
    labels.append(dataset_name)
# Reorder so "Ideal" comes first
ideal_idx = next((i for i, label in enumerate(labels) if label == 'Ideal'), None)
if ideal_idx is not None and ideal_idx != 0:
    handles = [handles[ideal_idx]] + [h for i, h in enumerate(handles) if i != ideal_idx]
    labels = [labels[ideal_idx]] + [l for i, l in enumerate(labels) if i != ideal_idx]
fig.legend(handles, labels, loc='upper left', ncol=len(handles), fontsize=10,
           bbox_to_anchor=(0.0, 1.1))

plt.savefig('conrad_validity_3datasets_20pct_overlapped.pdf', dpi=300, bbox_inches='tight')
plt.savefig('conrad_validity_3datasets_20pct_overlapped.png', dpi=300, bbox_inches='tight')

print("Plot saved to conrad_validity_3datasets_20pct_overlapped.png")

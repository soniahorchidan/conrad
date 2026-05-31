import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Compact 3x8 plot (20% sparsity across 3 datasets).
# Rows: datasets (FB15k-237, NELL-995, YAGO3-10) — shown as row labels.
# Columns: paper order — 2p, 3p, 2i, 3i, ip, pi, 2u, up — shown as column titles.
# Each subplot keeps its own per-dataset CI band (width ~ 1/sqrt(n)), so the bands
# correctly widen on YAGO3-10 where the evaluation set is smaller.
datasets = [
    ("FB15k-237", "fb15k-237"),
    ("NELL-995",  "nell-955"),
    ("YAGO3-10",  "yago310"),
]

ORIGINAL_BASE = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
NEW_BASE = Path("/data/sonia/conrad/artifacts/benchmark_revision")

templates = [
    # Ordered as in the original paper: 2p, 3p, 2i, 3i, ip, pi, 2u, up
    ("TwoHop",             "2p", '#1f77b4', 'o', NEW_BASE),
    ("ThreeHop",           "3p", '#ff7f0e', 's', ORIGINAL_BASE),
    ("TwoIntersect",       "2i", '#2ca02c', '^', NEW_BASE),
    ("ThreeIntersect",     "3i", '#17becf', 'v', NEW_BASE),
    ("TwoIntersectProject","ip", '#9467bd', 'D', ORIGINAL_BASE),
    ("ProjectIntersect",   "pi", '#8c564b', 'P', NEW_BASE),
    ("TwoUnion",           "2u", '#e377c2', '*', ORIGINAL_BASE),
    ("UnionProject",       "up", '#7f7f7f', 'X', NEW_BASE),
]

sparsity_val = 20
confidence_levels = [0.6, 0.7, 0.8, 0.9]
xtick_levels = [0.6, 0.8]  # fewer ticks to reduce clutter in a compact grid

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
# Compact 3x8 grid: rows = datasets, columns = query templates
n_rows = len(datasets)
n_cols = len(templates)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 3.0), sharex=True, sharey=True)

legend_added = False
all_points = []  # list of (target_recall, empirical_recall, dataset_name, query_label)

for r, (dataset_name, dataset_key) in enumerate(datasets):  # Rows: datasets
    for c, (query_type, query_label, query_color, query_marker, base_path) in enumerate(templates):
        ax = axes[r, c]

        target_recall, empirical_recall, num_queries = load_data(
            base_path, dataset_key, query_type, sparsity_val
        )

        # Per-subplot CI band around the diagonal (uses this dataset's own n)
        if num_queries is not None and len(num_queries) > 0:
            x_band = np.linspace(0.55, 0.95, 200)
            lower_band, upper_band = compute_confidence_band(x_band, num_queries[0], z_score)
            if lower_band is not None and upper_band is not None:
                label_band = f'{int(ci_level*100)}% CI band' if not legend_added else None
                ax.fill_between(x_band, lower_band, upper_band,
                                alpha=0.25, color='gray', zorder=0, label=label_band)
                if label_band:
                    legend_added = True

        # Ideal line (y=x)
        x_line = np.linspace(0.55, 0.95, 100)
        ideal_label = 'Ideal' if r == 0 and c == 0 else None
        ax.plot(x_line, x_line, linestyle='--', color='red', label=ideal_label,
                linewidth=1.2, zorder=2)

        # Data for this (dataset, topology)
        if target_recall is not None and empirical_recall is not None:
            ax.plot(target_recall, empirical_recall, marker=query_marker,
                    label=None, color=query_color, linewidth=1.5, markersize=4, zorder=3)
            for t, e in zip(target_recall, empirical_recall):
                all_points.append((t, e, dataset_name, query_label))

        # Formatting
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.55, 1)
        ax.set_xticks(xtick_levels)
        ax.set_yticks([0.6, 0.8, 1.0])
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.4, linewidth=0.5)

        # Column titles (topology) on the top row only
        if r == 0:
            ax.set_title(query_label, fontsize=10, pad=3)

        # Dataset name as a row label on the first column
        if c == 0:
            ax.set_ylabel(dataset_name, fontsize=9)

        # X-axis label on the bottom row only
        if r == n_rows - 1:
            ax.set_xlabel('Target Recall', fontsize=8)

# Shared y-axis label for the whole grid
fig.supylabel('Empirical Recall', fontsize=9, x=0.005)

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

# Compact figure-level legend: only Ideal + CI band (topology is shown by column titles)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right', ncol=len(handles), fontsize=8,
           frameon=False, bbox_to_anchor=(1.0, 1.02))

plt.tight_layout(rect=[0.01, 0, 1, 0.97])
plt.subplots_adjust(wspace=0.12, hspace=0.18)

plt.savefig('conrad_validity_3datasets_20pct_compact.pdf', dpi=300, bbox_inches='tight')
plt.savefig('conrad_validity_3datasets_20pct_compact.png', dpi=300, bbox_inches='tight')

print("Plot saved to conrad_validity_3datasets_20pct_compact.png")

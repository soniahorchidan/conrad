import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# Configuration for the 2x8 plot (FB15k-237 across sparsity levels)
# Rows: sparsity levels (5%, 40%)
# Columns: paper order — 2p, 3p, 2i, 3i, ip, pi, 2u, up
datasets = [("FB15k-237", "fb15k-237")]

# Per-template config: (query_type, query_label, color, marker, base_path)
ORIGINAL_BASE = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
NEW_BASE = Path("/data/sonia/conrad/artifacts/benchmark_revision")

templates = [
    # Ordered as in the original paper: 2p, 3p, 2i, 3i, ip, pi, 2u, up
    # Colors follow the default matplotlib tab10 cycle (C0..C7) left-to-right
    # Each topology has a distinct marker
    ("TwoHop",             "2p", '#1f77b4', 'o', NEW_BASE),
    ("ThreeHop",           "3p", '#ff7f0e', 's', ORIGINAL_BASE),
    ("TwoIntersect",       "2i", '#2ca02c', '^', NEW_BASE),
    ("ThreeIntersect",     "3i", '#17becf', 'v', NEW_BASE),
    ("TwoIntersectProject","ip", '#9467bd', 'D', ORIGINAL_BASE),
    ("ProjectIntersect",   "pi", '#8c564b', 'P', NEW_BASE),
    ("TwoUnion",           "2u", '#e377c2', '*', ORIGINAL_BASE),
    ("UnionProject",       "up", '#7f7f7f', 'X', NEW_BASE),
]

sparsity_values = [5, 40]
sparsity_labels = ["5% sparsity", "40% sparsity"]
confidence_levels = [0.6, 0.7, 0.8, 0.9]

# Confidence level for CI (e.g., 0.95 for 95% CI)
ci_level = 0.95
z_score = stats.norm.ppf((1 + ci_level) / 2)  # z-score for 95% CI (≈1.96)

def compute_confidence_band(x_values, num_queries, z_score=z_score):
    """
    Compute confidence band around the diagonal (ideal line) for mean recall estimate.

    The CRC guarantee is on the expected risk. The empirical mean recall on a finite
    validation set is an estimate of that expectation, with sampling variance.

    Standard error of mean recall: SE = σ/√n, where σ is the standard deviation of
    per-query recall. Using normal approximation for proportions:
    SE ≈ sqrt(p(1-p)/n) where p is the target recall.

    Confidence band: target ± z_{0.975} · SE

    Args:
        x_values: Array of target recall values (x-axis, same as y-axis for ideal line)
        num_queries: Number of inference queries (sample size)
        z_score: Z-score for desired confidence level (default: 1.96 for 95% CI)

    Returns:
        Tuple of (lower_band, upper_band) arrays
    """
    if num_queries <= 0:
        return None, None

    x_array = np.array(x_values)

    # Standard error for proportion: SE = sqrt(p(1-p)/n)
    # This accounts for sampling variance of the mean recall estimate
    se = np.sqrt(x_array * (1 - x_array) / num_queries)

    # Confidence band around diagonal
    lower_band = np.maximum(0, x_array - z_score * se)
    upper_band = np.minimum(1, x_array + z_score * se)

    return lower_band, upper_band

def load_data(base_path, dataset, query_type, sparsity):
    """Load recall data from CSV file."""
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
# 2x8 grid: rows = sparsity levels, columns = query templates
n_cols = len(templates)
fig, axes = plt.subplots(2, n_cols, figsize=(13, 2.5), sharex=True, sharey=True)

legend_added = False

# Collect all (target, empirical) with metadata for deviation analysis
all_points = []  # list of (target_recall, empirical_recall, sparsity_label, query_label)

dataset_name, dataset_key = datasets[0]

for r, sparsity_val in enumerate(sparsity_values):  # Rows: sparsity levels
    sparsity_label = sparsity_labels[r]

    for c, (query_type, query_label, query_color, query_marker, base_path) in enumerate(templates):
        ax = axes[r, c]

        target_recall, empirical_recall, num_queries = load_data(
            base_path, dataset_key, query_type, sparsity_val
        )

        # Draw confidence band around ideal line (diagonal)
        if num_queries is not None and len(num_queries) > 0:
            num_queries_for_band = num_queries[0]

            x_band = np.linspace(0.55, 0.95, 200)
            lower_band, upper_band = compute_confidence_band(x_band, num_queries_for_band, z_score)

            if lower_band is not None and upper_band is not None:
                label_band = f'{int(ci_level*100)}% CI band' if not legend_added else None
                ax.fill_between(x_band, lower_band, upper_band,
                              alpha=0.2, color='gray', zorder=0, label=label_band)
                if label_band:
                    legend_added = True

        # Draw Ideal Line (y=x) on top of the band
        x_line = np.linspace(0.55, 0.95, 100)
        ideal_label = 'Ideal' if r == 0 and c == 0 else None
        ax.plot(x_line, x_line, linestyle='--', color='red', label=ideal_label, linewidth=2, zorder=2)

        # Plot the data for this query type
        if target_recall is not None and empirical_recall is not None:
            ax.plot(target_recall, empirical_recall, marker=query_marker,
                    label=None, color=query_color, linewidth=2.5, markersize=6, zorder=3)
            for t, e in zip(target_recall, empirical_recall):
                all_points.append((t, e, sparsity_label, query_label))

        # Formatting
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.55, 1)
        ax.set_xticks(confidence_levels)
        ax.grid(True, alpha=0.6)

        # Y-axis labels (first column only); include sparsity level
        if c == 0:
            ax.set_ylabel(f'{sparsity_label}\nEmpirical Recall', fontsize=8)

        # X-axis labels (bottom row only)
        if r == len(sparsity_values) - 1:
            ax.set_xlabel('Target Recall', fontsize=8)

# --- Highest deviation from ideal (y = x) ---
upward = [(e - t, (t, e, sp, ql)) for t, e, sp, ql in all_points if e > t]
downward = [(t - e, (t, e, sp, ql)) for t, e, sp, ql in all_points if e < t]
max_up = max(upward, key=lambda x: x[0]) if upward else None
max_down = max(downward, key=lambda x: x[0]) if downward else None

print("\n--- Deviation from ideal (empirical recall vs target recall) ---")
if max_up is not None:
    dev, (t, e, sp, ql) = max_up
    print(f"Highest upward deviation:   {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{sp}, {ql}]")
else:
    print("Highest upward deviation:   (none; no point above the diagonal)")
if max_down is not None:
    dev, (t, e, sp, ql) = max_down
    print(f"Highest downward deviation: {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{sp}, {ql}]")
else:
    print("Highest downward deviation: (none; no point below the diagonal)")
print()

plt.tight_layout()
# Figure-level legend: Ideal, CI band, then all 8 query types
handles, labels = axes[0, 0].get_legend_handles_labels()
for query_type, query_label, query_color, query_marker, _ in templates:
    handles.append(Line2D([0], [0], color=query_color, marker=query_marker, linestyle='-',
                         linewidth=2.5, markersize=6, label=query_label))
    labels.append(query_label)
# Reorder so "Ideal" comes first
ideal_idx = next((i for i, label in enumerate(labels) if label == 'Ideal'), None)
if ideal_idx is not None and ideal_idx != 0:
    handles = [handles[ideal_idx]] + [h for i, h in enumerate(handles) if i != ideal_idx]
    labels = [labels[ideal_idx]] + [l for i, l in enumerate(labels) if i != ideal_idx]
fig.legend(handles, labels, loc='upper left', ncol=len(handles), fontsize=8,
           bbox_to_anchor=(0.05, 1.04))

plt.savefig('conrad_validity_all_topologies_fb15k-237_5_40.pdf', dpi=300, bbox_inches='tight')
plt.savefig('conrad_validity_all_topologies_fb15k-237_5_40.png', dpi=300, bbox_inches='tight')

print("Plot saved to conrad_validity_all_topologies_fb15k-237_5_40.png")

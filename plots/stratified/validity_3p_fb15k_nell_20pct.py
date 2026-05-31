import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats

# 1x2 plot: 3p query at 20% sparsity for FB15k-237 and NELL-995
datasets = [
    ("FB15k-237", "fb15k-237"),
    ("NELL-995",  "nell-955"),
]

BASE = Path("/data/sonia/conrad/artifacts/benchmark")

query_type = "ThreeHop"
query_label = "3p"
query_color = '#ff7f0e'
query_marker = 's'

sparsity_val = 20
confidence_levels = [0.6, 0.7, 0.8, 0.9]

ci_level = 0.95
z_score = stats.norm.ppf((1 + ci_level) / 2)

def compute_confidence_band(x_values, num_queries, z_score=z_score):
    if num_queries <= 0:
        return None, None
    x_array = np.array(x_values)
    se = np.sqrt(x_array * (1 - x_array) / num_queries)
    lower_band = np.maximum(0, x_array - z_score * se)
    upper_band = np.minimum(1, x_array + z_score * se)
    return lower_band, upper_band

def load_data(base_path, dataset, query_type, sparsity):
    dir_name = f"conrad_bench_{dataset}_{query_type}Pipeline_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    if not csv_path.exists():
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
        df = df[df['confidence'].isin(confidence_levels)]
        return df['confidence'].tolist(), df['recall'].tolist(), df['num_queries'].tolist()
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

fig, axes = plt.subplots(1, 2, figsize=(6, 2.8), sharex=True, sharey=True)

legend_added = False
all_points = []

for c, (dataset_name, dataset_key) in enumerate(datasets):
    ax = axes[c]

    target_recall, empirical_recall, num_queries = load_data(
        BASE, dataset_key, query_type, sparsity_val
    )

    if num_queries is not None and len(num_queries) > 0:
        x_band = np.linspace(0.55, 0.95, 200)
        lower_band, upper_band = compute_confidence_band(x_band, num_queries[0], z_score)
        if lower_band is not None and upper_band is not None:
            label_band = f'{int(ci_level*100)}% CI band' if not legend_added else None
            ax.fill_between(x_band, lower_band, upper_band,
                            alpha=0.2, color='gray', zorder=0, label=label_band)
            if label_band:
                legend_added = True

    x_line = np.linspace(0.55, 0.95, 100)
    ideal_label = 'Ideal' if c == 0 else None
    ax.plot(x_line, x_line, linestyle='--', color='red', label=ideal_label, linewidth=2, zorder=2)

    if target_recall is not None and empirical_recall is not None:
        ax.plot(target_recall, empirical_recall, marker=query_marker,
                color=query_color, linewidth=2.5, markersize=7, zorder=3)
        for t, e in zip(target_recall, empirical_recall):
            all_points.append((t, e, dataset_name, query_label))

    ax.set_xlim(0.55, 0.95)
    ax.set_ylim(0.55, 1)
    ax.set_xticks(confidence_levels)
    ax.grid(True, alpha=0.6)
    ax.set_title(dataset_name, fontsize=10)
    ax.set_xlabel('Target Recall', fontsize=9)
    if c == 0:
        ax.set_ylabel('Empirical Recall', fontsize=9)

upward = [(e - t, (t, e, ds, ql)) for t, e, ds, ql in all_points if e > t]
downward = [(t - e, (t, e, ds, ql)) for t, e, ds, ql in all_points if e < t]
max_up = max(upward, key=lambda x: x[0]) if upward else None
max_down = max(downward, key=lambda x: x[0]) if downward else None

print("\n--- Deviation from ideal (empirical recall vs target recall) ---")
if max_up is not None:
    dev, (t, e, ds, ql) = max_up
    print(f"Highest upward deviation:   {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{ds}, {ql}]")
else:
    print("Highest upward deviation:   (none)")
if max_down is not None:
    dev, (t, e, ds, ql) = max_down
    print(f"Highest downward deviation: {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{ds}, {ql}]")
else:
    print("Highest downward deviation: (none)")
print()

plt.tight_layout()

handles, labels = axes[0].get_legend_handles_labels()
handles.append(Line2D([0], [0], color=query_color, marker=query_marker, linestyle='-',
                     linewidth=2.5, markersize=7, label=query_label))
labels.append(query_label)
ideal_idx = next((i for i, label in enumerate(labels) if label == 'Ideal'), None)
if ideal_idx is not None and ideal_idx != 0:
    handles = [handles[ideal_idx]] + [h for i, h in enumerate(handles) if i != ideal_idx]
    labels = [labels[ideal_idx]] + [l for i, l in enumerate(labels) if i != ideal_idx]
fig.legend(handles, labels, loc='upper center', ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.5, 1.08))

plt.savefig('conrad_validity_3p_fb15k_nell_20pct.pdf', dpi=300, bbox_inches='tight')
plt.savefig('conrad_validity_3p_fb15k_nell_20pct.png', dpi=300, bbox_inches='tight')

print("Plot saved to conrad_validity_3p_fb15k_nell_20pct.png")

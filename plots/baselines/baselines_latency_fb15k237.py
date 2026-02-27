import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import re
from pathlib import Path

# --- Configuration ---
dataset = "fb15k-237"
query_pipeline = "ThreeHopPipeline"
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
hybrid_thresholds = [0.45, 0.5, 0.6, 0.7]

base_path = Path("/data/sonia/conrad/artifacts/plots_results")

# Sparsity regimes and their target recall ranges:
#   1. 5% missing  → ~90% recall achievers
#   2. 20% missing → ~60% recall achievers
#   3. 40% missing → <50% recall achievers
regimes = [
    {"sparsity": 5,  "label": "5% Missing Data",  "recall_target": 0.90, "recall_range": (0.85, 1.00)},
    {"sparsity": 20, "label": "20% Missing Data", "recall_target": 0.60, "recall_range": (0.55, 0.70)},
    {"sparsity": 40, "label": "40% Missing Data", "recall_target": 0.45, "recall_range": (0.00, 0.55)},
]

# --- Data loading helpers ---

def parse_conrad_avg_time(sparsity, confidence):
    """Parse avg latency (ms) from Conrad benchmark log file."""
    dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
    log_path = base_path / dir_name / f"benchmark_conf_{confidence}.log"
    if not log_path.exists():
        print(f"  Conrad log not found: {log_path}")
        return None
    text = log_path.read_text()
    # Match: "OVERALL AVERAGE (All queries): ... Avg Time: 122.45ms"
    m = re.search(r'OVERALL AVERAGE \(All queries\).*?Avg Time:\s*([0-9.]+)ms', text)
    if m:
        return float(m.group(1))
    return None


def load_conrad_points(sparsity):
    """Return list of (recall, precision, avg_time_ms, label) for Conrad."""
    dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    points = []
    for _, row in df.iterrows():
        conf = row['confidence']
        avg_t = parse_conrad_avg_time(sparsity, conf)
        if avg_t is not None:
            points.append({
                "recall": row['recall'],
                "precision": row['precision'],
                "latency_ms": avg_t,
                "label": f"α={conf}",
            })
    return points


def load_baseline_points(baseline, sparsity):
    """Return list of (recall, precision, avg_time_ms, label) for a baseline."""
    prefix = f"{baseline}_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / prefix / "baseline_results_summary.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    rows = df[df['baseline'] == baseline]
    points = []
    for _, row in rows.iterrows():
        t = row['threshold']
        label = f"t={t}" if t != "N/A" else ""
        points.append({
            "recall": row['recall'],
            "precision": row['precision'],
            "latency_ms": row['avg_time_ms'],
            "label": label,
        })
    return points


def pick_closest(points, recall_target, recall_range):
    """From a list of point dicts, pick the one whose recall is closest to
    recall_target while falling within recall_range. If none are in range,
    pick the closest overall."""
    if not points:
        return None
    in_range = [p for p in points if recall_range[0] <= p['recall'] <= recall_range[1]]
    pool = in_range if in_range else points
    return min(pool, key=lambda p: abs(p['recall'] - recall_target))


# --- Collect selected points per regime ---
methods_config = [
    ("Conrad",   "C0", "conrad",   None),
    ("Symbolic", "C2", "symbolic", "//"),
    ("Neural",   "C1", "neural",   ".."),
    ("Hybrid",   "C3", "hybrid",   "xx"),
]

selected = []  # list of dicts per regime
for regime in regimes:
    sp = regime["sparsity"]
    target = regime["recall_target"]
    rng = regime["recall_range"]

    regime_data = {"regime": regime, "methods": {}}

    # Conrad
    conrad_pts = load_conrad_points(sp)
    best_c = pick_closest(conrad_pts, target, rng)
    regime_data["methods"]["Conrad"] = best_c
    if best_c:
        print(f"[{regime['label']}] Conrad  → recall={best_c['recall']:.4f}, latency={best_c['latency_ms']:.2f}ms ({best_c['label']})")

    # Baselines
    for method_name, _, baseline_key, _ in methods_config[1:]:
        pts = load_baseline_points(baseline_key, sp)
        best = pick_closest(pts, target, rng)
        regime_data["methods"][method_name] = best
        if best:
            print(f"[{regime['label']}] {method_name:8s} → recall={best['recall']:.4f}, latency={best['latency_ms']:.2f}ms ({best['label']})")

    selected.append(regime_data)

# --- Plotting: grouped bar chart 1×3 ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3.5), sharey=False)

bar_width = 0.6
method_names = [m[0] for m in methods_config]
method_colors = {m[0]: m[1] for m in methods_config}
method_hatches = {m[0]: m[3] for m in methods_config}

for col, regime_sel in enumerate(selected):
    ax = axes[col]
    regime = regime_sel["regime"]
    methods_data = regime_sel["methods"]

    x_positions = np.arange(len(method_names))
    latencies = []
    recalls = []
    colors = []
    hatches = []

    for mname in method_names:
        pt = methods_data.get(mname)
        if pt is not None:
            latencies.append(pt['latency_ms'])
            recalls.append(pt['recall'])
        else:
            latencies.append(0)
            recalls.append(None)
        colors.append(method_colors[mname])
        hatches.append(method_hatches[mname])

    bars = ax.bar(x_positions, latencies, bar_width,
                  color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)

    # Apply hatches
    for bar, hatch in zip(bars, hatches):
        if hatch:
            bar.set_hatch(hatch)

    # Annotate bars with latency value and recall
    for i, (bar, lat, rec) in enumerate(zip(bars, latencies, recalls)):
        if lat > 0 and rec is not None:
            # Latency on top
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f'{lat:.1f} ms', ha='center', va='bottom', fontsize=7.5,
                    fontweight='bold')
            # Recall inside bar
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 0.5,
                    f'R={rec:.2f}', ha='center', va='center', fontsize=7,
                    color='white', fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.45))

    ax.set_xticks(x_positions)
    ax.set_xticklabels(method_names, fontsize=9, rotation=15, ha='right')
    ax.set_title(regime["label"], fontsize=11, fontweight='bold')
    ax.set_ylabel("Avg Latency (ms)" if col == 0 else "", fontsize=10)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Set y-axis with some headroom for annotation
    max_lat = max(latencies) if latencies else 10
    ax.set_ylim(0, max_lat * 1.35)

# Suptitle
fig.suptitle("FB15K-237 — 3p Query Latency at Comparable Recall", fontsize=12, fontweight='bold', y=1.02)

# Legend
legend_patches = []
for mname, mcolor, _, mhatch in methods_config:
    p = mpatches.Patch(facecolor=mcolor, edgecolor='black', hatch=mhatch,
                       alpha=0.85, label=mname)
    legend_patches.append(p)
fig.legend(handles=legend_patches, loc='upper center', ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, 1.0), frameon=True)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig('baselines_latency_fb15k237.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to baselines_latency_fb15k237.png")

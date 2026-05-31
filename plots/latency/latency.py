import re
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
from pathlib import Path
from brokenaxes import brokenaxes

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
# Neural baseline was re-run with --no-score-cache to get threshold-specific
# latencies (the original cached path stored one latency value and reused it
# across all thresholds). Prefer the no-cache dirs when present.
nocache_path = Path("/data/sonia/conrad/artifacts/benchmark")

# Configuration (mirrors abstention.py)
query_type = "ThreeHopPipeline"  # For 3p queries
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
hybrid_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]

confidence_levels = None  # Filled dynamically from CSVs
width = 0.6

# Conrad at 40% sparsity has runaway latencies (902 ms at α=0.8, 4499 ms at
# α=0.9). Use a broken y-axis with three segments so the bulk of the data
# (0–200 ms) is readable while still showing the tall outlier bars.
# brokenaxes expects ylims in increasing order (bottom segment first); it
# reverses them internally when wiring the gridspec.
Y_SEGMENTS = ((0, 200), (850, 950), (4400, 4550))
# height_ratios is in matplotlib grid order (top-to-bottom). The high-range
# outlier segments at the top get a thin strip; the 0–200 segment at the
# bottom gets the bulk of the height.
Y_HEIGHT_RATIOS_GRID = (1, 1, 4)

# Regex to pull "Avg Time: 136.20ms" from the SUMMARY block.
# Conrad logs print both "All queries" and "Non-abstained only" — use All queries
# so that abstaining queries still count against the average (parallel to how
# baseline avg_time_ms is reported over all queries).
conrad_avg_time_pattern = re.compile(
    r"OVERALL AVERAGE \(All queries\):.*?Avg Time:\s*([\d.]+)\s*ms"
)

# ============================================================================
# DATA LOADING
# ============================================================================

def load_conrad_latency(dataset, query_type, sparsity):
    """Parse Conrad avg latency (ms) per confidence level from benchmark_conf_*.log."""
    dir_name = f"conrad_bench_{dataset}_{query_type}_{sparsity}"
    dir_path = base_path / dir_name
    if not dir_path.exists():
        print(f"Conrad not found: {dir_name}")
        return {}

    out = {}
    for log_path in sorted(dir_path.glob("benchmark_conf_*.log")):
        m = re.search(r"benchmark_conf_([\d.]+)\.log", log_path.name)
        if not m:
            continue
        conf = float(m.group(1))
        try:
            with open(log_path, "r") as f:
                text = f.read()
            match = conrad_avg_time_pattern.search(text)
            if match:
                out[conf] = float(match.group(1))
            else:
                print(f"  No 'OVERALL AVERAGE (All queries)' summary in {log_path.name}")
        except Exception as e:
            print(f"  Error reading {log_path}: {e}")
    return out

def load_neural_latency(dataset, query_type, sparsity):
    # Prefer the --no-score-cache re-run (threshold-specific latencies) when present.
    nocache_csv = (nocache_path / f"neural_bench_{dataset}_{query_type}_{sparsity}_nocache"
                   / "baseline_results_summary.csv")
    dir_name = f"neural_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = nocache_csv if nocache_csv.exists() else (base_path / dir_name / "baseline_results_summary.csv")
    if not csv_path.exists():
        print(f"Neural not found: {dir_name}")
        return {}
    if csv_path == nocache_csv:
        print(f"  Using no-cache neural results: {nocache_csv.parent.name}")
    df = pd.read_csv(csv_path)
    neural_rows = df[df["baseline"] == "neural"]
    return {row["threshold"]: row["avg_time_ms"] for _, row in neural_rows.iterrows()
            if row["threshold"] in neural_thresholds}

def load_hybrid_latency(dataset, query_type, sparsity):
    dir_name = f"hybrid_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    if not csv_path.exists():
        print(f"Hybrid not found: {dir_name}")
        return {}
    df = pd.read_csv(csv_path)
    hybrid_rows = df[df["baseline"] == "hybrid"]
    return {row["threshold"]: row["avg_time_ms"] for _, row in hybrid_rows.iterrows()
            if row["threshold"] in hybrid_thresholds}

def load_symbolic_latency(dataset, query_type, sparsity):
    dir_name = f"symbolic_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    if not csv_path.exists():
        print(f"Symbolic not found: {dir_name}")
        return np.nan
    df = pd.read_csv(csv_path)
    row = df[df["baseline"] == "symbolic"].iloc[0]
    return row["avg_time_ms"]

# ============================================================================
# COLLECT DATA
# ============================================================================

datasets = {}
missing_levels = [5, 20, 40]
dataset_configs = [
    ("FB15k-237", "fb15k-237"),
    # ('NELL-995', 'nell-955'),
    # ('YAGO3-10', 'yago310')
]

# Discover confidence levels across all Conrad logs (same as abstention.py)
all_confidence_levels = set()
for display_name, dataset_key in dataset_configs:
    for m_level in missing_levels:
        d = base_path / f"conrad_bench_{dataset_key}_{query_type}_{m_level}"
        for log_path in d.glob("benchmark_conf_*.log") if d.exists() else []:
            mm = re.search(r"benchmark_conf_([\d.]+)\.log", log_path.name)
            if mm:
                all_confidence_levels.add(float(mm.group(1)))

confidence_levels = sorted(all_confidence_levels)
print(f"Found confidence levels: {confidence_levels}")

x_labels = [str(c) for c in confidence_levels]
x = np.arange(len(x_labels))

for display_name, dataset_key in dataset_configs:
    datasets[display_name] = {}
    for m_level in missing_levels:
        print(f"\nLoading {display_name} {m_level}%...")
        datasets[display_name][m_level] = {
            "conrad": load_conrad_latency(dataset_key, query_type, m_level),
            "symbolic": load_symbolic_latency(dataset_key, query_type, m_level),
            "neural": load_neural_latency(dataset_key, query_type, m_level),
            "hybrid": load_hybrid_latency(dataset_key, query_type, m_level),
        }
        print(f"  Conrad:   {datasets[display_name][m_level]['conrad']}")
        print(f"  Symbolic: {datasets[display_name][m_level]['symbolic']}")
        print(f"  Neural:   {datasets[display_name][m_level]['neural']}")
        print(f"  Hybrid:   {datasets[display_name][m_level]['hybrid']}")

dataset_names = [name for name, _ in dataset_configs]

# ============================================================================
# PLOTTING (mirrors abstention.py)
# ============================================================================

def plot_panel(bax, data, is_leftmost, is_bottom, m_level):
    """Draw Conrad bars + baseline horizontal lines onto a BrokenAxes object."""
    # 1. Conrad as bars
    conrad_dict = data["conrad"]
    conrad_vals = [conrad_dict.get(conf, np.nan) for conf in confidence_levels]
    bax.bar(x, conrad_vals, width, label="conrad",
            color="lightgray", edgecolor="black", zorder=6)

    # 2. Symbolic as a horizontal dashed line
    if not np.isnan(data["symbolic"]):
        bax.axhline(y=data["symbolic"], color="black", linestyle="--",
                    linewidth=2.5, alpha=0.7, zorder=3, label="neo4j")

    # 3. Neural as horizontal lines (red shades). With the no-cache re-run,
    #    neural latency now varies modestly with θ (stricter θ → earlier
    #    pruning → less GPU work).
    neural_colors = ["#942c4a", "#ca4641", "#e57d56", "#edad8d"]
    for i, thresh in enumerate(neural_thresholds):
        val = data["neural"].get(thresh)
        if val is None or np.isnan(val):
            continue
        bax.axhline(y=val, color=neural_colors[i], linestyle="-",
                    linewidth=2.5, label=f"neural (θ={thresh})")

    # 4. Hybrid as horizontal lines (blue shades)
    hybrid_colors = ["#08306b", "#2171b5", "#4292c6", "#6baed6", "#9ecae1"]
    for thresh in hybrid_thresholds:
        val = data["hybrid"].get(thresh)
        if val is None or np.isnan(val):
            continue
        color_idx = hybrid_thresholds.index(thresh)
        bax.axhline(y=val, color=hybrid_colors[color_idx], linestyle=":",
                    linewidth=2.5, label=f"hybrid (θ={thresh})")

    # bax.axs is in grid order (top-to-bottom). The top two sub-axes show
    # the outlier segments, the bottom sub-axis shows the 0–200 ms bulk.
    for sub_ax in bax.axs:
        sub_ax.set_xticks(x)
        sub_ax.set_xticklabels(x_labels)
        sub_ax.set_xlim(-0.5, len(confidence_levels) - 0.5)
        sub_ax.grid(True, alpha=0.6, zorder=0)

    bax.set_title(f"{m_level}% sparsity", fontsize=10, pad=8)
    if is_bottom:
        bax.set_xlabel("Target Recall", fontsize=10, labelpad=20)
    if is_leftmost:
        bax.set_ylabel("Avg Latency (ms)", fontsize=10, labelpad=30)


nrows = len(dataset_names)
fig = plt.figure(figsize=(7.5, 4.0 * nrows))
outer_gs = gridspec.GridSpec(nrows, 3, figure=fig, wspace=0.4, hspace=0.4,
                              top=0.78, bottom=0.1, left=0.08, right=0.98)

bax_grid = [[None] * 3 for _ in range(nrows)]
for row_idx, d_name in enumerate(dataset_names):
    for col_idx, m_level in enumerate(missing_levels):
        bax = brokenaxes(
            ylims=Y_SEGMENTS,
            height_ratios=Y_HEIGHT_RATIOS_GRID,
            hspace=0.08,
            despine=False,
            subplot_spec=outer_gs[row_idx, col_idx],
            d=0.008,
        )
        bax_grid[row_idx][col_idx] = bax
        plot_panel(
            bax,
            datasets[d_name][m_level],
            is_leftmost=(col_idx == 0),
            is_bottom=(row_idx == nrows - 1),
            m_level=m_level,
        )

# Legend gathered from a panel that has all methods (the 20% column shows
# every neural & hybrid threshold).
handles, labels = bax_grid[0][1].axs[-1].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.0),
           ncols=4, fontsize=9, frameon=True)

plt.savefig("conrad_latency.png", dpi=300, bbox_inches="tight")
plt.savefig("conrad_latency.pdf", dpi=300, bbox_inches="tight")
print("\nPlot saved to conrad_latency.png")

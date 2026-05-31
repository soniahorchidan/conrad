"""Sanity-check plot: avg latency vs empirical recall, per method.

For each method we get a sequence of (recall, latency) points by sweeping the
method's knob (Conrad → α; neural → θ; hybrid → θ; symbolic → single point).
A method that's "doing useful work" should fall on or near the Pareto front
of higher recall costing more latency. This is the standard quality/cost
scatter and complements the bar plot in latency.py.

Generates one figure per pipeline (3p, 2u, 2ip).
"""
import re
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

base_path = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
# Neural baseline was re-run with --no-score-cache to get threshold-specific
# latencies. Prefer the no-cache dirs when present.
nocache_path = Path("/data/sonia/conrad/artifacts/benchmark")

pipelines = [
    ("ThreeHopPipeline", "3p", "three_hop"),
]
missing_levels = [5, 20, 40]
dataset_configs = [("FB15k-237", "fb15k-237")]

conrad_avg_time_pattern = re.compile(
    r"OVERALL AVERAGE \(All queries\):.*?Recall=([\d.]+).*?Avg Time:\s*([\d.]+)\s*ms"
)

def load_conrad(dataset, query_type, sparsity):
    """Returns list of (alpha, recall, latency_ms), sorted by alpha."""
    dir_path = base_path / f"conrad_bench_{dataset}_{query_type}_{sparsity}"
    if not dir_path.exists():
        return []
    out = []
    for log_path in sorted(dir_path.glob("benchmark_conf_*.log")):
        m = re.search(r"benchmark_conf_([\d.]+)\.log", log_path.name)
        if not m:
            continue
        alpha = float(m.group(1))
        text = log_path.read_text()
        mm = conrad_avg_time_pattern.search(text)
        if mm:
            out.append((alpha, float(mm.group(1)), float(mm.group(2))))
    return sorted(out)

def load_baseline(method, dataset, query_type, sparsity):
    """Returns list of (knob, recall, latency_ms) for the given baseline."""
    dir_path = base_path / f"{method}_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = dir_path / "baseline_results_summary.csv"
    if method == "neural":
        nocache_csv = (nocache_path
                       / f"neural_bench_{dataset}_{query_type}_{sparsity}_nocache"
                       / "baseline_results_summary.csv")
        if nocache_csv.exists():
            csv_path = nocache_csv
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    rows = df[df["baseline"] == method]
    out = []
    for _, r in rows.iterrows():
        knob = r["threshold"] if method != "symbolic" else None
        out.append((knob, r["recall"], r["avg_time_ms"]))
    if method != "symbolic":
        out.sort(key=lambda t: t[0])
    return out

method_styles = {
    "conrad":   dict(color="black",   marker="o", linestyle="-",  label="conrad"),
    "symbolic": dict(color="#444444", marker="s", linestyle="",   label="neo4j"),
    "neural":   dict(color="#ca4641", marker="^", linestyle="-",  label="neural"),
    "hybrid":   dict(color="#2171b5", marker="D", linestyle="-",  label="hybrid"),
}

def make_figure(query_type, short_label, out_stem):
    nrows = len(dataset_configs)
    fig, axes = plt.subplots(nrows, 3, figsize=(9, 3 * nrows),
                             sharey=True, squeeze=False)

    for row_idx, (d_name, d_key) in enumerate(dataset_configs):
        for col_idx, m_level in enumerate(missing_levels):
            ax = axes[row_idx, col_idx]

            conrad_pts = load_conrad(d_key, query_type, m_level)
            if conrad_pts:
                rec = [p[1] for p in conrad_pts]
                lat = [p[2] for p in conrad_pts]
                ax.plot(rec, lat, **method_styles["conrad"], markersize=8,
                        linewidth=2.5, zorder=5)

            for method in ("neural", "hybrid", "symbolic"):
                pts = load_baseline(method, d_key, query_type, m_level)
                if not pts:
                    continue
                rec = [p[1] for p in pts]
                lat = [p[2] for p in pts]
                style = method_styles[method].copy()
                ax.plot(rec, lat, **style, markersize=7, linewidth=2,
                        alpha=0.85)

            ax.set_yscale("log")
            ax.grid(True, which="major", axis="y", alpha=0.4)
            ax.set_title(f"{m_level}% sparsity", fontsize=16)
            if row_idx == nrows - 1:
                ax.set_xlabel("Empirical Recall", fontsize=16)
            if col_idx == 0:
                ax.set_ylabel("Average Latency", fontsize=16)

    ymin, ymax = axes[0, 0].get_ylim()
    axes[0, 0].set_ylim(ymin, ymax * 1.8)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = {}
    for h, l in zip(handles, labels):
        if l not in by_label:
            by_label[l] = h
    legend_order = ["neo4j", "neural", "hybrid", "conrad"]
    h2 = [by_label[l] for l in legend_order if l in by_label]
    l2 = [l for l in legend_order if l in by_label]
    fig.legend(h2, l2, loc="upper left", bbox_to_anchor=(0.05, 1.15),
               ncols=len(l2), fontsize=16, frameon=True)

    plt.tight_layout()
    png = f"conrad_latency_vs_recall_{out_stem}.png"
    pdf = f"conrad_latency_vs_recall_{out_stem}.pdf"
    plt.savefig(png, dpi=300, bbox_inches="tight")
    plt.savefig(pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {png}")

for query_type, short_label, out_stem in pipelines:
    make_figure(query_type, short_label, out_stem)

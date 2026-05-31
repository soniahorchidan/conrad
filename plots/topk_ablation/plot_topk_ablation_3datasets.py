"""
Top-K calibration ablation across three datasets (FB15k-237, NELL, YAGO),
3p pipeline (ThreeHopPipeline), 20% sparsity.

Two output figures:
  - topk_validity_3datasets.{pdf,png}  : 1x3 grid, empirical recall vs target
  - topk_precision_3datasets.{pdf,png} : 1x3 grid, precision vs target

Data sources:
  - Top-K=1000 (paper default): /data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized
  - Top-K=10/100:               /data/fabian/benchmark_top{10,100}
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from scipy import stats

BASELINE_BASE = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
ABLATION_BASES = {
    10:  Path("/data/fabian/benchmark_top10"),
    100: Path("/data/fabian/benchmark_top100"),
}

PIPELINE = "ThreeHopPipeline"
DATASETS = [
    ("fb15k-237", "FB15k-237"),
    ("nell-955",  "NELL-995"),
    ("yago310",   "YAGO3-10"),
]

TOPK_VALUES = [10, 100, 1000]
TOPK_COLORS  = {10: "#1f77b4", 100: "#2ca02c", 1000: "#ff7f0e"}
TOPK_MARKERS = {10: "o",        100: "s",       1000: "^"}

CONFIDENCE_LEVELS = [0.6, 0.7, 0.8, 0.9]
SPARSITY = 20

CI_LEVEL = 0.95
Z_SCORE  = stats.norm.ppf((1 + CI_LEVEL) / 2)

OUT_DIR = Path(__file__).parent


def results_dir(topk, dataset_key):
    base = BASELINE_BASE if topk == 1000 else ABLATION_BASES[topk]
    return base / f"conrad_bench_{dataset_key}_{PIPELINE}_{SPARSITY}"


def load_summary(topk, dataset_key):
    csv_path = results_dir(topk, dataset_key) / "results_summary.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return (df[df["confidence"].isin(CONFIDENCE_LEVELS)]
              .sort_values("confidence").reset_index(drop=True))


def ci_band(x, n):
    se = np.sqrt(x * (1 - x) / n)
    return np.maximum(0, x - Z_SCORE * se), np.minimum(1, x + Z_SCORE * se)


def plot_validity():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)
    x_band = np.linspace(0.55, 0.95, 200)

    for col, (key, label) in enumerate(DATASETS):
        ax = axes[col]

        ns = []
        for tk in TOPK_VALUES:
            df = load_summary(tk, key)
            if df is not None and len(df) > 0:
                ns.append(int(df["num_queries"].iloc[0]))
        if ns:
            lo, hi = ci_band(x_band, min(ns))
            ax.fill_between(x_band, lo, hi, alpha=0.18, color="gray", zorder=0,
                            label=f"{int(CI_LEVEL*100)}% CI (n={min(ns)})")
        ax.plot(x_band, x_band, linestyle="--", color="red", linewidth=1.6,
                zorder=1, label="Ideal")

        for tk in TOPK_VALUES:
            df = load_summary(tk, key)
            if df is None or df.empty:
                continue
            ax.plot(df["confidence"], df["recall"],
                    marker=TOPK_MARKERS[tk], color=TOPK_COLORS[tk],
                    linewidth=2.0, markersize=7, zorder=2,
                    label=f"Top-K={tk}")

        ax.set_title(label, fontsize=11)
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.55, 1.0)
        ax.set_xticks(CONFIDENCE_LEVELS)
        ax.grid(True, alpha=0.5)
        ax.set_xlabel("Target Recall (confidence)")
        if col == 0:
            ax.set_ylabel("Empirical Recall")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               fontsize=9, bbox_to_anchor=(0.5, 1.05))

    fig.suptitle("Top-K ablation — Validity (3p pipeline, 20% sparsity)",
                 fontsize=12, y=1.10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topk_validity_3datasets.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "topk_validity_3datasets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_precision():
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=True, sharey=True)

    best_per_dataset = {}

    for col, (key, label) in enumerate(DATASETS):
        ax = axes[col]
        best = None  # (precision, target_recall, topk)
        for tk in TOPK_VALUES:
            df = load_summary(tk, key)
            if df is None or df.empty:
                continue
            ax.plot(df["confidence"], df["precision"],
                    marker=TOPK_MARKERS[tk], color=TOPK_COLORS[tk],
                    linewidth=2.0, markersize=7,
                    label=f"Top-K={tk}")
            row_max = df.loc[df["precision"].idxmax()]
            cand = (float(row_max["precision"]), float(row_max["confidence"]), tk)
            if best is None or cand[0] > best[0]:
                best = cand
        best_per_dataset[label] = best

        ax.set_title(label, fontsize=11)
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.55, 1.0)
        ax.set_xticks(CONFIDENCE_LEVELS)
        ax.grid(True, alpha=0.5)
        ax.set_xlabel("Target Recall (confidence)")
        if col == 0:
            ax.set_ylabel("Precision")

    handles = [Line2D([0], [0], color=TOPK_COLORS[tk],
                      marker=TOPK_MARKERS[tk], linewidth=2.0, markersize=7)
               for tk in TOPK_VALUES]
    labels = [f"Top-K={tk}" for tk in TOPK_VALUES]
    fig.legend(handles, labels, loc="upper center", ncol=len(handles),
               fontsize=9, bbox_to_anchor=(0.5, 1.04))

    fig.suptitle("Top-K ablation — Precision (3p pipeline, 20% sparsity)",
                 fontsize=12, y=1.10)
    plt.tight_layout()
    plt.savefig(OUT_DIR / "topk_precision_3datasets.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "topk_precision_3datasets.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\nBest precision per dataset:")
    print(f"{'Dataset':<12} {'Precision':>10}  {'Target recall':>14}  {'Top-K':>6}")
    for label, best in best_per_dataset.items():
        if best is None:
            print(f"{label:<12} {'n/a':>10}")
            continue
        p, conf, tk = best
        print(f"{label:<12} {p:>10.4f}  {conf:>14.1f}  {tk:>6d}")


if __name__ == "__main__":
    plot_validity()
    plot_precision()
    print("Wrote:")
    for f in sorted(OUT_DIR.glob("topk_*_3datasets.*")):
        print(f"  {f.name}")

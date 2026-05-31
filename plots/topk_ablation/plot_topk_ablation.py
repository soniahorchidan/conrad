"""
Ablation plots for the Top-K parameter (number of candidates retained from
ULTRA scoring per hop).

We focus on the 3p (ThreeHopPipeline) query, FB15k-237, 20% sparsity.

Outputs:
  - topk_ablation_3p.pdf/png : single figure, two subplots
      (left)  validity  — empirical recall vs target recall, one curve per K
      (right) precision — precision vs target recall, one curve per K
  - overhead_table.md / .csv : calibration time and total inference time
                               (sum across all confidence levels)

Data sources:
  - Top-K=1000 (default): /data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized
                          (1000 test queries)
  - Top-K=10/100:         /data/fabian/benchmark_top{10,100}
                          (2000 test queries)
"""

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BASELINE_BASE = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
ABLATION_BASES = {
    10:  Path("/data/fabian/benchmark_top10"),
    100: Path("/data/fabian/benchmark_top100"),
}

PIPELINE = "ThreeHopPipeline"
PIPELINE_LABEL = "3p"

TOPK_VALUES = [10, 100, 1000]
TOPK_COLORS = {10: "#1f77b4", 100: "#2ca02c", 1000: "#ff7f0e"}
TOPK_MARKERS = {10: "o", 100: "s", 1000: "^"}

CONFIDENCE_LEVELS = [0.6, 0.7, 0.8, 0.9]
DATASET = "fb15k-237"
SPARSITY = 20

CI_LEVEL = 0.95
Z_SCORE = stats.norm.ppf((1 + CI_LEVEL) / 2)

OUT_DIR = Path(__file__).parent


# Datasets covered by the top-K ablation (all for the 3p / ThreeHopPipeline query).
ABLATION_DATASETS = ["fb15k-237", "nell-955", "yago310"]


def results_dir(topk, dataset=DATASET):
    base = BASELINE_BASE if topk == 1000 else ABLATION_BASES[topk]
    return base / f"conrad_bench_{dataset}_{PIPELINE}_{SPARSITY}"


def load_summary(topk, dataset=DATASET):
    csv_path = results_dir(topk, dataset) / "results_summary.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    return df[df["confidence"].isin(CONFIDENCE_LEVELS)].sort_values("confidence").reset_index(drop=True)


# ---------------- Wall-clock parsing ----------------

TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")


def first_last_timestamp(log_path):
    if not log_path.exists():
        return None, None
    first, last = None, None
    with open(log_path) as fh:
        for line in fh:
            m = TS_RE.match(line)
            if m:
                ts = datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
                if first is None:
                    first = ts
                last = ts
    return first, last


def calibration_seconds(topk):
    log_path = results_dir(topk) / "calibration.log"
    first, last = first_last_timestamp(log_path)
    if first is None or last is None:
        return None
    return (last - first).total_seconds()


BENCH_TIME_RE = re.compile(r"Total time to process all benchmark queries: ([\d.]+)s")
BENCH_NUM_RE = re.compile(r"for (\d+) queries")


def benchmark_seconds(topk):
    """Sum across all conf. levels. Also returns num_queries seen."""
    d = results_dir(topk)
    total = 0.0
    nq = None
    found = False
    for conf in CONFIDENCE_LEVELS:
        log_path = d / f"benchmark_conf_{conf}.log"
        if not log_path.exists():
            continue
        with open(log_path) as fh:
            for line in fh:
                m = BENCH_TIME_RE.search(line)
                if m:
                    total += float(m.group(1))
                    found = True
                    n_match = BENCH_NUM_RE.search(line)
                    if n_match:
                        nq = int(n_match.group(1))
                    break
    return (total if found else None), nq


# ---------------- Plot ----------------

def ci_band(x, n):
    se = np.sqrt(x * (1 - x) / n)
    return np.maximum(0, x - Z_SCORE * se), np.minimum(1, x + Z_SCORE * se)


def plot_3p():
    fig, (ax_val, ax_prec) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # ---- Validity ----
    x_band = np.linspace(0.55, 0.95, 200)
    # CI band based on smallest n among the runs (conservative)
    ns = []
    for tk in TOPK_VALUES:
        df = load_summary(tk)
        if df is not None and len(df) > 0:
            ns.append(int(df["num_queries"].iloc[0]))
    if ns:
        lo, hi = ci_band(x_band, min(ns))
        ax_val.fill_between(x_band, lo, hi, alpha=0.18, color="gray", zorder=0,
                            label=f"{int(CI_LEVEL*100)}% CI band (n={min(ns)})")
    ax_val.plot(x_band, x_band, linestyle="--", color="red", linewidth=1.6,
                zorder=1, label="Ideal")

    for tk in TOPK_VALUES:
        df = load_summary(tk)
        if df is None or df.empty:
            continue
        ax_val.plot(df["confidence"], df["recall"],
                    marker=TOPK_MARKERS[tk], color=TOPK_COLORS[tk],
                    linewidth=2.0, markersize=7, zorder=2,
                    label=f"Top-K={tk}")

    ax_val.set_xlim(0.55, 0.95)
    ax_val.set_ylim(0.55, 1.0)
    ax_val.set_xticks(CONFIDENCE_LEVELS)
    ax_val.set_xlabel("Target Recall (confidence)")
    ax_val.set_ylabel("Empirical Recall")
    ax_val.set_title("Validity", fontsize=11)
    ax_val.grid(True, alpha=0.5)
    ax_val.legend(loc="lower right", fontsize=8)

    # ---- Precision ----
    for tk in TOPK_VALUES:
        df = load_summary(tk)
        if df is None or df.empty:
            continue
        ax_prec.plot(df["confidence"], df["precision"],
                     marker=TOPK_MARKERS[tk], color=TOPK_COLORS[tk],
                     linewidth=2.0, markersize=7,
                     label=f"Top-K={tk}")

    ax_prec.set_xlim(0.55, 0.95)
    ax_prec.set_ylim(0.6, 1.0)
    ax_prec.set_xticks(CONFIDENCE_LEVELS)
    ax_prec.set_xlabel("Target Recall (confidence)")
    ax_prec.set_ylabel("Precision")
    ax_prec.set_title("Precision", fontsize=11)
    ax_prec.grid(True, alpha=0.5)
    ax_prec.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"Top-K ablation — {PIPELINE_LABEL} pipeline (FB15k-237, 20% sparsity)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(OUT_DIR / "topk_ablation_3p.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "topk_ablation_3p.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- Overhead table ----------------

def overhead_table():
    rows = []
    for tk in TOPK_VALUES:
        cal = calibration_seconds(tk)
        bench, nq = benchmark_seconds(tk)
        per_1k = None
        if bench is not None and nq:
            # Per 1k queries × number of conf levels we summed
            per_1k = bench / (nq / 1000.0) / len(CONFIDENCE_LEVELS)
        rows.append({
            "top_k": tk,
            "num_test_queries": nq,
            "calibration_min": None if cal is None else round(cal / 60, 2),
            "inference_total_s": None if bench is None else round(bench, 1),
            "inference_per_1k_per_conf_s": None if per_1k is None else round(per_1k, 1),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "overhead_table.csv", index=False)

    lines = [
        "| Top-K | # test queries | calibration (min) | inference, summed over 4 conf. levels (s) | per-1k-queries × per-conf-level (s) |",
        "|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['top_k']} "
            f"| {r['num_test_queries']} "
            f"| {r['calibration_min']:.2f} "
            f"| {r['inference_total_s']:.1f} "
            f"| {r['inference_per_1k_per_conf_s']:.1f} |"
        )
    md = "\n".join(lines)
    (OUT_DIR / "overhead_table.md").write_text(md + "\n")
    return df, md


# ---------------- Precision difference across top-K configurations ----------------

def precision_difference_table():
    """For each (dataset, confidence), the precision spread across K in {10,100,1000}.

    Returns (rows, max_diff, max_ctx) where max_diff is the largest precision
    spread (max - min over the three K values) across all configurations.
    """
    rows = []
    max_diff = 0.0
    max_ctx = None

    for dataset in ABLATION_DATASETS:
        summaries = {tk: load_summary(tk, dataset) for tk in TOPK_VALUES}
        if any(df is None or df.empty for df in summaries.values()):
            continue
        for conf in CONFIDENCE_LEVELS:
            precs = {}
            for tk in TOPK_VALUES:
                row = summaries[tk].loc[summaries[tk]["confidence"] == conf]
                if row.empty:
                    break
                precs[tk] = float(row["precision"].iloc[0])
            if len(precs) != len(TOPK_VALUES):
                continue

            diff = max(precs.values()) - min(precs.values())
            rows.append({
                "dataset": dataset,
                "confidence": conf,
                **{f"precision_k{tk}": precs[tk] for tk in TOPK_VALUES},
                "precision_diff": diff,
            })
            if diff > max_diff:
                max_diff = diff
                max_ctx = (dataset, conf, dict(precs))

    return rows, max_diff, max_ctx


def print_precision_difference():
    _, max_diff, max_ctx = precision_difference_table()
    print(f"\nMaximum precision difference across all configurations: "
          f"{max_diff:.4f} ({max_diff*100:.2f} pp / {max_diff*100:.1f}%)")
    if max_ctx is not None:
        ds, conf, precs = max_ctx
        vals = ", ".join(f"K={tk}: {precs[tk]:.4f}" for tk in TOPK_VALUES)
        print(f"  attained on {ds} at confidence {conf} ({vals})")


if __name__ == "__main__":
    plot_3p()

    print("=== Top-K ablation (3p pipeline, 20% sparsity) ===")
    print_precision_difference()
    print("\nWrote:")
    for f in sorted(OUT_DIR.glob("topk_ablation_3p.*")):
        print(f"  {f.name}")

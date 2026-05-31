"""
Ablation plots for the `max_hop_size` (intermediate-set sampling cap) parameter.

We focus on the 3p (ThreeHopPipeline) query — the most load-bearing case, since
3p is where the intermediate-set cap actually bites.

Outputs:
  - max_hop_ablation_3p.pdf/png : single figure, two subplots
      (left)  validity  — empirical recall vs target recall, one curve per cap
      (right) precision — precision vs target recall, one curve per cap
  - overhead_table.md / .csv    : calibration time and total inference time
                                  (sum across all confidence levels for 1k queries each)

max_hop=500 is omitted (calibration was estimated to take ~17 h and never finished).

Data sources:
  - max_hop=50  (default): /data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized
  - max_hop=100/200:       /data/fabian/benchmark_max_hop_{100,200}
"""

import re
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

BASELINE_BASE = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")
ABLATION_BASES = {
    100: Path("/data/fabian/benchmark_max_hop_100"),
    200: Path("/data/fabian/benchmark_max_hop_200"),
}

PIPELINE = "ThreeHopPipeline"
PIPELINE_LABEL = "3p"

MAX_HOPS = [50, 100, 200]
MAX_HOP_COLORS = {50: "#1f77b4", 100: "#2ca02c", 200: "#ff7f0e"}
MAX_HOP_MARKERS = {50: "o", 100: "s", 200: "^"}

CONFIDENCE_LEVELS = [0.6, 0.7, 0.8, 0.9]
DATASET = "fb15k-237"
SPARSITY = 20
NUM_QUERIES = 1000

CI_LEVEL = 0.95
Z_SCORE = stats.norm.ppf((1 + CI_LEVEL) / 2)

OUT_DIR = Path(__file__).parent


def results_dir(max_hop):
    base = BASELINE_BASE if max_hop == 50 else ABLATION_BASES[max_hop]
    return base / f"conrad_bench_{DATASET}_{PIPELINE}_{SPARSITY}"


def load_summary(max_hop):
    csv_path = results_dir(max_hop) / "results_summary.csv"
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


def calibration_seconds(max_hop):
    log_path = results_dir(max_hop) / "calibration.log"
    first, last = first_last_timestamp(log_path)
    if first is None or last is None:
        return None
    return (last - first).total_seconds()


BENCH_TIME_RE = re.compile(r"Total time to process all benchmark queries: ([\d.]+)s")


def benchmark_seconds(max_hop):
    """Sum across all confidence levels of the inference time for 1k queries."""
    d = results_dir(max_hop)
    total = 0.0
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
                    break
    return total if found else None


# ---------------- Plot ----------------

def ci_band(x, n):
    se = np.sqrt(x * (1 - x) / n)
    return np.maximum(0, x - Z_SCORE * se), np.minimum(1, x + Z_SCORE * se)


def plot_3p():
    fig, (ax_val, ax_prec) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # ---- Validity ----
    x_band = np.linspace(0.55, 0.95, 200)
    ref = load_summary(50)
    if ref is not None and len(ref) > 0:
        lo, hi = ci_band(x_band, int(ref["num_queries"].iloc[0]))
        ax_val.fill_between(x_band, lo, hi, alpha=0.18, color="gray", zorder=0,
                            label=f"{int(CI_LEVEL*100)}% CI band")
    ax_val.plot(x_band, x_band, linestyle="--", color="red", linewidth=1.6,
                zorder=1, label="Ideal")

    for mh in MAX_HOPS:
        df = load_summary(mh)
        if df is None or df.empty:
            continue
        ax_val.plot(df["confidence"], df["recall"],
                    marker=MAX_HOP_MARKERS[mh], color=MAX_HOP_COLORS[mh],
                    linewidth=2.0, markersize=7, zorder=2,
                    label=f"max_hop_size={mh}")

    ax_val.set_xlim(0.55, 0.95)
    ax_val.set_ylim(0.55, 1.0)
    ax_val.set_xticks(CONFIDENCE_LEVELS)
    ax_val.set_xlabel("Target Recall (confidence)")
    ax_val.set_ylabel("Empirical Recall")
    ax_val.set_title("Validity", fontsize=11)
    ax_val.grid(True, alpha=0.5)
    ax_val.legend(loc="lower right", fontsize=8)

    # ---- Precision ----
    for mh in MAX_HOPS:
        df = load_summary(mh)
        if df is None or df.empty:
            continue
        ax_prec.plot(df["confidence"], df["precision"],
                     marker=MAX_HOP_MARKERS[mh], color=MAX_HOP_COLORS[mh],
                     linewidth=2.0, markersize=7,
                     label=f"max_hop_size={mh}")

    ax_prec.set_xlim(0.55, 0.95)
    ax_prec.set_ylim(0.65, 1.0)
    ax_prec.set_xticks(CONFIDENCE_LEVELS)
    ax_prec.set_xlabel("Target Recall (confidence)")
    ax_prec.set_ylabel("Precision")
    ax_prec.set_title("Precision", fontsize=11)
    ax_prec.grid(True, alpha=0.5)
    ax_prec.legend(loc="lower right", fontsize=8)

    fig.suptitle(f"max_hop_size ablation — {PIPELINE_LABEL} pipeline (FB15k-237, 20% sparsity)",
                 fontsize=12)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.savefig(OUT_DIR / "max_hop_ablation_3p.pdf", dpi=300, bbox_inches="tight")
    plt.savefig(OUT_DIR / "max_hop_ablation_3p.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------- Overhead table ----------------

def overhead_table():
    rows = []
    for mh in MAX_HOPS:
        cal = calibration_seconds(mh)
        bench = benchmark_seconds(mh)
        rows.append({
            "max_hop_size": mh,
            "calibration_s": None if cal is None else round(cal, 1),
            "calibration_min": None if cal is None else round(cal / 60, 2),
            "inference_total_s": None if bench is None else round(bench, 1),
            "inference_per_1k_avg_s": (
                None if bench is None
                else round(bench / len(CONFIDENCE_LEVELS), 1)
            ),
        })
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "overhead_table.csv", index=False)

    # Markdown rendering: cumulative inference time = sum over 4 confidence levels,
    # each level running on 1k queries.
    lines = [
        "| max_hop_size | calibration (min) | inference, 1k queries × 4 conf. levels (s) | per-conf-level avg (s) |",
        "|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| {r['max_hop_size']} "
            f"| {r['calibration_min']:.2f} "
            f"| {r['inference_total_s']:.1f} "
            f"| {r['inference_per_1k_avg_s']:.1f} |"
        )
    md = "\n".join(lines)
    (OUT_DIR / "overhead_table.md").write_text(md + "\n")
    return df, md


if __name__ == "__main__":
    plot_3p()
    df, md = overhead_table()

    print("=== max_hop_size ablation (3p pipeline, FB15k-237, 20% sparsity) ===")
    print("\nOverhead table:")
    print(md)
    print("\nWrote:")
    for f in sorted(OUT_DIR.glob("max_hop_ablation_3p.*")) + \
             sorted(OUT_DIR.glob("overhead_table.*")):
        print(f"  {f.name}")

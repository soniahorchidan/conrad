"""Extract max-precision values at recall >= 0.60 from benchmark CSVs and write a markdown table.

For each (dataset, topology), reports the best configuration per method:
  - neo4j  (symbolic): single threshold-free run; passes or Fails (max recall).
  - neural / hybrid : best precision among thresholds with recall >= RECALL_FLOOR.
  - conrad           : best precision among alphas with recall >= RECALL_FLOOR.

Missing CSVs (e.g. incomplete ConRAD runs) yield blank cells.
"""

import csv
from pathlib import Path

BENCH_DIRS = [
    Path("/data/sonia/conrad/artifacts/benchmark_revision"),
    Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized"),
]
OUTPUT_PATH = Path(__file__).parent / "baselines_table.md"
RECALL_FLOOR = 0.60
SPARSITY = 20


def find_file(subdir: str, filename: str):
    """Return the first existing path across BENCH_DIRS, or None."""
    for base in BENCH_DIRS:
        candidate = base / subdir / filename
        if candidate.exists():
            return candidate
    return None

DATASETS = [
    ("FB15k-237", "fb15k-237"),
    ("NELL-995", "nell-955"),
    ("YAGO3-10", "yago310"),
]

TOPOLOGIES = [
    ("2p",  "TwoHopPipeline"),
    ("3p",  "ThreeHopPipeline"),
    ("2i",  "TwoIntersectPipeline"),
    ("3i",  "ThreeIntersectPipeline"),
    ("2ip", "TwoIntersectProjectPipeline"),
    ("pi",  "ProjectIntersectPipeline"),
    ("2u",  "TwoUnionPipeline"),
    ("up",  "UnionProjectPipeline"),
]


def read_csv(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def best_baseline(rows, knob_col, knob_label):
    """Return (precision, knob_value) for the row with max precision and recall >= floor,
    or ('Fails', max_recall) if none qualifies. Returns None if rows is None/empty."""
    if not rows:
        return None
    qualifying = [r for r in rows if float(r["recall"]) >= RECALL_FLOOR]
    if not qualifying:
        max_r = max(float(r["recall"]) for r in rows)
        return ("Fails", max_r)
    best = max(qualifying, key=lambda r: float(r["precision"]))
    return (float(best["precision"]), best[knob_col])


def extract_symbolic(dataset_slug, pipeline):
    path = find_file(f"symbolic_bench_{dataset_slug}_{pipeline}_{SPARSITY}", "baseline_results_summary.csv")
    rows = read_csv(path) if path else None
    if not rows:
        return None
    r = rows[0]
    p, rec = float(r["precision"]), float(r["recall"])
    if rec < RECALL_FLOOR:
        return ("Fails", rec)
    return (p, None)


def extract_baseline(dataset_slug, pipeline, method):
    path = find_file(f"{method}_bench_{dataset_slug}_{pipeline}_{SPARSITY}", "baseline_results_summary.csv")
    return best_baseline(read_csv(path) if path else None, "threshold", "theta")


def extract_conrad(dataset_slug, pipeline):
    path = find_file(f"conrad_bench_{dataset_slug}_{pipeline}_{SPARSITY}", "results_summary.csv")
    return best_baseline(read_csv(path) if path else None, "confidence", "alpha")


def fmt_cell(result, knob_symbol):
    if result is None:
        return ""
    val, knob = result
    if val == "Fails":
        return f"Fails ({knob:.2f})"
    if knob is None:
        return f"{val:.2f}"
    return f"{val:.2f} ({knob_symbol}={float(knob):.1f})"


def numeric_value(result):
    """Return the numeric precision if the row qualified, else None."""
    if result is None or result[0] == "Fails":
        return None
    return result[0]


def build_rows():
    table = []
    for dataset_label, dataset_slug in DATASETS:
        for topo_label, pipeline in TOPOLOGIES:
            cells = {
                "neo4j":  extract_symbolic(dataset_slug, pipeline),
                "neural": extract_baseline(dataset_slug, pipeline, "neural"),
                "hybrid": extract_baseline(dataset_slug, pipeline, "hybrid"),
                "conrad": extract_conrad(dataset_slug, pipeline),
            }
            table.append((dataset_label, topo_label, cells))
    return table


def bold_best(formatted, cells):
    """Bold the cell with the max precision. Ties go to the rightmost column
    (conrad > hybrid > neural > neo4j) to match the paper convention."""
    order = ["neo4j", "neural", "hybrid", "conrad"]
    best_col, best_val = None, -1.0
    for col in order:
        v = numeric_value(cells[col])
        if v is not None and v >= best_val:
            best_col, best_val = col, v
    if best_col is None:
        return formatted
    formatted[best_col] = f"**{formatted[best_col]}**"
    return formatted


def write_markdown(table, path: Path):
    lines = [
        f"# Baselines: max precision at recall >= {RECALL_FLOOR:.2f} ({SPARSITY}% incompleteness)",
        "",
        "| Dataset | Topology | neo4j | neural (best θ) | hybrid (best θ) | conrad (best α) |",
        "|---|---|---|---|---|---|",
    ]
    prev_dataset = None
    for dataset, topo, cells in table:
        formatted = {
            "neo4j":  fmt_cell(cells["neo4j"], "θ"),
            "neural": fmt_cell(cells["neural"], "θ"),
            "hybrid": fmt_cell(cells["hybrid"], "θ"),
            "conrad": fmt_cell(cells["conrad"], "α"),
        }
        formatted = bold_best(formatted, cells)
        dataset_cell = dataset if dataset != prev_dataset else ""
        prev_dataset = dataset
        lines.append(
            f"| {dataset_cell} | `{topo}` | {formatted['neo4j']} | "
            f"{formatted['neural']} | {formatted['hybrid']} | {formatted['conrad']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    table = build_rows()
    write_markdown(table, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

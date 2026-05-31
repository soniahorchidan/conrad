"""Extract max-precision values at recall >= 0.60 for the 3p query on FB15k-237 and NELL-995
at 20% incompleteness, comparing ConRAD against the neo4j / neural / hybrid baselines.

Mirrors the logic of plots/efficiency/extract_baselines_table.py but restricted to the
single topology (3p / ThreeHopPipeline) and the two stratified datasets.
"""

import csv
from pathlib import Path

BENCH_DIR = Path("/data/sonia/conrad/artifacts/benchmark")
OUTPUT_PATH = Path(__file__).parent / "precision_3p_table.md"
RECALL_FLOOR = 0.60
SPARSITY = 20
PIPELINE = "ThreeHopPipeline"

DATASETS = [
    ("FB15k-237", "fb15k-237"),
    ("NELL-995",  "nell-955"),
]


def read_csv(path: Path):
    if not path.exists():
        return None
    with path.open() as f:
        return list(csv.DictReader(f))


def best_baseline(rows, knob_col):
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


def extract_symbolic(dataset_slug):
    path = BENCH_DIR / f"symbolic_bench_{dataset_slug}_{PIPELINE}_{SPARSITY}" / "baseline_results_summary.csv"
    rows = read_csv(path)
    if not rows:
        return None
    r = rows[0]
    p, rec = float(r["precision"]), float(r["recall"])
    if rec < RECALL_FLOOR:
        return ("Fails", rec)
    return (p, None)


def extract_baseline(dataset_slug, method):
    path = BENCH_DIR / f"{method}_bench_{dataset_slug}_{PIPELINE}_{SPARSITY}" / "baseline_results_summary.csv"
    return best_baseline(read_csv(path), "threshold")


def extract_conrad(dataset_slug):
    path = BENCH_DIR / f"conrad_bench_{dataset_slug}_{PIPELINE}_{SPARSITY}" / "results_summary.csv"
    return best_baseline(read_csv(path), "confidence")


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
    if result is None or result[0] == "Fails":
        return None
    return result[0]


def bold_best(formatted, cells):
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


def build_rows():
    table = []
    for dataset_label, dataset_slug in DATASETS:
        cells = {
            "neo4j":  extract_symbolic(dataset_slug),
            "neural": extract_baseline(dataset_slug, "neural"),
            "hybrid": extract_baseline(dataset_slug, "hybrid"),
            "conrad": extract_conrad(dataset_slug),
        }
        table.append((dataset_label, cells))
    return table


def write_markdown(table, path: Path):
    lines = [
        f"# 3p precision: best precision at recall >= {RECALL_FLOOR:.2f} ({SPARSITY}% incompleteness)",
        "",
        "| Dataset | neo4j | neural (best θ) | hybrid (best θ) | conrad (best α) |",
        "|---|---|---|---|---|",
    ]
    for dataset, cells in table:
        formatted = {
            "neo4j":  fmt_cell(cells["neo4j"], "θ"),
            "neural": fmt_cell(cells["neural"], "θ"),
            "hybrid": fmt_cell(cells["hybrid"], "θ"),
            "conrad": fmt_cell(cells["conrad"], "α"),
        }
        formatted = bold_best(formatted, cells)
        lines.append(
            f"| {dataset} | {formatted['neo4j']} | "
            f"{formatted['neural']} | {formatted['hybrid']} | {formatted['conrad']} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main():
    table = build_rows()
    write_markdown(table, OUTPUT_PATH)
    print(f"Wrote {OUTPUT_PATH}")
    print()
    for dataset, cells in table:
        print(f"{dataset}:")
        for method in ["neo4j", "neural", "hybrid", "conrad"]:
            knob = "α" if method == "conrad" else "θ"
            print(f"  {method:8s} {fmt_cell(cells[method], knob)}")


if __name__ == "__main__":
    main()

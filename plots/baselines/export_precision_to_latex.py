"""
Print raw recall/precision data from baseline CSVs (same data loading as baseline_precision2).
Run with base_path where neural/hybrid/symbolic/conrad CSVs exist.
"""
import pandas as pd
from pathlib import Path

# Same config as baseline_precision2.py
alphas = [0.6, 0.7, 0.8, 0.9]
sparsity = 20
datasets = ["fb15k-237", "nell-955", "yago310"]
dataset_labels = ["FB15K-237", "NELL-955", "YAGO3-10"]
pipelines = [
    ("3p", "ThreeHopPipeline"),
    ("2u", "TwoUnionPipeline"),
    ("2ip", "TwoIntersectProjectPipeline"),
]

# Use path that has all baselines (benchmark_main_objective)
base_path = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")

def load_conrad_data(dataset, sparsity, query_pipeline):
    """Load Conrad recall, precision, and confidence (alpha) from CSV."""
    dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    if not csv_path.exists():
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
        recall = df['recall'].tolist()
        precision = df['precision'].tolist()
        alpha = df['confidence'].tolist()
        return recall, precision, alpha
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

def load_neural_data(dataset, sparsity, query_pipeline):
    dir_name = f"neural_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    if not csv_path.exists():
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
        neural_rows = df[df['baseline'] == 'neural'].sort_values('threshold')
        if len(neural_rows) == 0:
            return None, None, None
        recalls = neural_rows['recall'].tolist()
        precisions = neural_rows['precision'].tolist()
        thresholds = neural_rows['threshold'].tolist()
        return recalls, precisions, thresholds
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

def load_symbolic_data(dataset, sparsity, query_pipeline):
    dir_name = f"symbolic_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    if not csv_path.exists():
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
        symbolic_row = df[df['baseline'] == 'symbolic'].iloc[0]
        return symbolic_row['recall'], symbolic_row['precision'], None
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

def load_hybrid_data(dataset, sparsity, query_pipeline):
    dir_name = f"hybrid_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    if not csv_path.exists():
        return None, None, None
    try:
        df = pd.read_csv(csv_path)
        hybrid_rows = df[(df['baseline'] == 'hybrid') & (df['threshold'] != 0.45)].sort_values('threshold')
        recalls = hybrid_rows['recall'].tolist()
        precisions = hybrid_rows['precision'].tolist()
        thresholds = hybrid_rows['threshold'].tolist()
        return recalls, precisions, thresholds
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

# Load all data: data[dataset][baseline][row] = (recalls, precisions, param) where param is alpha or threshold
data = {}
for dataset in datasets:
    data[dataset] = {"conrad": [], "neural": [], "symbolic": [], "hybrid": []}
    for short_name, query_pipeline in pipelines:
        c_r, c_p, c_a = load_conrad_data(dataset, sparsity, query_pipeline)
        data[dataset]["conrad"].append((c_r, c_p, c_a))
        n_r, n_p, n_t = load_neural_data(dataset, sparsity, query_pipeline)
        data[dataset]["neural"].append((n_r, n_p, n_t))
        s_r, s_p, _ = load_symbolic_data(dataset, sparsity, query_pipeline)
        data[dataset]["symbolic"].append((s_r, s_p, None))
        h_r, h_p, h_t = load_hybrid_data(dataset, sparsity, query_pipeline)
        data[dataset]["hybrid"].append((h_r, h_p, h_t))

def fmt(x):
    if x is None:
        return "?"
    return f"{float(x):.4f}"

# Raw data (recall, precision, param) for each (dataset, pipeline)
print("% ========== Raw data (recall, precision, param) for each (dataset, pipeline) ==========")
for row, (short_name, _) in enumerate(pipelines):
    for col, dataset in enumerate(datasets):
        print(f"% --- {dataset_labels[col]} {short_name} ---")
        s = data[dataset]["symbolic"][row]
        if s[0] is not None:
            print(f"% symbolic: recall={fmt(s[0])}, precision={fmt(s[1])}")
        n = data[dataset]["neural"][row]
        if n[0]:
            for r, p, t in zip(n[0], n[1], n[2]):
                print(f"% neural threshold {t}: recall={fmt(r)}, precision={fmt(p)}")
        h = data[dataset]["hybrid"][row]
        if h[0]:
            for r, p, t in zip(h[0], h[1], h[2]):
                print(f"% hybrid threshold {t}: recall={fmt(r)}, precision={fmt(p)}")
        c = data[dataset]["conrad"][row]
        if c[0]:
            for r, p, a in zip(c[0], c[1], c[2]):
                print(f"% conrad alpha {a}: recall={fmt(r)}, precision={fmt(p)}")
        print()

import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path

# --- Configuration ---
alphas = [0.5, 0.6, 0.7, 0.8, 0.9]
sparsities = ["5% Missing", "20% Missing", "40% Missing"]
sparsity_values = [5, 20, 40]
query_types = ["3p", "2u", "2ip"]
query_type_map = {
    "3p": "ThreeHopPipeline",
    "2u": "TwoUnionPipeline",
    "2ip": "TwoIntersectProjectPipeline"
}
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
neural_markers = ['s', '^', 'D', 'v']  # square, triangle, diamond, inverted triangle
hybrid_thresholds = [0.45, 0.5, 0.6, 0.7]
hybrid_markers = ['*', 'P', 'h', 'H']  # star, plus, hexagon1, hexagon2

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/benchmark")

def load_conrad_data(query_type, sparsity):
    """Load Conrad latency and precision data from CSV and log files."""
    dir_name = f"conrad_bench_fb15k-237_{query_type}_{sparsity}"
    dir_path = base_path / dir_name
    csv_path = dir_path / "results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Conrad CSV path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        precisions = df['precision'].tolist()
        
        # Extract latencies from log files
        latencies = []
        for alpha in alphas:
            log_path = dir_path / f"benchmark_conf_{alpha}.log"
            if not log_path.exists():
                print(f"  Conrad log not found: {log_path}")
                latencies.append(None)
                continue
            
            # Parse the log file for average time
            with open(log_path, 'r') as f:
                content = f.read()
                # Look for: OVERALL AVERAGE (All queries): ... | Avg Time: XXms
                match = re.search(r'OVERALL AVERAGE \(All queries\):.*\| Avg Time: ([\d.]+)ms', content)
                if match:
                    latencies.append(float(match.group(1)))
                else:
                    latencies.append(None)
        
        # Filter out None values (maintain alignment)
        valid_data = [(l, p) for l, p in zip(latencies, precisions) if l is not None]
        if valid_data:
            latencies, precisions = zip(*valid_data)
            return list(latencies), list(precisions)
        return None, None
    except Exception as e:
        print(f"Error loading Conrad data from {dir_path}: {e}")
        return None, None

def load_neural_data(query_type, sparsity):
    """Load Neural baseline latency and precision data from single CSV."""
    dir_name = f"neural_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Neural path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all neural rows (sorted by threshold)
        neural_rows = df[df['baseline'] == 'neural'].sort_values('threshold')
        latencies = neural_rows['avg_time_ms'].tolist()
        precisions = neural_rows['precision'].tolist()
        return latencies, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_symbolic_data(query_type, sparsity):
    """Load Symbolic baseline latency and precision data (single point)."""
    dir_name = f"symbolic_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Symbolic path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get the symbolic row
        symbolic_row = df[df['baseline'] == 'symbolic'].iloc[0]
        return symbolic_row['avg_time_ms'], symbolic_row['precision']
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_hybrid_data(query_type, sparsity):
    """Load Hybrid baseline latency and precision data (multiple threshold points)."""
    dir_name = f"hybrid_bench_fb15k-237_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Hybrid path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all hybrid rows
        hybrid_rows = df[df['baseline'] == 'hybrid']
        latencies = hybrid_rows['avg_time_ms'].tolist()
        precisions = hybrid_rows['precision'].tolist()
        return latencies, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

# --- Data Dictionary ---
# Build data dictionary from CSV files
data = {}
for q_type in query_types:
    query_pipeline = query_type_map[q_type]
    data[q_type] = {
        "conrad": [],
        "neural": [],
        "symbolic": [],
        "hybrid": []
    }
    
    for sparsity in sparsity_values:
        # Load Conrad data
        c_l, c_p = load_conrad_data(query_pipeline, sparsity)
        data[q_type]["conrad"].append((c_l, c_p))
        print(f"Loaded conrad {q_type} {sparsity}%: {len(c_l) if c_l else 0} points")
        
        # Load Neural data
        n_l, n_p = load_neural_data(query_pipeline, sparsity)
        data[q_type]["neural"].append((n_l, n_p))
        print(f"Loaded neural {q_type} {sparsity}%: {len(n_l) if n_l else 0} points")
        
        # Load Symbolic data
        s_l, s_p = load_symbolic_data(query_pipeline, sparsity)
        data[q_type]["symbolic"].append((s_l, s_p))
        print(f"Loaded symbolic {q_type} {sparsity}%: {'yes' if s_l is not None else 'no'}")
        
        # Load Hybrid data
        h_l, h_p = load_hybrid_data(query_pipeline, sparsity)
        data[q_type]["hybrid"].append((h_l, h_p))
        print(f"Loaded hybrid {q_type} {sparsity}%: {len(h_l) if h_l else 0} points")

# --- Plotting ---
fig, axes = plt.subplots(3, 3, figsize=(10, 4.5))

for row, q_type in enumerate(query_types):
    for col, sparsity in enumerate(sparsities):
        ax = axes[row, col]
        
        # Plot Symbolic
        s_data = data[q_type]["symbolic"][col]
        if s_data[0] is not None and s_data[1] is not None:
            s_l, s_p = s_data
            ax.scatter(s_l, s_p, marker='X', color='C2', s=120, edgecolors='black', label='symbolic', zorder=5)

        # Plot Neural Points (different markers for each threshold)
        n_data = data[q_type]["neural"][col]
        if n_data[0] is not None and n_data[1] is not None:
            n_l, n_p = n_data[0], n_data[1]
            for i, (l, p) in enumerate(zip(n_l, n_p)):
                label = f'neural (t={neural_thresholds[i]})' if row == 0 and col == 0 else None
                ax.scatter(l, p, marker=neural_markers[i], color='C1', s=50, 
                          edgecolors='black', linewidths=1, label=label, zorder=2)
            # Draw dashed line connecting neural points
            ax.plot(n_l, n_p, linestyle='--', color='C1', linewidth=1, alpha=0.5, zorder=3)

        # Plot Hybrid Points (different markers for each threshold)
        h_data = data[q_type]["hybrid"][col]
        if h_data[0] is not None and h_data[1] is not None:
            h_l, h_p = h_data[0], h_data[1]
            for i, (l, p) in enumerate(zip(h_l, h_p)):
                marker_idx = min(i, len(hybrid_markers) - 1)
                label = f'hybrid (t={hybrid_thresholds[marker_idx]})' if row == 0 and col == 0 and i < len(hybrid_thresholds) else None
                ax.scatter(l, p, marker=hybrid_markers[marker_idx], color='C3', s=60, 
                          edgecolors='black', linewidths=1, label=label, zorder=4)
            # Draw dashed line connecting hybrid points
            ax.plot(h_l, h_p, linestyle='--', color='C3', linewidth=1, alpha=0.5, zorder=3)

        # Plot Conrad Curve
        c_data = data[q_type]["conrad"][col]
        if c_data[0] is not None and c_data[1] is not None:
            c_l, c_p = c_data[0], c_data[1]
            ax.plot(c_l, c_p, marker='o', linestyle='-', color='C0', linewidth=2.5, label='conrad')
            for i, a in enumerate(alphas[:len(c_l)]):
                ax.annotate(f'α={a}', (c_l[i], c_p[i]), textcoords="offset points", 
                            xytext=(0, 10), fontsize=7, color='C0', fontweight='bold', ha='center')

        # Axis limits and Titles
        ax.set_ylim(0.1, 1.2)
            
        if row == 0: ax.set_title(sparsity, fontsize=12)
        if col == 0: ax.set_ylabel(f"{q_type.upper()}\nPrecision", fontsize=12)
        if row == 2: ax.set_xlabel("Avg Query Latency (ms)", fontsize=12)
        
        ax.grid(True)
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
# Create figure-level legend positioned at the top center
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=8.5, bbox_to_anchor=(0.5, 1.035))
plt.savefig('baselines_latency_fb15k237_from_csv.png', dpi=300, bbox_inches='tight')
print("Plot saved to baselines_latency_fb15k237_from_csv.png")

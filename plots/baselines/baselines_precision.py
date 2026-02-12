import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# --- Configuration ---
alphas = [0.6, 0.7, 0.8, 0.9]
sparsities = ["5% sparsity", "20% sparsity", "40% sparsity"]
sparsity_values = [5, 20, 40]
datasets = ["fb15k-237", "nell-955", "yago310"]
dataset_labels = ["FB15K-237", "NELL-955", "YAGO3-10"]
query_pipeline = "ThreeHopPipeline"  # 3p only
conrad_markers = ['*', '^', 'h', 'd', '>'] #['o', 'p', 'd', '8', '>']  # circle, pentagon, thin diamond, octagon, right triangle
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
neural_markers = ['*', '^', 'h', 'd', '>']  #['s', '^', 'D', 'v']  # square, triangle, diamond, inverted triangle
hybrid_thresholds = [0.45, 0.5, 0.6, 0.7]
hybrid_markers = ['*', '^', 'h', 'd', '>']  #['*', 'P', 'h', 'H']  # star, plus, hexagon1, hexagon2

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/plots_results")

def load_conrad_data(dataset, sparsity):
    """Load Conrad recall and precision data from CSV file."""
    dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Conrad path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        recall = df['recall'].tolist()
        precision = df['precision'].tolist()
        return recall, precision
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_neural_data(dataset, sparsity):
    """Load Neural baseline data for all thresholds from single CSV."""
    dir_name = f"neural_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Neural path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all neural rows (sorted by threshold)
        neural_rows = df[df['baseline'] == 'neural'].sort_values('threshold')
        if len(neural_rows) == 0:
            print(f"  No neural rows found in {csv_path}")
            return None, None
        recalls = neural_rows['recall'].tolist()
        precisions = neural_rows['precision'].tolist()
        return recalls, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_symbolic_data(dataset, sparsity):
    """Load Symbolic baseline data (single point)."""
    dir_name = f"symbolic_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Symbolic path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get the symbolic row
        symbolic_row = df[df['baseline'] == 'symbolic'].iloc[0]
        return symbolic_row['recall'], symbolic_row['precision']
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

def load_hybrid_data(dataset, sparsity):
    """Load Hybrid baseline data (multiple threshold points)."""
    dir_name = f"hybrid_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Hybrid path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all hybrid rows
        hybrid_rows = df[df['baseline'] == 'hybrid']
        recalls = hybrid_rows['recall'].tolist()
        precisions = hybrid_rows['precision'].tolist()
        return recalls, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

# --- Data Dictionary ---
# Build data dictionary from CSV files, keyed by dataset
data = {}
for dataset in datasets:
    data[dataset] = {
        "conrad": [],
        "neural": [],
        "symbolic": [],
        "hybrid": []
    }
    
    for sparsity in sparsity_values:
        # Load Conrad data
        c_r, c_p = load_conrad_data(dataset, sparsity)
        data[dataset]["conrad"].append((c_r, c_p))
        print(f"Loaded conrad {dataset} 3p {sparsity}%: {len(c_r) if c_r else 0} points")
        
        # Load Neural data
        n_r, n_p = load_neural_data(dataset, sparsity)
        data[dataset]["neural"].append((n_r, n_p))
        print(f"Loaded neural {dataset} 3p {sparsity}%: {len(n_r) if n_r else 0} points")
        
        # Load Symbolic data
        s_r, s_p = load_symbolic_data(dataset, sparsity)
        data[dataset]["symbolic"].append((s_r, s_p))
        print(f"Loaded symbolic {dataset} 3p {sparsity}%: {'yes' if s_r is not None else 'no'}")
        
        # Load Hybrid data
        h_r, h_p = load_hybrid_data(dataset, sparsity)
        data[dataset]["hybrid"].append((h_r, h_p))
        print(f"Loaded hybrid {dataset} 3p {sparsity}%: {len(h_r) if h_r else 0} points")

# --- Plotting ---
fig, axes = plt.subplots(3, 3, figsize=(8, 5), sharex=True, sharey=True)

for row, dataset in enumerate(datasets):
    for col, sparsity in enumerate(sparsities):
        ax = axes[row, col]
        
        # Plot Symbolic
        s_data = data[dataset]["symbolic"][col]
        if s_data[0] is not None and s_data[1] is not None:
            s_r, s_p = s_data
            ax.scatter(s_r, s_p, marker='X', color='C2', s=120, edgecolors='black', label='symbolic', zorder=5)

        # Plot Neural Points (different markers for each threshold)
        n_data = data[dataset]["neural"][col]
        if n_data[0] is not None and n_data[1] is not None and len(n_data[0]) > 0 and len(n_data[1]) > 0:
            n_r, n_p = n_data[0], n_data[1]
            for i, (r, p) in enumerate(zip(n_r, n_p)):
                label = f'neural (t={neural_thresholds[i]})' if row == 0 and col == 0 else None
                ax.scatter(r, p, marker=neural_markers[i], color='C1', s=50, 
                          edgecolors='black', linewidths=1, label=label, zorder=7)
            # Draw dashed line connecting neural points
            ax.plot(n_r, n_p, linestyle='--', color='C1', linewidth=2, alpha=0.5, zorder=6)

        # Plot Hybrid Points (different markers for each threshold)
        h_data = data[dataset]["hybrid"][col]
        if h_data[0] is not None and h_data[1] is not None:
            h_r, h_p = h_data[0], h_data[1]
            for i, (r, p) in enumerate(zip(h_r, h_p)):
                marker_idx = min(i, len(hybrid_markers) - 1)  # Handle cases with fewer points
                label = f'hybrid (t={hybrid_thresholds[marker_idx]})' if row == 0 and col == 0 and i < len(hybrid_thresholds) else None
                ax.scatter(r, p, marker=hybrid_markers[marker_idx], color='C3', s=60, 
                          edgecolors='black', linewidths=1, label=label, zorder=4)
            # Draw dashed line connecting hybrid points
            ax.plot(h_r, h_p, linestyle='--', color='C3', linewidth=2, alpha=0.5, zorder=3)

        # Plot Conrad Points (different markers for each alpha)
        c_data = data[dataset]["conrad"][col]
        if c_data[0] is not None and c_data[1] is not None:
            c_r, c_p = c_data[0], c_data[1]
            for i, (r, p) in enumerate(zip(c_r, c_p)):
                marker_idx = min(i, len(conrad_markers) - 1)
                label = f'conrad (α={alphas[i]})' if row == 0 and col == 0 and i < len(alphas) else None
                ax.scatter(r, p, marker=conrad_markers[marker_idx], color='C0', s=60,
                          edgecolors='black', linewidths=1, label=label, zorder=6)
            # Draw line connecting conrad points
            ax.plot(c_r, c_p, linestyle='-', color='C0', linewidth=2, zorder=5)

        # Axis limits and Titles
        ax.set_xlim(0, 1.2)
        ax.set_ylim(0, 1.2)
            
        if row == 0: ax.set_title(sparsity, fontsize=11)
        if col == 0: ax.set_ylabel(f"{dataset_labels[row]}\nPrecision", fontsize=11)
        if row == 2: ax.set_xlabel("Empirical Recall", fontsize=11)
        
        ax.grid(True)
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()

plt.subplots_adjust(wspace=0.2, hspace=0.25)
# Create figure-level legend positioned at the top center
# Group legend: symbolic, conrad, hybrid thresholds, neural thresholds
fig.legend(handles, labels, loc='upper center', ncol=5, fontsize=7.5, bbox_to_anchor=(0.45, 1.06))
plt.savefig('baselines_precision.png', dpi=300, bbox_inches='tight')
print("Plot saved to baselines_precision.png")

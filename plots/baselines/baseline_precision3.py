import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import numpy as np

# --- Configuration ---
alphas = [0.6, 0.7, 0.8, 0.9]
sparsities = ["5% sparsity", "20% sparsity", "40% sparsity"]
sparsity_values = [5, 20, 40]
datasets = ["fb15k-237", "nell-955", "yago310"]
dataset_labels = ["FB15K-237", "NELL-955", "YAGO3-10"]
query_pipelines = {
    "3p": "ThreeHopPipeline",
    "2u": "TwoUnionPipeline",
    "3ip": "TwoIntersectProjectPipeline"
}
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
hybrid_thresholds = [0.5, 0.6, 0.7]

# Colors matching the reference image
colors = {
    'symbolic': '#2ca02c',  # Green
    'neural': '#ff7f0e',    # Orange
    'hybrid': '#d62728',    # Red
    'conrad': '#1f77b4'     # Blue
}

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/plots_results")

def load_conrad_data(dataset, sparsity, query_pipeline):
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

def load_neural_data(dataset, sparsity, query_pipeline):
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

def load_symbolic_data(dataset, sparsity, query_pipeline):
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

def load_hybrid_data(dataset, sparsity, query_pipeline):
    """Load Hybrid baseline data (multiple threshold points)."""
    dir_name = f"hybrid_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Hybrid path not found: {csv_path}")
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Get all hybrid rows, excluding threshold 0.45
        hybrid_rows = df[(df['baseline'] == 'hybrid') & (df['threshold'] != 0.45)].sort_values('threshold')
        recalls = hybrid_rows['recall'].tolist()
        precisions = hybrid_rows['precision'].tolist()
        return recalls, precisions
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None

# --- Data Dictionary ---
# Build data dictionary from CSV files, keyed by (dataset, query_template)
data = {}
for dataset in datasets:
    for query_key, query_pipeline in query_pipelines.items():
        key = (dataset, query_key)
        data[key] = {
            "conrad": [],
            "neural": [],
            "symbolic": [],
            "hybrid": []
        }
        
        for sparsity in sparsity_values:
            # Load Conrad data
            c_r, c_p = load_conrad_data(dataset, sparsity, query_pipeline)
            data[key]["conrad"].append((c_r, c_p))
            print(f"Loaded conrad {dataset} {query_key} {sparsity}%: {len(c_r) if c_r else 0} points")
            
            # Load Neural data
            n_r, n_p = load_neural_data(dataset, sparsity, query_pipeline)
            data[key]["neural"].append((n_r, n_p))
            print(f"Loaded neural {dataset} {query_key} {sparsity}%: {len(n_r) if n_r else 0} points")
            
            # Load Symbolic data
            s_r, s_p = load_symbolic_data(dataset, sparsity, query_pipeline)
            data[key]["symbolic"].append((s_r, s_p))
            print(f"Loaded symbolic {dataset} {query_key} {sparsity}%: {'yes' if s_r is not None else 'no'}")
            
            # Load Hybrid data
            h_r, h_p = load_hybrid_data(dataset, sparsity, query_pipeline)
            data[key]["hybrid"].append((h_r, h_p))
            print(f"Loaded hybrid {dataset} {query_key} {sparsity}%: {len(h_r) if h_r else 0} points")

# --- Plotting ---
# Show 20% sparsity (index 1) in 3x3 format (3 rows, 3 columns)
# Rows: query types (3p, 2u, 3ip)
# Columns: datasets (fb15k-237, nell-955, yago310)
sparsity_idx = 1  # Index for 20% Missing Data
fig, axes = plt.subplots(3, 3, figsize=(8, 6), sharex=True, sharey=True)
fig.patch.set_facecolor('white')

query_types = ["3p", "2u", "3ip"]
query_type_labels = ["3p", "2u", "3ip"]

# Helper function to plot data for a single dataset+query combination
def plot_data_on_ax(ax, dataset, query_key, col, row):
    """Plot all baseline data for a given dataset and query type."""
    key = (dataset, query_key)
    
    # Plot Symbolic (single point with green X marker)
    s_data = data[key]["symbolic"][sparsity_idx]
    if s_data[0] is not None and s_data[1] is not None:
        s_r, s_p = s_data
        label = 'neo4j' if (row == 0 and col == 0) else None
        ax.scatter(s_r, s_p, marker='X', color=colors['symbolic'], s=150, 
                  edgecolors='black', linewidths=1.5, label=label, zorder=10)

    # Plot Neural Points (orange circles with dashed line and shaded area)
    n_data = data[key]["neural"][sparsity_idx]
    if n_data[0] is not None and n_data[1] is not None and len(n_data[0]) > 0 and len(n_data[1]) > 0:
        n_r, n_p = n_data[0], n_data[1]
        # Sort by recall for proper line drawing
        sorted_pairs = sorted(zip(n_r, n_p))
        n_r_sorted, n_p_sorted = zip(*sorted_pairs)
        n_r_sorted, n_p_sorted = list(n_r_sorted), list(n_p_sorted)
        
        # Draw shaded area (simulated confidence interval)
        ax.fill_between(n_r_sorted, 
                       [max(0, p - 0.02) for p in n_p_sorted],
                       [min(1, p + 0.02) for p in n_p_sorted],
                       color=colors['neural'], alpha=0.2, zorder=2, label='_nolegend_')
        
        # Draw dashed line connecting neural points
        label = 'neural (varying t)' if (row == 0 and col == 0) else None
        ax.plot(n_r_sorted, n_p_sorted, linestyle='--', color=colors['neural'], 
               linewidth=2, alpha=0.8, label=label, zorder=3)
        
        # Plot filled circle markers
        for r, p in zip(n_r_sorted, n_p_sorted):
            ax.scatter(r, p, marker='o', color=colors['neural'], s=60,
                      edgecolors='black', linewidths=1, zorder=4)

    # Plot Hybrid Points (red squares with dashed line and shaded area)
    h_data = data[key]["hybrid"][sparsity_idx]
    if h_data[0] is not None and h_data[1] is not None and len(h_data[0]) > 0 and len(h_data[1]) > 0:
        h_r, h_p = h_data[0], h_data[1]
        # Sort by recall for proper line drawing
        sorted_pairs = sorted(zip(h_r, h_p))
        h_r_sorted, h_p_sorted = zip(*sorted_pairs)
        h_r_sorted, h_p_sorted = list(h_r_sorted), list(h_p_sorted)
        
        # Draw shaded area
        ax.fill_between(h_r_sorted,
                       [max(0, p - 0.02) for p in h_p_sorted],
                       [min(1, p + 0.02) for p in h_p_sorted],
                       color=colors['hybrid'], alpha=0.2, zorder=2, label='_nolegend_')
        
        # Draw dashed line connecting hybrid points
        label = 'hybrid (varying t)' if (row == 0 and col == 0) else None
        ax.plot(h_r_sorted, h_p_sorted, linestyle='--', color=colors['hybrid'],
               linewidth=2, alpha=0.8, label=label, zorder=3)
        
        # Plot filled square markers
        for r, p in zip(h_r_sorted, h_p_sorted):
            ax.scatter(r, p, marker='s', color=colors['hybrid'], s=60,
                      edgecolors='black', linewidths=1, zorder=4)

    # Plot ConRAD Points (blue diamonds with dashed line, shaded area, and alpha labels)
    c_data = data[key]["conrad"][sparsity_idx]
    if c_data[0] is not None and c_data[1] is not None and len(c_data[0]) > 0 and len(c_data[1]) > 0:
        c_r, c_p = c_data[0], c_data[1]
        # Sort by recall for proper line drawing
        sorted_pairs = sorted(zip(c_r, c_p))
        c_r_sorted, c_p_sorted = zip(*sorted_pairs)
        c_r_sorted, c_p_sorted = list(c_r_sorted), list(c_p_sorted)
        
        # Draw shaded area
        ax.fill_between(c_r_sorted,
                       [max(0, p - 0.02) for p in c_p_sorted],
                       [min(1, p + 0.02) for p in c_p_sorted],
                       color=colors['conrad'], alpha=0.2, zorder=2, label='_nolegend_')
        
        # Draw dashed line connecting conrad points
        label = 'conrad (varying α)' if (row == 0 and col == 0) else None
        ax.plot(c_r_sorted, c_p_sorted, linestyle='--', color=colors['conrad'],
               linewidth=2, alpha=0.8, label=label, zorder=3)
        
        # Plot filled diamond markers with alpha annotations
        for i, (r, p) in enumerate(zip(c_r_sorted, c_p_sorted)):
            ax.scatter(r, p, marker='D', color=colors['conrad'], s=60,
                      edgecolors='black', linewidths=1, zorder=4)

# Plot all combinations
handles, labels = None, None
for row, query_key in enumerate(query_types):
    for col, dataset in enumerate(datasets):
        ax = axes[row, col]
        ax.set_facecolor('white')
        
        # Set up grid
        ax.grid(True, axis='x', alpha=0.3, zorder=1)
        ax.grid(True, axis='y', alpha=0.3, zorder=1)
        
        # Plot all data types
        plot_data_on_ax(ax, dataset, query_key, col, row)
        
        # Axis limits and formatting
        ax.set_xlim(0, 1.1)
        ax.set_ylim(0, 1.1)
        ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        # Format tick labels
        ax.set_xticklabels(['0.00', '0.25', '0.50', '0.75', '1.00'])
        ax.set_yticklabels(['0.0', '0.2', '0.4', '0.6', '0.8', '1.0'])
        
        # Titles and labels
        if row == 0:
            # Top row: dataset names
            ax.set_title(dataset_labels[col], fontsize=11, pad=10)
        if col == 0:
            # Left column: query type labels
            ax.set_ylabel(f"{query_type_labels[row]}\nPrecision", fontsize=11)
        if row == 2:
            # Bottom row: x-axis labels
            ax.set_xlabel("Empirical Recall", fontsize=11)
        
        # Collect legend handles and labels from first subplot
        if row == 0 and col == 0:
            handles, labels = ax.get_legend_handles_labels()
            # Filter out the _nolegend_ entries
            handles_labels = [(h, l) for h, l in zip(handles, labels) if l != '_nolegend_']
            handles, labels = zip(*handles_labels) if handles_labels else ([], [])

plt.subplots_adjust(wspace=0.3, hspace=0.25)
# Create figure-level legend positioned at the top center
fig.legend(handles, labels, loc='upper left', ncol=4, fontsize=9, 
          bbox_to_anchor=(0.1, 0.98))
plt.savefig('baseline_precision3.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Plot saved to baseline_precision3.png")

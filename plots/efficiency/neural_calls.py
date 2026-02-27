import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap
from matplotlib.ticker import FixedLocator
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Paths to the result directories
base_dir = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized")

# Incompleteness levels
incompleteness_levels = [5, 20]

# Dataset configuration
dataset_name = 'fb15k-237'
dataset_label = 'FB15k-237'

# Load data for each incompleteness level
data = {}
calibrated_lambdas = {}
for inc_level in incompleteness_levels:
    conrad_path = base_dir / f"conrad_bench_{dataset_name}_ThreeHopPipeline_{inc_level}"
    hybrid_path = base_dir / f"hybrid_bench_{dataset_name}_ThreeHopPipeline_{inc_level}"
    
    conrad_file = conrad_path / "results_summary.csv"
    hybrid_file = hybrid_path / "baseline_results_summary.csv"
    lambdas_file = conrad_path / "calibrated_lambdas.json"
    
    # Check if files exist before loading
    if conrad_file.exists() and hybrid_file.exists():
        # Load conrad data
        conrad_df = pd.read_csv(conrad_file)
        conrad_df['neural_call_pct'] = (conrad_df['avg_ultra_calls'] / 
                                         (conrad_df['avg_ultra_calls'] + conrad_df['avg_neo4j_calls']) * 100)
        conrad_df['neural_call_pct'] = conrad_df['neural_call_pct'].fillna(0)
        
        # Load hybrid data
        hybrid_df = pd.read_csv(hybrid_file)
        hybrid_df['neural_call_pct'] = 50.0
        
        data[inc_level] = {
            'conrad': conrad_df,
            'hybrid': hybrid_df
        }
        
        # Load calibrated lambdas if available
        if lambdas_file.exists():
            with open(lambdas_file, 'r') as f:
                calibrated_lambdas[inc_level] = json.load(f)
        else:
            # Try alternative location
            alt_lambdas_file = Path(f"/data/sonia/conrad/artifacts/plots_results/conrad_bench_{dataset_name}_ThreeHopPipeline_{inc_level}/calibrated_lambdas.json")
            if alt_lambdas_file.exists():
                with open(alt_lambdas_file, 'r') as f:
                    calibrated_lambdas[inc_level] = json.load(f)
            else:
                calibrated_lambdas[inc_level] = None
    else:
        # Mark as missing
        data[inc_level] = None
        calibrated_lambdas[inc_level] = None

# Build combined table and print CSV to stdout
csv_rows = []
for inc_level in incompleteness_levels:
    if data[inc_level] is None:
        continue
    conrad_df = data[inc_level]['conrad'].copy()
    hybrid_df = data[inc_level]['hybrid'].copy()
    conrad_df['dataset'] = dataset_label
    conrad_df['incompleteness_pct'] = inc_level
    conrad_df['method'] = 'conrad'
    hybrid_df['dataset'] = dataset_label
    hybrid_df['incompleteness_pct'] = inc_level
    hybrid_df['method'] = 'hybrid'
    # Select common columns for CSV (include key plot columns)
    conrad_cols = ['dataset', 'incompleteness_pct', 'method', 'recall', 'confidence', 'neural_call_pct']
    hybrid_cols = ['dataset', 'incompleteness_pct', 'method', 'recall', 'neural_call_pct']
    for c in ['avg_ultra_calls', 'avg_neo4j_calls']:
        if c in conrad_df.columns:
            conrad_cols.append(c)
    conrad_out = conrad_df[[c for c in conrad_cols if c in conrad_df.columns]]
    hybrid_out = hybrid_df[[c for c in hybrid_cols if c in hybrid_df.columns]]
    csv_rows.append(conrad_out)
    csv_rows.append(hybrid_out)
if csv_rows:
    csv_df = pd.concat(csv_rows, ignore_index=True, sort=False)
    # Ensure column order: identity cols first, then metrics
    col_order = [c for c in ['dataset', 'incompleteness_pct', 'method', 'recall', 'confidence',
                             'neural_call_pct', 'avg_ultra_calls', 'avg_neo4j_calls'] if c in csv_df.columns]
    csv_df = csv_df[col_order]
    csv_df.to_csv(sys.stdout, index=False)
    print(file=sys.stderr)  # newline after CSV on stderr so stdout is valid CSV
else:
    print("dataset,incompleteness_pct,method,recall,confidence,neural_call_pct", file=sys.stdout)

fontsz = 11
# --- VLDB Style Configuration ---
plt.rcParams.update({
    'font.size': fontsz,
    'axes.labelsize': 12,
    'axes.titlesize': fontsz,
    'xtick.labelsize': fontsz,
    'ytick.labelsize': fontsz,
    'legend.fontsize': fontsz,
    'axes.linewidth': 1.2,
    'lines.linewidth': 2.5
})

# Create 2x2 grid (2 rows, 2 columns: top row for line plots, bottom row for heatmaps)
fig = plt.figure(figsize=(6, 3))
gs = fig.add_gridspec(2, 2, height_ratios=[2, 2])
# Create subplots (don't use sharex since bottom uses categorical positions)
axes = [fig.add_subplot(gs[0, i]) for i in range(2)]
heatmap_axes = [fig.add_subplot(gs[1, i]) for i in range(2)]

for idx, inc_level in enumerate(incompleteness_levels):
    ax = axes[idx]
    
    # Check if data exists for this incompleteness level
    if data[inc_level] is None:
        # Show empty plot with message
        ax.text(0.5, 0.5, 'Data not available', 
               ha='center', va='center', transform=ax.transAxes, fontsize=fontsz)
        ax.set_title(f'{dataset_label}, {inc_level}%', fontsize=fontsz)
        ax.grid(True, alpha=0.6, zorder=0)
        # Don't set xlabel on top plots (they share x-axis)
        if idx == 0:
            ax.set_ylabel('Neural Invocations (%)')
        continue
    
    conrad_df = data[inc_level]['conrad']
    hybrid_df = data[inc_level]['hybrid']
    
    # Plot hybrid baseline first (so it's behind ConRAD)
    # Plot as a solid line across the domain to emphasize it's a static baseline
    min_recall = min(conrad_df['recall'].min(), hybrid_df['recall'].min())
    max_recall = max(conrad_df['recall'].max(), hybrid_df['recall'].max())
    
    # Only add label to first panel to avoid duplicate legend entries
    label = 'hybrid' if idx == 0 else None
    ax.axhline(y=50.0, color='#ca4641', linestyle='--', linewidth=2.5, zorder=2, label=label)
    
    # Plot baseline points on the line
    for _, hybrid_row in hybrid_df.iterrows():
        ax.plot(hybrid_row['recall'], hybrid_row['neural_call_pct'], 
                marker='X', markersize=10, color='#ca4641', markeredgecolor='white',
                markeredgewidth=1.0, zorder=4)
    
    # Plot ConRAD curve
    label = 'conrad' if idx == 0 else None
    ax.plot(conrad_df['recall'], conrad_df['neural_call_pct'], 
            marker='o', markersize=9, label=label, color='#2171b5', zorder=5)
    
    # Formatting to match paper subplots
    if idx == 0:
        ax.set_ylabel('Neural\nInvocations (%)')
    
    # Set title with dataset and incompleteness level
    title = f'{inc_level}% sparsity'
    ax.set_title(title, fontsize=fontsz+2)
    
    # Set x-axis ticks to match bottom plots (same step of 0.1)
    # Get confidence levels to match the bottom heatmap ticks
    if data[inc_level] is not None:
        confidence_levels = sorted(conrad_df['confidence'].unique())
        # Set ticks at confidence levels (0.6, 0.7, 0.8, 0.9) with step 0.1
        ax.set_xticks(confidence_levels)
        ax.set_xticklabels([f'{c:.1f}' for c in confidence_levels])
        # Ensure x-axis range accommodates these ticks
        ax.set_xlim(left=min(confidence_levels) - 0.05, right=max(confidence_levels) + 0.05)
    
    # Clean grid
    ax.grid(True, alpha=0.6, zorder=0)
    ax.set_ylim(-5, 60)
    
    # Stylish legend (only on first panel)
    if idx == 0:
        ax.legend(loc='lower right', fontsize=fontsz-2)
    
    # Create heatmap below each plot
    heatmap_ax = heatmap_axes[idx]
    
    if calibrated_lambdas[inc_level] is not None and data[inc_level] is not None:
        # Get confidence levels from results
        confidence_levels = sorted(conrad_df['confidence'].unique())
        
        # Map confidence to alpha (alpha = 1 - confidence)
        # Create heatmap data: rows = hops (1, 2, 3), cols = recall targets (confidence levels)
        heatmap_data = []
        hop_labels = ['Hop 1', 'Hop 2', 'Hop 3']
        recall_labels = []
        
        for conf in confidence_levels:
            alpha = str(1 - conf)
            # Handle floating point precision issues
            alpha_keys = [k for k in calibrated_lambdas[inc_level].keys() if abs(float(k) - (1 - conf)) < 0.01]
            
            if alpha_keys:
                alpha_key = alpha_keys[0]
                thresholds = calibrated_lambdas[inc_level][alpha_key]
                # Convert thresholds to binary: 0 = neural (<=0.5), 1 = retrieval (>0.5)
                hop_decisions = [1 if t > 0.5 else 0 for t in thresholds]
                heatmap_data.append(hop_decisions)
                recall_labels.append(f'{conf:.1f}')
        
        if heatmap_data:
            # Transpose so rows are hops and columns are recall targets
            heatmap_array = np.array(heatmap_data).T
            
            # Create custom colormap with the specified colors
            # 0 = Neural (#c7e9c0 - light green), 1 = Retrieval (#fcae91 - light orange)
            colors = ['#c7e9c0', '#fcae91']
            cmap = ListedColormap(colors)
            
            # Map categorical positions to actual recall values for alignment with top plots
            recall_values = [float(label) for label in recall_labels]
            if recall_values:
                # Use extent to map heatmap positions to actual recall values
                x_min = min(recall_values) - 0.05
                x_max = max(recall_values) + 0.05
                # Create heatmap with extent mapping categorical positions to recall values
                im = heatmap_ax.imshow(heatmap_array, cmap=cmap, aspect='auto', vmin=0, vmax=1,
                                      extent=[x_min, x_max, -0.5, len(hop_labels) - 0.5])
                
                # Set ticks at actual recall values to match top plots
                heatmap_ax.set_xticks(recall_values)
                heatmap_ax.set_xticklabels(recall_labels)
                heatmap_ax.set_xlim(x_min, x_max)
            else:
                # Fallback if no recall values
                im = heatmap_ax.imshow(heatmap_array, cmap=cmap, aspect='auto', vmin=0, vmax=1)
                heatmap_ax.set_xticks(np.arange(len(recall_labels)))
                heatmap_ax.set_xticklabels(recall_labels)
            
            heatmap_ax.set_yticks(np.arange(len(hop_labels)))
            heatmap_ax.set_yticklabels(hop_labels)
            # Grid at cell boundaries: between confidence targets (x) and between hops (y)
            n_rows, n_cols = heatmap_array.shape
            y_boundaries = np.linspace(-0.5, n_rows - 0.5, n_rows + 1)
            heatmap_ax.yaxis.set_minor_locator(FixedLocator(y_boundaries))
            if recall_values:
                x_boundaries = np.linspace(x_min, x_max, n_cols + 1)
                heatmap_ax.xaxis.set_minor_locator(FixedLocator(x_boundaries))
            else:
                x_boundaries = np.linspace(-0.5, n_cols - 0.5, n_cols + 1)
                heatmap_ax.xaxis.set_minor_locator(FixedLocator(x_boundaries))
            heatmap_ax.grid(True, which='minor', alpha=0.6, color='gray', linewidth=0.5)
            heatmap_ax.tick_params(which='minor', size=0)  # no tick marks at boundaries
            
            # Set xlabel only on bottom plots
            if idx == 0:
                heatmap_ax.set_xlabel('Recall', fontsize=fontsz)
                heatmap_ax.set_ylabel('Hop', fontsize=fontsz)
            else:
                heatmap_ax.set_xlabel('Recall', fontsize=fontsz)
            
            # Add legend (only on first heatmap to avoid duplication)
            if idx == 1:
                from matplotlib.patches import Patch
                legend_elements = [
                    Patch(facecolor='#c7e9c0', label='Hybrid'),
                    Patch(facecolor='#fcae91', label='Retrieval')
                ]
                heatmap_ax.legend(handles=legend_elements, loc='lower right', fontsize=fontsz-2)
            # heatmap_ax.set_title(f'Neural (N) vs Retrieval (R)', fontsize=10)
        else:
            heatmap_ax.text(0.5, 0.5, 'No threshold data', 
                           ha='center', va='center', transform=heatmap_ax.transAxes, fontsize=fontsz)
            # heatmap_ax.set_title(f'Neural vs Retrieval Decision', fontsize=10)
    else:
        heatmap_ax.text(0.5, 0.5, 'Data not available', 
                       ha='center', va='center', transform=heatmap_ax.transAxes, fontsize=fontsz)
        # heatmap_ax.set_title(f'Neural vs Retrieval Decision', fontsize=10)

plt.tight_layout()

# Save the plot
output_dir = Path("/data/sonia/conrad/plots")
output_dir.mkdir(exist_ok=True, parents=True)
plt.savefig(output_dir / 'neural_calls.pdf', dpi=300, bbox_inches='tight')
plt.savefig(output_dir / 'neural_calls.png', dpi=300, bbox_inches='tight')
print(f"\nPlot saved to {output_dir / 'neural_calls.png'}", file=sys.stderr)
print(f"Plot saved to {output_dir / 'neural_calls.pdf'}", file=sys.stderr)
plt.show()
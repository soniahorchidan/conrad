import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os
from pathlib import Path
from scipy import stats

# Configuration for the 2x3 plot (FB15k-237 across sparsity levels)
# Rows: sparsity levels (5%, 40%)
# Columns: query templates (3p, 2u, 2ip)
datasets = [("FB15k-237", "fb15k-237")]
sparsities = ["5% sparsity", "40% sparsity"]
sparsity_values = [5, 40]
query_types = ["ThreeHop", "TwoUnion", "TwoIntersectProject"]
query_labels = ["3p", "2u", "2ip"]
query_colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
query_markers = ['o', 's', '^']
confidence_levels = [0.6, 0.7, 0.8, 0.9]

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/final_results/benchmark_latency_objective_normalized/")

# Confidence level for CI (e.g., 0.95 for 95% CI)
ci_level = 0.95
z_score = stats.norm.ppf((1 + ci_level) / 2)  # z-score for 95% CI (≈1.96)

def compute_confidence_band(x_values, num_queries, z_score=z_score):
    """
    Compute confidence band around the diagonal (ideal line) for mean recall estimate.
    
    The CRC guarantee is on the expected risk. The empirical mean recall on a finite
    validation set is an estimate of that expectation, with sampling variance.
    
    Standard error of mean recall: SE = σ/√n, where σ is the standard deviation of
    per-query recall. Using normal approximation for proportions:
    SE ≈ sqrt(p(1-p)/n) where p is the target recall.
    
    Confidence band: target ± z_{0.975} · SE
    
    Args:
        x_values: Array of target recall values (x-axis, same as y-axis for ideal line)
        num_queries: Number of inference queries (sample size)
        z_score: Z-score for desired confidence level (default: 1.96 for 95% CI)
    
    Returns:
        Tuple of (lower_band, upper_band) arrays
    """
    if num_queries <= 0:
        return None, None
    
    x_array = np.array(x_values)
    
    # Standard error for proportion: SE = sqrt(p(1-p)/n)
    # This accounts for sampling variance of the mean recall estimate
    se = np.sqrt(x_array * (1 - x_array) / num_queries)
    
    # Confidence band around diagonal
    lower_band = np.maximum(0, x_array - z_score * se)
    upper_band = np.minimum(1, x_array + z_score * se)
    
    return lower_band, upper_band

def load_data(dataset, query_type, sparsity):
    """Load recall data from CSV file."""
    dir_name = f"conrad_bench_{dataset}_{query_type}Pipeline_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        return None, None, None
    
    try:
        df = pd.read_csv(csv_path)
        # Filter to only the desired confidence levels
        df = df[df['confidence'].isin(confidence_levels)]
        # Extract confidence (target recall), recall (empirical recall), and num_queries
        target_recall = df['confidence'].tolist()
        empirical_recall = df['recall'].tolist()
        num_queries = df['num_queries'].tolist()
        return target_recall, empirical_recall, num_queries
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None, None, None

# --- Plotting Setup ---
# 2x3 grid: rows = sparsity levels, columns = query templates
fig, axes = plt.subplots(2, 3, figsize=(5, 3), sharex=True, sharey=True)

# Track legend handles/labels (only add once)
legend_added = False

# Collect all (target, empirical) with metadata for deviation analysis
all_points = []  # list of (target_recall, empirical_recall, sparsity_label, query_label)

for r, sparsity_val in enumerate(sparsity_values):  # Rows: sparsity levels
    sparsity_label = sparsities[r]
    
    for c, query_type in enumerate(query_types):  # Columns: query templates
        ax = axes[r, c]
        dataset_key = datasets[0][1]  # FB15k-237
        query_label = query_labels[c]
        query_color = query_colors[c]
        query_marker = query_markers[c]
        
        # Load data for this query type and sparsity level
        target_recall, empirical_recall, num_queries = load_data(
            dataset_key, query_type, sparsity_val
        )
        
        # Draw confidence band around ideal line (diagonal)
        if num_queries is not None and len(num_queries) > 0:
            num_queries_for_band = num_queries[0]  # Use first value
            
            # Create fine-grained x values for smooth band
            x_band = np.linspace(0.55, 0.95, 200)
            lower_band, upper_band = compute_confidence_band(x_band, num_queries_for_band, z_score)
            
            if lower_band is not None and upper_band is not None:
                # Fill the confidence band (generic label, use light grey)
                label_band = f'{int(ci_level*100)}% CI band' if not legend_added else None
                ax.fill_between(x_band, lower_band, upper_band, 
                              alpha=0.2, color='gray', zorder=0, label=label_band)
                if label_band:
                    legend_added = True
        
        # Draw Ideal Line (y=x) on top of the band
        x_line = np.linspace(0.55, 0.95, 100)
        ideal_label = 'Ideal' if r == 0 and c == 0 else None
        ax.plot(x_line, x_line, linestyle='--', color='red', label=ideal_label, 
                linewidth=2, zorder=2)
        
        # Plot the data for this query type (no label needed - title indicates template)
        if target_recall is not None and empirical_recall is not None:
            ax.plot(target_recall, empirical_recall, marker=query_marker, 
                    label=None, color=query_color, linewidth=2, markersize=5, zorder=3)
            for t, e in zip(target_recall, empirical_recall):
                all_points.append((t, e, sparsity_label, query_label))
        
        # Formatting
        ax.set_xlim(0.55, 0.95)
        ax.set_ylim(0.55, 1)
        ax.set_xticks(confidence_levels)
        ax.grid(True, alpha=0.6)
        
        # Title for each subplot showing sparsity only (query type in legend)
        ax.set_title(f"{sparsity_label}", fontsize=9)
        
        # Y-axis labels (first column only)
        if c == 0:
            ax.set_ylabel('Empirical Recall', fontsize=9)
        
        # X-axis labels (bottom row only)
        if r == 1:
            ax.set_xlabel('Target Recall', fontsize=9)

# --- Highest deviation from ideal (y = x) ---
# Upward: empirical > target; downward: empirical < target
upward = [(e - t, (t, e, sp_lbl, q_lbl)) for t, e, sp_lbl, q_lbl in all_points if e > t]
downward = [(t - e, (t, e, sp_lbl, q_lbl)) for t, e, sp_lbl, q_lbl in all_points if e < t]
max_up = max(upward, key=lambda x: x[0]) if upward else None
max_down = max(downward, key=lambda x: x[0]) if downward else None

print("\n--- Deviation from ideal (empirical recall vs target recall) ---")
if max_up is not None:
    dev, (t, e, sp_lbl, q_lbl) = max_up
    print(f"Highest upward deviation:   {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{sp_lbl}, {q_lbl}]")
else:
    print("Highest upward deviation:   (none; no point above the diagonal)")
if max_down is not None:
    dev, (t, e, sp_lbl, q_lbl) = max_down
    print(f"Highest downward deviation: {dev:.4f}  at target={t:.2f}, empirical={e:.4f}  [{sp_lbl}, {q_lbl}]")
else:
    print("Highest downward deviation: (none; no point below the diagonal)")
print()

# Figure-level legend: Ideal, CI band, then query types (3p, 2u, 2ip)
handles, labels = axes[0, 0].get_legend_handles_labels()
# Proxy artists for query types (line + marker, same as in plot)
for ql, color, marker in zip(query_labels, query_colors, query_markers):
    handles.append(Line2D([0], [0], color=color, marker=marker, linestyle='-',
                         linewidth=2, markersize=5, label=ql))
    labels.append(ql)
# Reorder so "Ideal" comes first
ideal_idx = next((i for i, label in enumerate(labels) if label == 'Ideal'), None)
if ideal_idx is not None and ideal_idx != 0:
    handles = [handles[ideal_idx]] + [h for i, h in enumerate(handles) if i != ideal_idx]
    labels = [labels[ideal_idx]] + [l for i, l in enumerate(labels) if i != ideal_idx]

plt.tight_layout()
# Create figure-level legend positioned at the top center
fig.legend(handles, labels, loc='upper left', ncol=len(handles), fontsize=9,
           bbox_to_anchor=(0.05, 1.05))

plt.savefig('conrad_validity_fb15k_sparsity_with_ci.pdf', dpi=300, bbox_inches='tight')
plt.savefig('conrad_validity_fb15k_sparsity_with_ci.png', dpi=300, bbox_inches='tight')
print("Plot saved to conrad_validity_fb15k_sparsity_with_ci.png")

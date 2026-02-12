import matplotlib.pyplot as plt
import pandas as pd
import re
from pathlib import Path
import numpy as np

# --- Configuration ---
# Using fb15k-237 ThreeHopPipeline 5% sparsity
# Conrad at alpha=0.7 has recall=0.7313
# Lambdas at alpha=0.3 are [0.543, 0.590, 0.640] - ALL > 0.5 (bypasses neural execution)
# Matches well with Hybrid threshold 0.60 (recall=0.7428) and Neural threshold 0.99 (recall=0.8287)
dataset = "fb15k-237"
query_pipeline = "ThreeHopPipeline"
sparsity = 5
conrad_alpha = 0.7  # Has recall ~0.7313, lambdas at alpha=0.3 all > 0.5

# Colors matching the reference image
colors = {
    'symbolic': '#2ca02c',  # Green
    'neural': '#ff7f0e',    # Orange
    'hybrid': '#d62728',    # Red
    'conrad': '#1f77b4'     # Blue
}

# Base path to benchmark results
# Using artifacts/benchmark for Conrad (latest results with correct latencies)
# Using artifacts/plots_results for baselines
conrad_base_path = Path("/data/sonia/conrad/artifacts/benchmark")
baseline_base_path = Path("/data/sonia/conrad/artifacts/plots_results")

def extract_avg_latency_from_log(log_path: Path) -> float:
    """Extract average latency from Conrad benchmark log."""
    if not log_path.exists():
        print(f"  Log path not found: {log_path}")
        return None
    
    times = []
    with open(log_path, 'r') as f:
        for line in f:
            match = re.search(r'Time: ([\d.]+)ms', line)
            if match:
                times.append(float(match.group(1)))
    
    if times:
        return sum(times) / len(times)
    return None

def load_conrad_latency(dataset, sparsity, query_pipeline, alpha):
    """Load Conrad average latency from benchmark log."""
    dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
    log_path = conrad_base_path / dir_name / f"benchmark_conf_{alpha}.log"
    
    avg_time = extract_avg_latency_from_log(log_path)
    return avg_time

def load_baseline_latency(dataset, sparsity, query_pipeline, baseline_type, threshold=None):
    """Load baseline average latency from CSV."""
    dir_name = f"{baseline_type}_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = baseline_base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"  Baseline path not found: {csv_path}")
        return None
    
    try:
        df = pd.read_csv(csv_path)
        if baseline_type == 'symbolic':
            # Symbolic has no threshold
            row = df[df['baseline'] == 'symbolic'].iloc[0]
            return row['avg_time_ms']
        else:
            # Neural or hybrid - filter by threshold
            if threshold is not None:
                row = df[(df['baseline'] == baseline_type) & (df['threshold'] == threshold)]
                if len(row) > 0:
                    return row.iloc[0]['avg_time_ms']
            return None
    except Exception as e:
        print(f"Error loading {csv_path}: {e}")
        return None

def load_conrad_recall(dataset, sparsity, query_pipeline, alpha):
    """Load Conrad recall from CSV."""
    dir_name = f"conrad_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = conrad_base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        return None
    
    try:
        df = pd.read_csv(csv_path)
        row = df[df['confidence'] == alpha]
        if len(row) > 0:
            return row.iloc[0]['recall']
        return None
    except Exception as e:
        print(f"Error loading recall from {csv_path}: {e}")
        return None

def find_matching_baseline_recall(dataset, sparsity, query_pipeline, target_recall, baseline_type, tolerance=0.1):
    """Find baseline threshold that matches target recall."""
    dir_name = f"{baseline_type}_bench_{dataset}_{query_pipeline}_{sparsity}"
    csv_path = baseline_base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        return None, None
    
    try:
        df = pd.read_csv(csv_path)
        if baseline_type == 'symbolic':
            row = df[df['baseline'] == 'symbolic'].iloc[0]
            recall = row['recall']
            if abs(recall - target_recall) <= tolerance:
                return None, row['avg_time_ms']  # No threshold for symbolic
            return None, None
        else:
            # Find closest recall match
            baseline_rows = df[df['baseline'] == baseline_type].copy()
            baseline_rows['recall_diff'] = abs(baseline_rows['recall'] - target_recall)
            closest = baseline_rows.loc[baseline_rows['recall_diff'].idxmin()]
            
            if closest['recall_diff'] <= tolerance:
                return closest['threshold'], closest['avg_time_ms']
            return None, None
    except Exception as e:
        print(f"Error finding matching baseline: {e}")
        return None, None

# Load Conrad data
conrad_latency = load_conrad_latency(dataset, sparsity, query_pipeline, conrad_alpha)
conrad_recall = load_conrad_recall(dataset, sparsity, query_pipeline, conrad_alpha)

print(f"Conrad (alpha={conrad_alpha}):")
print(f"  Recall: {conrad_recall:.4f}")
print(f"  Latency: {conrad_latency:.2f} ms")

# Find matching baselines
print(f"\nFinding baselines with recall ~{conrad_recall:.4f}:")

# Symbolic
sym_threshold, sym_latency = find_matching_baseline_recall(dataset, sparsity, query_pipeline, conrad_recall, 'symbolic')
if sym_latency:
    print(f"  Symbolic: latency={sym_latency:.2f} ms")

# Neural - find closest match
neural_threshold, neural_latency = find_matching_baseline_recall(dataset, sparsity, query_pipeline, conrad_recall, 'neural')
if neural_latency:
    print(f"  Neural (threshold={neural_threshold}): latency={neural_latency:.2f} ms")

# Hybrid - find closest match
hybrid_threshold, hybrid_latency = find_matching_baseline_recall(dataset, sparsity, query_pipeline, conrad_recall, 'hybrid')
if hybrid_latency:
    print(f"  Hybrid (threshold={hybrid_threshold}): latency={hybrid_latency:.2f} ms")

# --- Plotting ---
fig, ax = plt.subplots(figsize=(8, 5))
fig.patch.set_facecolor('white')
ax.set_facecolor('white')

# Prepare data for plotting
systems = []
latencies = []
bar_colors = []

if conrad_latency:
    systems.append('ConRAD\n(α=0.7)')
    latencies.append(conrad_latency)
    bar_colors.append(colors['conrad'])

if sym_latency:
    systems.append('neo4j\n(symbolic)')
    latencies.append(sym_latency)
    bar_colors.append(colors['symbolic'])

if neural_latency:
    systems.append(f'neural\n(t={neural_threshold})')
    latencies.append(neural_latency)
    bar_colors.append(colors['neural'])

if hybrid_latency:
    systems.append(f'hybrid\n(t={hybrid_threshold})')
    latencies.append(hybrid_latency)
    bar_colors.append(colors['hybrid'])

# Create bar plot
bars = ax.bar(systems, latencies, color=bar_colors, edgecolor='black', linewidth=1.5, alpha=0.8)

# Add value labels on bars
for bar, latency in zip(bars, latencies):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{latency:.1f} ms',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Formatting
ax.set_ylabel('Average Latency (ms)', fontsize=12)
ax.set_title(f'Latency Comparison: {dataset.upper()} ThreeHopPipeline ({sparsity}% sparsity)\nTarget Recall: {conrad_recall:.3f} (ConRAD α={conrad_alpha}, lambdas > 0.5)', 
             fontsize=13, pad=15)
ax.grid(True, axis='y', alpha=0.3, zorder=0)

# Set y-axis to start from 0
ax.set_ylim(bottom=0, top=max(latencies) * 1.2 if latencies else 50)

plt.tight_layout()
plt.savefig('NEW_baseline_latency.png', dpi=300, bbox_inches='tight', facecolor='white')
print(f"\nPlot saved to NEW_baseline_latency.png")

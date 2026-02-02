import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

# Base path to benchmark results
base_path = Path("/data/sonia/conrad/artifacts/benchmark")

# Configuration
query_type = "ThreeHopPipeline"  # For 3p queries
confidence_levels = [0.5, 0.6, 0.7, 0.8, 0.9]
neural_thresholds = [0.7, 0.8, 0.9, 0.99]
hybrid_thresholds = [0.45, 0.5, 0.6, 0.7]

# Setup
x_labels = ['0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(x_labels))
width = 0.6  # Width for the CONRAD bars

# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_conrad_abstention(dataset, query_type, sparsity):
    """Load Conrad abstention rates from CSV file."""
    dir_name = f"conrad_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "results_summary.csv"
    
    if not csv_path.exists():
        print(f"Conrad not found: {dir_name}")
        return [np.nan] * len(confidence_levels)
    
    try:
        df = pd.read_csv(csv_path)
        abstention_rates = df['abstention_rate'].tolist()
        # Pad with NaN if we have fewer than expected
        while len(abstention_rates) < len(confidence_levels):
            abstention_rates.append(np.nan)
        return abstention_rates[:len(confidence_levels)]
    except Exception as e:
        print(f"Error loading Conrad {dir_name}: {e}")
        return [np.nan] * len(confidence_levels)

def load_neural_abstention(dataset, query_type, sparsity):
    """Load Neural baseline abstention rates from CSV files."""
    dir_name = f"neural_bench_{dataset}_{query_type}_{sparsity}"
    
    neural_dict = {}
    for threshold in neural_thresholds:
        csv_path = base_path / dir_name / f"threshold_{threshold}" / "baseline_results_summary.csv"
        
        if not csv_path.exists():
            continue
        
        try:
            df = pd.read_csv(csv_path)
            neural_row = df[df['baseline'] == 'neural'].iloc[0]
            neural_dict[threshold] = neural_row['abstention_rate']
        except Exception as e:
            print(f"Error loading Neural {dir_name}/threshold_{threshold}: {e}")
            continue
    
    return neural_dict

def load_hybrid_abstention(dataset, query_type, sparsity):
    """Load Hybrid baseline abstention rates from CSV file."""
    dir_name = f"hybrid_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"Hybrid not found: {dir_name}")
        return {}
    
    try:
        df = pd.read_csv(csv_path)
        hybrid_rows = df[df['baseline'] == 'hybrid']
        hybrid_dict = {}
        for _, row in hybrid_rows.iterrows():
            threshold = row['threshold']
            if threshold in hybrid_thresholds:
                hybrid_dict[threshold] = row['abstention_rate']
        return hybrid_dict
    except Exception as e:
        print(f"Error loading Hybrid {dir_name}: {e}")
        return {}

def load_symbolic_abstention(dataset, query_type, sparsity):
    """Load Symbolic baseline abstention rate from CSV file."""
    dir_name = f"symbolic_bench_{dataset}_{query_type}_{sparsity}"
    csv_path = base_path / dir_name / "baseline_results_summary.csv"
    
    if not csv_path.exists():
        print(f"Symbolic not found: {dir_name}")
        return np.nan
    
    try:
        df = pd.read_csv(csv_path)
        symbolic_row = df[df['baseline'] == 'symbolic'].iloc[0]
        return symbolic_row['abstention_rate']
    except Exception as e:
        print(f"Error loading Symbolic {dir_name}: {e}")
        return np.nan

# ============================================================================
# DATA LOADING
# ============================================================================

datasets = {}
missing_levels = [5, 20, 40]
dataset_configs = [
    ('FB15k-237', 'fb15k-237'),
    ('NELL-995', 'nell-995')
]

for display_name, dataset_key in dataset_configs:
    datasets[display_name] = {}
    for m_level in missing_levels:
        print(f"\nLoading {display_name} {m_level}%...")
        datasets[display_name][m_level] = {
            'conrad': load_conrad_abstention(dataset_key, query_type, m_level),
            'symbolic': load_symbolic_abstention(dataset_key, query_type, m_level),
            'neural': load_neural_abstention(dataset_key, query_type, m_level),
            'hybrid': load_hybrid_abstention(dataset_key, query_type, m_level)
        }
        print(f"  Conrad: {datasets[display_name][m_level]['conrad']}")
        print(f"  Symbolic: {datasets[display_name][m_level]['symbolic']}")
        print(f"  Neural: {datasets[display_name][m_level]['neural']}")
        print(f"  Hybrid: {datasets[display_name][m_level]['hybrid']}")

dataset_names = [name for name, _ in dataset_configs]

# ============================================================================
# PLOTTING
# ============================================================================

fig, axes = plt.subplots(len(dataset_names), 3, figsize=(8, 4), sharex=True, sharey=True)

for row_idx, d_name in enumerate(dataset_names):
    for col_idx, m_level in enumerate(missing_levels):
        ax = axes[row_idx, col_idx]
        data = datasets[d_name][m_level]
        
        # 1. Plot CONRAD as bars
        conrad_vals = data['conrad']
        ax.bar(x, conrad_vals, width, label='conrad', color='lightgray', edgecolor='black', zorder=2)
        
        # 2. Plot Symbolic as a horizontal line
        if not np.isnan(data['symbolic']):
            ax.axhline(y=data['symbolic'], color='grey', linestyle='--', linewidth=1.5, label='symbolic')

        # 3. Plot Neural as horizontal lines
        neural_data = data['neural']
        neural_colors = ['#fee08b', '#fdae61', '#f46d43', '#d73027']
        for i, (thresh, val) in enumerate(neural_data.items()):
            ax.axhline(y=val, color=neural_colors[i], linestyle='-', linewidth=2, label=f'neural (t={thresh})')

        # 4. Plot Hybrid as horizontal lines
        hybrid_data = data['hybrid']
        hybrid_colors = ['#c7e9c0', '#a1d99b', '#74c476', '#41ab5d']  # Green shades
        for i, (thresh, val) in enumerate(hybrid_data.items()):
            color_idx = hybrid_thresholds.index(thresh) if thresh in hybrid_thresholds else 0
            ax.axhline(y=val, color=hybrid_colors[color_idx], linestyle=':', linewidth=2, label=f'hybrid (t={thresh})')

        # Formatting
        if row_idx == 0:
            ax.set_title(f'{m_level}% Missing Data', fontsize=10)
        if col_idx == 0:
            ax.set_ylabel(f'{d_name}\nAbstention Rate', fontsize=10)
        if row_idx == 1:
            ax.set_xlabel('Target Recall', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(True, alpha=0.6, zorder=0)
        
        # Adjust Y limits
        ax.set_ylim(0, 0.7)

# Create a custom legend with all entries from a subplot that has data
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1.06), ncols=6, fontsize=8)
plt.tight_layout()
plt.savefig('conrad_abstention_3p_from_csv.png', dpi=300, bbox_inches='tight')
print("\nPlot saved to conrad_abstention_3p_from_csv.png")
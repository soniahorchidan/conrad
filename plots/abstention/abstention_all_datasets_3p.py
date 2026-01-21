import matplotlib.pyplot as plt
import numpy as np

# Setup
x_labels = ['0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(x_labels))
width = 0.6  # Width for the CONRAD bars

# ============================================================================
# DATA STORAGE
# ============================================================================

datasets = {
    'FB15k-237': {
        5: {
            'conrad': [0.332, 0.248, 0.172, 0.089, 0.04],
            'symbolic': 0.069,
            'neural': {0.7: 0.024, 0.8: 0.027, 0.9: 0.032, 0.99: 0.086}
        },
        20: {
            'conrad': [0.25, 0.205, 0.113, 0.072, 0.041],
            'symbolic': 0.222,
            'neural': {0.7: 0.131, 0.8: 0.158, 0.9: 0.188, 0.99: 0.276}
        },
        40: {
            'conrad': [0.18, 0.14, 0.079, 0.033, np.nan], # 0.9 is missing/OOM
            'symbolic': 0.427,
            'neural': {0.7: 0.326, 0.8: 0.380, 0.9: 0.455, 0.99: 0.608}
        }
    },
    'NELL-995': {
        5: {
            'conrad': [0.437, 0.352, 0.185, 0.148, np.nan], # 0.9 OOM
            'symbolic': np.nan, # OOM
            'neural': {} # OOM
        },
        20: {
            'conrad': [0.395, 0.3279, 0.2318, 0.149, 0.072],
            'symbolic': 0.586,
            'neural': {0.7: 0.573, 0.8: 0.573, 0.9: 0.573, 0.99: 0.574}
        },
        40: {
            'conrad': [np.nan] * 5, # OOM
            'symbolic': 0.673,
            'neural': {0.7: 0.604, 0.8: 0.604, 0.9: 0.604, 0.99: 0.604}
        }
    }
}

missing_levels = [5, 20, 40]
dataset_names = ['FB15k-237', 'NELL-995'] # YAGO is still TODO

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
            # ax.text(4.2, data['symbolic'], 'symbolic', color='grey', fontweight='bold', va='center')

        # 3. Plot Neural as horizontal lines with labels for legend
        neural_data = data['neural']
        colors = ['#fee08b', '#fdae61', '#f46d43', '#d73027']
        for i, (thresh, val) in enumerate(neural_data.items()):
            ax.axhline(y=val, color=colors[i], linestyle='-', linewidth=2, label=f'neural (t={thresh})')

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
        if d_name == 'FB15k-237': ax.set_ylim(0, 0.7)
        else: ax.set_ylim(0, 0.7)

# Create a custom legend with all entries
handles, labels = axes[0, 1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0, 1.06), ncols=len(handles), fontsize=9)
plt.tight_layout()
plt.savefig('conrad_abstention_3p.png', dpi=300, bbox_inches='tight')
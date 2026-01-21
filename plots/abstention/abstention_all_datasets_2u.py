import matplotlib.pyplot as plt
import numpy as np

# Setup
x_labels = ['0.5', '0.6', '0.7', '0.8', '0.9']
x = np.arange(len(x_labels))
width = 0.6 

# ============================================================================
# DATA STORAGE (Extracted from 2u results)
# ============================================================================
datasets_2u = {
    'FB15k-237': {
        5: {
            'conrad': [0.22, 0.14, 0.072, 0.03, 0.006],
            'symbolic': 0.000, # abstention rate = 0 (1000 queries, 0 abstentions)
            'neural': {0.7: 0.001, 0.8: 0.001, 0.9: 0.001, 0.99: 0.003}
        },
        20: {
            'conrad': [0.177, 0.085, 0.034, 0.009, 0.001],
            'symbolic': 0.010, # 10/1000
            'neural': {0.7: 0.002, 0.8: 0.004, 0.9: 0.007, 0.99: 0.019}
        },
        40: {
            'conrad': [0.178, 0.113, 0.046, 0.014, 0.002],
            'symbolic': 0.120, # 120/1000
            'neural': {0.7: 0.029, 0.8: 0.030, 0.9: 0.032, 0.99: 0.037}
        }
    },
    'NELL-995': {
        5: {
            'conrad': [0.181, 0.099, 0.049, 0.021, 0.005],
            'symbolic': 0.003, # 3/1000
            'neural': {0.7: 0.001, 0.8: 0.001, 0.9: 0.001, 0.99: 0.001}
        },
        20: {
            'conrad': [0.188, 0.116, 0.061, 0.019, 0.007],
            'symbolic': 0.027, # 27/1000
            'neural': {0.7: 0.002, 0.8: 0.002, 0.9: 0.002, 0.99: 0.002}
        },
        40: {
            'conrad': [0.178, 0.113, 0.046, 0.014, 0.002],
            'symbolic': 0.120, # 120/1000
            'neural': {0.7: 0.029, 0.8: 0.030, 0.9: 0.032, 0.99: 0.037}
        }
    }
}

missing_levels = [5, 20, 40]
dataset_names = ['FB15k-237', 'NELL-995']

# ============================================================================
# PLOTTING
# ============================================================================

fig, axes = plt.subplots(len(dataset_names), 3, figsize=(10, 5), sharex=True, sharey=True)

for row_idx, d_name in enumerate(dataset_names):
    for col_idx, m_level in enumerate(missing_levels):
        ax = axes[row_idx, col_idx]
        data = datasets_2u[d_name][m_level]
        
        # 1. Plot CONRAD as bars
        conrad_vals = data['conrad']
        ax.bar(x, conrad_vals, width, label='ConRAD', color='#4393c3', edgecolor='black', zorder=3)
        
        # 2. Plot Symbolic as a horizontal line
        ax.axhline(y=data['symbolic'], color='black', linestyle='--', linewidth=1.5, label='symbolic', zorder=4)

        # 3. Plot Neural thresholds
        neural_data = data['neural']
        # Using a more distinct color palette for the baseline thresholds
        colors = ['#fee08b', '#fdae61', '#f46d43', '#d73027']
        for i, (thresh, val) in enumerate(neural_data.items()):
            ax.axhline(y=val, color=colors[i], linestyle='-', linewidth=2, label=f'neural (t={thresh})')

        # Aesthetics
        if row_idx == 0:
            ax.set_title(f'{m_level}% Missing Data', fontsize=11, fontweight='bold')
        if col_idx == 0:
            ax.set_ylabel(f'{d_name}\nAbstention Rate', fontsize=10, fontweight='bold')
        if row_idx == 1:
            ax.set_xlabel('Target Recall (1-$\\alpha$)', fontsize=10)
        
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.grid(axis='y', linestyle=':', alpha=0.7, zorder=0)
        ax.set_ylim(0, 0.3) # 2u abstention is generally lower than 3p due to union logic

# Legend configuration
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncols=len(handles), fontsize=9, frameon=False)

plt.tight_layout()
plt.savefig('conrad_abstention_2u.png', dpi=300, bbox_inches='tight')
plt.show()
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({'font.size': 10})

# --- OURS Data ---
target_recall = np.array([0.5, 0.6, 0.7, 0.8, 0.9])
empirical_recall = np.array([0.4939, 0.6483, 0.7615, 0.7615, 0.8961])
precision = np.array([0.5015, 0.5825, 0.6746, 0.7667, 0.919])
# f1 = np.array([0.3996, 0.4643, 0.5338, 0.5959, 0.599])
f1 = 2 * (precision * empirical_recall) / (precision + empirical_recall)

abstention_rate = np.array([0.447, 0.3609, 0.2653, 0.1744, 0.0473])
non_abs_precision = np.array([0.7051, 0.708, 0.7054, 0.6951, 0.5802])
non_abs_recall = np.array([0.9069, 0.9114, 0.9183, 0.9286, 0.9646])
non_abs_f1 = 2 * (non_abs_precision * non_abs_recall) / (non_abs_precision + non_abs_recall)
num_queries = np.array([4123, 4123, 4123, 4123, 4123])
num_non_abstained = np.array([2280, 2635, 3029, 3404, 3928])

# --- Baseline Data ---
# Baseline Models
baseline_names = ['Neo4j', 'Ultra (t=0.5)', 'Ultra (t=0.7)', 'Ultra (t=0.9)']

# Standard Metrics
baseline_precision = np.array([0.7670, 0.6261, 0.7930, 0.7816])
baseline_recall = np.array([0.5676, 0.7813, 0.6686, 0.6050])
baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)

# Abstention (Converted percentages to decimals: 23.3% -> 0.233)
baseline_abstention_rate = np.array([0.233, 0.037, 0.127, 0.195])

# Non-Abstention Metrics (Note: Neo4j recall non-abs was 0.7401 in your text)
baseline_non_abs_precision = np.array([1.0, 0.6501, 0.9084, 0.9709])
baseline_non_abs_recall = np.array([0.7401, 0.8113, 0.7658, 0.7516])

################ VALIDITY LINE CHART ONLY
fig, ax1 = plt.subplots(figsize=(5, 3))

# Create second y-axis for Precision and F1 bars
ax2 = ax1.twinx()

# Bar chart settings
width = 0.025  # width of individual bars
x = target_recall

# 1. Plot Bars on ax2 (Background)
bar1 = ax2.bar(x - width/2, precision, width, label='Precision', 
               color='#a1d99b', alpha=0.7, edgecolor='#31a354')
bar2 = ax2.bar(x + width/2, f1, width, label='F1 Score', 
               color='#bcbddc', alpha=0.7, edgecolor='#756bb1')

# 2. Plot Lines on ax1 (Foreground)
line1, = ax1.plot(target_recall, target_recall, 
                  linestyle='--', color='red', marker='s', 
                  markersize=7, linewidth=2, label='Ideal Recall')
line2, = ax1.plot(target_recall, empirical_recall, 
                  linestyle='-', color='blue', marker='o', 
                  markersize=7, linewidth=2, label='Empirical Recall')

# --- Formatting ---

# Ensure lines (ax1) are visually on top of bars (ax2)
ax1.set_zorder(ax2.get_zorder() + 1)
ax1.patch.set_visible(False) # Allows ax2 to be seen behind ax1

# Axis Labels
ax1.set_xlabel('Target Recall', fontsize=12)
ax1.set_ylabel('Recall (Lines)', fontsize=12, color='blue')
ax2.set_ylabel('Precision / F1 (Bars)', fontsize=12, color='#756bb1')

# Axis Limits and Ticks
ax1.set_xlim(0.43, 0.97)
ax1.set_ylim(0.4, 1.05)
ax2.set_ylim(0.3, 1.0) # Adjusted for Precision/F1 values
ax1.set_xticks(target_recall)

# Grid (drawn on the foreground axis)
ax1.grid(True, linestyle='-', linewidth=0.8, color='gray', alpha=0.2)

# Combine legends from both axes into one
handles = [line1, line2, bar1, bar2]
labels = [h.get_label() for h in handles]
ax1.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.25), 
           ncol=2, frameon=True, fontsize=10)

plt.tight_layout()
plt.savefig('validity_bars_lines.png', dpi=300, bbox_inches='tight')

################ --- Baseline Plot ---
print("WARNING: ALL DATA IS HARDCODED")
baseline_names = ['Neo4j', 'Ultra (t=0.5)', 'Ultra (t=0.7)', 'Ultra (t=0.9)']
baseline_precision = np.array([0.7670, 0.6261, 0.7930, 0.7816])
baseline_recall = np.array([0.5676, 0.7813, 0.6686, 0.6050])
baseline_f1 = 2 * (baseline_precision * baseline_recall) / (baseline_precision + baseline_recall)

# OURS Data (Selecting the last two indices to match the screenshot: 0.8 and 0.9)
ours_labels = ['OURS (r=0.8)', 'OURS (r=0.9)']
ours_precision = np.array([0.7667, 0.919]) # Using indices [3] and [4]
ours_recall = np.array([0.7615, 0.8961])
ours_f1 = 2 * (ours_precision * ours_recall) / (ours_precision + ours_recall)

# Combine lists
labels = baseline_names + ours_labels
precision = baseline_precision.tolist() + ours_precision.tolist()
recall = baseline_recall.tolist() + ours_recall.tolist()
f1 = baseline_f1.tolist() + ours_f1.tolist()

# --- Plotting with Style from Plot 1 ---
x = np.arange(len(labels))
width = 0.25  # Slightly wider bars to match style

fig, ax = plt.subplots(figsize=(7, 2.5))

# Precision bars: Green with dark green border
ax.bar(x - width, precision, width, label='Precision', 
       color='#B2E0B2', edgecolor='#4CAF50', linewidth=1)

# Recall bars: Red with dark red border (Keeping recall as bars for this layout)
ax.bar(x, recall, width, label='Recall', 
       color='#FFC1C1', edgecolor='#D32F2F', linewidth=1)

# F1 bars: Purple with dark purple border
ax.bar(x + width, f1, width, label='F1', 
       color='#CBC3E3', edgecolor='#7E57C2', linewidth=1)

# --- Aesthetic Refinements ---
ax.set_ylabel('Score')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=20, ha='right')

# Y-axis limits and light grid
ax.set_ylim(0, 1.1)
ax.grid(True, linestyle='-', alpha=0.3)
ax.set_axisbelow(True)

# Legend matching the boxed style
ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), 
           ncol=3, frameon=True, edgecolor='grey')

plt.tight_layout()

plt.savefig('baselines_vs_ours_fb15k_3hop.png', dpi=300, bbox_inches='tight')
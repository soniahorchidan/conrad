import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Configuration ---
GLOBAL_LAMBDA = 0.8
FIG_SIZE = (2, 4) # Tall and narrow aspect ratio

def draw_slider_figure(filename):
    fig, ax = plt.subplots(figsize=FIG_SIZE)
    
    # 1. Draw the "Track" (The Groove)
    track = patches.FancyBboxPatch(
        (0.45, 0.05), 0.1, 0.9,  # x, y, width, height
        boxstyle="round,pad=0.02", 
        facecolor="#f0f0f0", 
        edgecolor="#cccccc", 
        linewidth=1.5,
        zorder=1
    )
    ax.add_patch(track)
    
    # 2. Draw the "Fill" (The Active Part)
    fill = patches.FancyBboxPatch(
        (0.45, 0.05), 0.1, GLOBAL_LAMBDA * 0.9, # Scale height
        boxstyle="round,pad=0.02",
        facecolor="#d6eaf8", # Light blue fill
        edgecolor="none",
        zorder=2
    )
    ax.add_patch(fill)

    # 3. Draw the "Thumb" (The Handle)
    # We map 0.0-1.0 to the visual range 0.05-0.95
    visual_y = 0.05 + (GLOBAL_LAMBDA * 0.9)
    radius = 0.08
    
    # -- Manual Shadow (Correction) --
    # Draw a gray circle slightly offset to the bottom-right
    shadow = patches.Circle(
        (0.5 + 0.02, visual_y - 0.01), # Offset x, y
        radius=radius,
        facecolor="gray",
        edgecolor="none",
        alpha=0.3, # Transparent
        zorder=3
    )
    ax.add_patch(shadow)

    # -- Main Blue Circle --
    thumb = patches.Circle(
        (0.5, visual_y), 
        radius=radius, 
        facecolor="#1f77b4", # Standard Matplotlib Blue
        edgecolor="white", 
        linewidth=2,
        zorder=4
    )
    ax.add_patch(thumb)
    
    # 4. Add the Dashed "Projection" Line
    ax.annotate(
        "", xy=(1.2, visual_y), xytext=(0.5, visual_y),
        arrowprops=dict(arrowstyle="-", linestyle="--", color="#1f77b4", linewidth=2)
    )

    # 5. Labels and Annotations
    ax.text(0.5, 1.02, "Global Risk\nControl", ha='center', va='bottom', 
            fontsize=12, fontweight='bold')
    
    ax.text(0.15, visual_y, r'$\Lambda = {}$'.format(GLOBAL_LAMBDA), 
            ha='right', va='center', 
            fontsize=14, fontweight='bold', color='#1f77b4')

    ax.text(0.8, 0.05, "0.0", ha='left', va='center', fontsize=10, color='gray')
    ax.text(0.8, 0.95, "1.0", ha='left', va='center', fontsize=10, color='gray')
    
    ax.text(-0.1, 0.95, "Strict\n(High Precision)", ha='right', va='center', 
            fontsize=9, color='gray', style='italic')
    ax.text(-0.1, 0.05, "Loose\n(High Recall)", ha='right', va='center', 
            fontsize=9, color='gray', style='italic')

    # 6. Cleanup
    ax.set_xlim(-0.5, 1.2)
    ax.set_ylim(0, 1.1)
    ax.axis('off') 
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()

# --- Generate ---
draw_slider_figure("conrad_slider_panel.png")
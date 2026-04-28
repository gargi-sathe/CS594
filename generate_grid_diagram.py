import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches

fig, ax = plt.subplots(figsize=(6, 6))

grid_size = 5

# Draw grid lines
for i in range(grid_size + 1):
    ax.axhline(i, color='lightgray', linestyle='-', linewidth=1)
    ax.axvline(i, color='lightgray', linestyle='-', linewidth=1)

# Highlight hotspots (e.g. demand spikes)
hotspots = [(1, 1), (3, 3), (1, 4), (4, 1)]
for (x, y) in hotspots:
    rect = patches.Rectangle((x, y), 1, 1, linewidth=0, facecolor='#FFDDC1', alpha=0.8)
    ax.add_patch(rect)
    ax.text(x + 0.5, y + 0.5, "Demand", ha='center', va='center', color='#D9534F', fontsize=10, fontweight='bold')

# Mark Warehouses
warehouses = [(2, 2), (0, 4)]
for (x, y) in warehouses:
    rect = patches.Rectangle((x, y), 1, 1, linewidth=2, edgecolor='#1E293B', facecolor='#B9C3FF', alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + 0.5, y + 0.5, "Warehouse", ha='center', va='center', color='#001356', fontsize=10, fontweight='bold')

ax.set_xlim(0, grid_size)
ax.set_ylim(0, grid_size)
ax.set_xticks(np.arange(grid_size) + 0.5)
ax.set_yticks(np.arange(grid_size) + 0.5)
ax.set_xticklabels([])
ax.set_yticklabels([])
ax.tick_params(axis='both', which='both', length=0)
ax.set_title("Phase 1A: 5x5 Synthetic Grid Topology", fontsize=14, pad=20, fontname='Inter', fontweight='bold', color='#1E293B')

plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig("5x5_grid_diagram.png", dpi=300)
print("Diagram saved to 5x5_grid_diagram.png")

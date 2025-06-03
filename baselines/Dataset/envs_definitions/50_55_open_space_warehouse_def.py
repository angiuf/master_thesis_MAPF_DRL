import numpy as np
import matplotlib
import os

# Try to use interactive backend first, fallback to Agg if it fails
try:
    # Check if we're in a display-capable environment
    if os.environ.get('DISPLAY') and os.environ.get('DISPLAY') != '':
        matplotlib.use('TkAgg')  # Interactive backend for WSL with X11
        INTERACTIVE_MODE = True
    else:
        matplotlib.use('Agg')  # Non-interactive backend
        INTERACTIVE_MODE = False
except Exception:
    matplotlib.use('Agg')  # Fallback to non-interactive
    INTERACTIVE_MODE = False

import matplotlib.pyplot as plt

# Define the 50x55 warehouse with open space at the bottom
def create_warehouse_50_55_obstacles():
    """Create the obstacle matrix for 50x55 warehouse with open space at bottom."""
    rows, cols = 50, 55
    obstacles = np.zeros((rows, cols), dtype=int)
    
    # Create shelf structures (green areas in the image)
    # Top shelf rows
    for row in [2, 3, 6, 7, 10, 11]:
        for col in range(2, 26):  # Left section shelves
            obstacles[row, col] = 1
        for col in range(29, 53):  # Right section shelves
            obstacles[row, col] = 1
    
    # Middle shelf rows  
    for row in [14, 15, 18, 19, 22, 23]:
        for col in range(2, 26):  # Left section shelves
            obstacles[row, col] = 1
        for col in range(29, 53):  # Right section shelves
            obstacles[row, col] = 1
    
    # Bottom area should remain mostly open (this is our modification)
    # We'll keep the bottom area (rows 26-48) mostly clear except for some minimal obstacles
    
    # Add minimal obstacles in bottom open area for navigation interest
    obstacles[30:32, 15:17] = 1
    obstacles[30:32, 38:40] = 1
    obstacles[40:42, 15:17] = 1
    obstacles[40:42, 38:40] = 1
    
    return obstacles

# Create the obstacles array
warehouse_50_55_obstacles = create_warehouse_50_55_obstacles()

# Define open list (spawn/goal locations) - focusing on the bottom open area
warehouse_50_55_open_list = []

left_row_ranges = range(5,50)

for r in left_row_ranges:
    warehouse_50_55_open_list.append([r,0])
    warehouse_50_55_open_list.append([r,54])

for c in range(54):
    warehouse_50_55_open_list.append([49,c])
    
middle_row_starts = [4,5,8,9,12,13,16,17,20,21]

# Loop through the starting row of each middle band
for r in middle_row_starts:
    # Iterate through the 5 rows within the current band (e.g., 8 to 12)
    for c in range(5,23):
        warehouse_50_55_open_list.append([r,c])
    
    for c in range(32,50):
        warehouse_50_55_open_list.append([r,c])

ENV_DEFINITION = {
    "grid_size": "50_55",
    "map_name": "open_space_warehouse_bottom",
    "obstacles": warehouse_50_55_obstacles,
    "open_list": warehouse_50_55_open_list,
    "agent_counts": [4, 8, 16, 32, 64, 128, 256]
}

def plot_environment(obstacles, open_list, grid_size_str, map_name):
    """Plots the environment grid, marking obstacles and open list locations."""
    try:
        rows, cols = map(int, grid_size_str.split('_'))
    except ValueError:
        print(f"Error: Invalid grid_size format: {grid_size_str}. Expected 'rows_cols'.")
        return

    # Create a grid representing the environment
    # 0: open space, 1: obstacle, 2: open list location
    grid = np.zeros((rows, cols))

    # Mark obstacles
    if obstacles.size > 0:
        if obstacles.shape == (rows, cols):
            grid[obstacles == 1] = 1  # Mark obstacles from the grid array
        else:
            # Assuming obstacles is a list of coordinates [[r1, c1], [r2, c2], ...]
            for r, c in obstacles:
                if 0 <= r < rows and 0 <= c < cols:
                    grid[r, c] = 1

    # Mark open list locations
    for r, c in open_list:
        if 0 <= r < rows and 0 <= c < cols:
            if grid[r, c] == 1:
                print(f"Warning: Open list location ({r}, {c}) conflicts with obstacle!")
            else:
                grid[r, c] = 2

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    # Use white for free space, black for obstacles, light green for open list
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'lightgreen'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, cmap=cmap, norm=norm)

    # Draw gridlines
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    # Remove ticks and labels for cleaner look
    ax.set_xticks([])
    ax.set_yticks([])

    plt.title(f"Environment: {map_name} ({grid_size_str})")
    
    # Create legend handles
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, fc="white", ec="black", label='Free Space'),
        plt.Rectangle((0, 0), 1, 1, fc="black", ec="black", label='Obstacle'),
        plt.Rectangle((0, 0), 1, 1, fc="lightgreen", ec="black", label='Open List')
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0, 0, 0.85, 1])  # Adjust layout to make space for legend
    
    # Handle both interactive and non-interactive modes
    if INTERACTIVE_MODE:
        print("Displaying plot interactively...")
        plt.show()
    else:
        # Save the figure for non-interactive environments
        output_filename = f"{map_name}_{grid_size_str}.png"
        plt.savefig(output_filename, dpi=150, bbox_inches='tight')
        print(f"Environment plot saved as: {output_filename}")
        plt.close()  # Close the figure to free memory

if __name__ == "__main__":
    print(f"Plotting environment: {ENV_DEFINITION['map_name']} ({ENV_DEFINITION['grid_size']})")
    print(f"Total open list locations: {len(ENV_DEFINITION['open_list'])}")
    print(f"Obstacle density: {np.sum(ENV_DEFINITION['obstacles']) / (50 * 55) * 100:.1f}%")
    
    plot_environment(
        ENV_DEFINITION["obstacles"],
        ENV_DEFINITION["open_list"],
        ENV_DEFINITION["grid_size"],
        ENV_DEFINITION["map_name"]
    )
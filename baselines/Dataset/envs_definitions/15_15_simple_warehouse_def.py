import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# Define obstacles directly
warehouse_15_15_obstacles = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0],
                                    [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
                                    [0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0],
                                    [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
                                    [0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0],
                                    [0, 0, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, 0, 0, 0],
                                    [0, 0, 0, -1, 0, -1, 0, -1, 0, -1, 0, -1, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                    ])

# Define open list directly
warehouse_15_15_open_list = [[3, 0],
                            [4, 0],
                            [5, 0],
                            [6, 0],
                            [7, 0],
                            [8, 0],
                            [9, 0],
                            [10, 0],
                            [11, 0],
                            [12, 0],
                            [3, 14],
                            [4, 14],
                            [5, 14],
                            [6, 14],
                            [7, 14],
                            [8, 14],
                            [9, 14],
                            [10, 14],
                            [11, 14],
                            [12, 14],
                            [2, 4],
                            [2, 6],
                            [2, 8],
                            [2, 10],
                            [4, 4],
                            [4, 6],
                            [4, 8],
                            [4, 10],
                            [6, 4],
                            [6, 6],
                            [6, 8],
                            [6, 10],
                            [8, 4],
                            [8, 6],
                            [8, 8],
                            [8, 10],
                            [10, 4],
                            [10, 6],
                            [10, 8],
                            [10, 10],
                            [12, 4],
                            [12, 6],
                            [12, 8],
                            [12, 10]]


ENV_DEFINITION = {
    "grid_size": "15_15",
    "map_name": "simple_warehouse",
    "obstacles": warehouse_15_15_obstacles,
    "open_list": warehouse_15_15_open_list,
    "agent_counts": [4, 8, 12, 16, 20, 22]
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
    # Assuming obstacles are represented by -1 in the input array
    # or directly by coordinates if the obstacles array structure changes.
    if obstacles.size > 0:
        if obstacles.shape == (rows, cols):
            grid[obstacles == -1] = 1 # Mark obstacles from the grid array
        else:
             # Assuming obstacles is a list of coordinates [[r1, c1], [r2, c2], ...]
             for r, c in obstacles:
                 if 0 <= r < rows and 0 <= c < cols:
                     grid[r, c] = 1

    # Mark open list locations
    for r, c in open_list:
        if 0 <= r < rows and 0 <= c < cols:
            if grid[r, c] == 1:
                print(f"Warning: Open list location ({r},{c}) overlaps with an obstacle.")
            grid[r, c] = 2 # Mark open list locations

    # Plotting
    fig, ax = plt.subplots()
    # Use gray for obstacles, light green for open list, white for free space
    cmap = plt.cm.colors.ListedColormap(['white', 'black', 'lightgreen'])
    bounds = [-0.5, 0.5, 1.5, 2.5]
    norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)

    ax.imshow(grid, cmap=cmap, norm=norm)

    # Draw gridlines
    ax.set_xticks(np.arange(-.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-.5, rows, 1), minor=True)
    ax.grid(which='minor', color='grey', linestyle='-', linewidth=0.5)

    # Remove ticks and labels
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
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend
    plt.show()

if __name__ == "__main__":
    print(f"Plotting environment: {ENV_DEFINITION['map_name']} ({ENV_DEFINITION['grid_size']})")
    plot_environment(
        ENV_DEFINITION["obstacles"],
        ENV_DEFINITION["open_list"],
        ENV_DEFINITION["grid_size"],
        ENV_DEFINITION["map_name"]
    )

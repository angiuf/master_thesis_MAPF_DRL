import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_scenario(dataset_path, map_name, num_agents, case_id):
    map_file = dataset_path / map_name / 'input/map' / f'{map_name}.npy'
    map_data = np.load(map_file)

    case_filepath = dataset_path / map_name / 'input/start_and_goal' / f'{num_agents}_agents' / f'{map_name}_{num_agents}_agents_ID_{str(case_id).zfill(3)}.npy'
    pos = np.load(case_filepath, allow_pickle=True)
    start_pos = pos[:,0]
    goal_pos = pos[:,1]

    return map_data, start_pos, goal_pos

def visualize_scenario(map_data, start_pos, goal_pos):
    """
    Visualize the map scenario with obstacles, start positions, and goal positions.
    
    Args:
        map_data: 2D numpy array representing the map (0 = free space, 1 = obstacle)
        start_pos: Array of start positions for agents [(x1, y1), (x2, y2), ...]
        goal_pos: Array of goal positions for agents [(x1, y1), (x2, y2), ...]
    """
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Get map dimensions
    height, width = map_data.shape
    
    # Create the grid visualization
    # Map: 0 = free space (white), 1 = obstacle (black)
    colored_map = np.ones((height, width, 3))  # Start with white background
    
    # Set obstacles to black
    obstacle_mask = map_data == 1
    colored_map[obstacle_mask] = [0, 0, 0]  # Black for obstacles
    
    # Display the map
    ax.imshow(colored_map, origin='upper', extent=[0, width, 0, height])
    
    # Plot start positions (green circles) - swap x and y coordinates only for positions
    for i, (x, y) in enumerate(start_pos):
        circle = plt.Circle((y + 0.5, height - x - 0.5), 0.3, color='green', alpha=0.8, zorder=3)
        ax.add_patch(circle)
        ax.text(y + 0.5, height - x - 0.5, str(i), ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white', zorder=4)
    
    # Plot goal positions (red squares) - swap x and y coordinates only for positions
    for i, (x, y) in enumerate(goal_pos):
        square = patches.Rectangle((y + 0.2, height - x - 0.8), 0.6, 0.6,
                                 linewidth=2, edgecolor='red', facecolor='red', 
                                 alpha=0.8, zorder=3)
        ax.add_patch(square)
        ax.text(y + 0.5, height - x - 0.5, str(i), ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white', zorder=4)
    
    # Set up the plot
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    ax.set_title(f'Map Visualization\nAgents: {len(start_pos)}, Map size: {width}x{height}')
    
    # Add grid lines
    ax.set_xticks(range(0, width + 1))
    ax.set_yticks(range(0, height + 1))
    ax.grid(True, alpha=0.3)
    
    # Add legend
    legend_elements = [
        plt.Circle((0, 0), 0.1, color='green', alpha=0.8, label='Start positions'),
        patches.Rectangle((0, 0), 0.1, 0.1, facecolor='red', alpha=0.8, label='Goal positions'),
        patches.Rectangle((0, 0), 0.1, 0.1, facecolor='black', label='Obstacles'),
        patches.Rectangle((0, 0), 0.1, 0.1, facecolor='white', edgecolor='black', label='Free space')
    ]
    ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))
    
    plt.tight_layout()
    plt.show()
    
    # Print summary information
    print(f"\nMap Summary:")
    print(f"Map dimensions: {width} x {height}")
    print(f"Number of agents: {len(start_pos)}")
    print(f"Number of obstacles: {np.sum(map_data == 1)}")
    print(f"Free space cells: {np.sum(map_data == 0)}")
    print(f"Obstacle density: {np.sum(map_data == 1) / (width * height) * 100:.1f}%")

if __name__ == "__main__":
    dataset_path = Path(__file__).parent
    map_name = '50_55_open_space_warehouse_bottom'
    num_agents = 128
    case_id = 25
    agent_id = []

    map_data, start_pos, goal_pos = load_scenario(dataset_path, map_name, num_agents, case_id)

    print(f"Loaded scenario: {map_name}, Case ID: {case_id}, Number of agents: {num_agents}")
    print(map_data[2, 2:10])  # Print a small section of the map for verification
    # print(f"Start positions: {start_pos}")
    # print(f"Goal positions: {goal_pos}")

    for agent in agent_id:
        print(f"Agent {agent} - Start: {start_pos[agent]}, Goal: {goal_pos[agent]}")
    
    # Visualize the scenario instead of printing
    visualize_scenario(map_data, start_pos, goal_pos)
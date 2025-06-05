from od_mstar3 import cpp_mstar
from od_mstar3 import od_mstar
from od_mstar3.col_set_addition import OutOfTimeError,NoSolutionError
import os
import numpy as np
import yaml
import subprocess
import csv
import datetime
import time
import json
from pathlib import Path
from tqdm import tqdm

def count_collisions(solution, obstacle_map):
    """Count agent-agent and obstacle collisions in the solution."""
    agent_agent_collisions = 0
    obstacle_collisions = 0
    
    if not solution or len(solution) == 0:
        return agent_agent_collisions, obstacle_collisions
    
    # Convert solution format to timestep-based format
    timestep_based_solution = []
    num_agents = len(solution)
    max_timesteps = max(len(agent_path) for agent_path in solution)
    
    for timestep in range(max_timesteps):
        positions_at_timestep = []
        for agent_idx in range(num_agents):
            if timestep < len(solution[agent_idx]):
                # Extract only x, y coordinates from the solution tuple
                pos = solution[agent_idx][timestep]
                if len(pos) >= 2:
                    positions_at_timestep.append((pos[0], pos[1]))
                else:
                    positions_at_timestep.append(pos)
            else:
                # Agent has reached goal, stays at final position
                final_pos = solution[agent_idx][-1]
                if len(final_pos) >= 2:
                    positions_at_timestep.append((final_pos[0], final_pos[1]))
                else:
                    positions_at_timestep.append(final_pos)
        timestep_based_solution.append(positions_at_timestep)
    
    # Count collisions
    for timestep in range(len(timestep_based_solution)):
        positions = timestep_based_solution[timestep]
        
        # Check agent-agent collisions
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if positions[i] == positions[j]:
                    agent_agent_collisions += 1
        
        # Check obstacle collisions
        for pos in positions:
            if len(pos) >= 2:
                x, y = pos[0], pos[1]
                if (0 <= x < obstacle_map.shape[0] and 
                    0 <= y < obstacle_map.shape[1] and 
                    obstacle_map[x, y] == 1):  # Obstacle
                    obstacle_collisions += 1
    
    return agent_agent_collisions, obstacle_collisions

def convert_numpy_to_native(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, (np.integer, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_native(item) for item in obj]
    else:
        return obj

def compute_episode_length(solution):
    ep_len = 0
    for agent in solution:
        if len(agent) > ep_len:
            ep_len = len(agent)
    return ep_len

def compute_max_step(solution):
    max_step = 0
    for agent in solution:
        prev_pos = None
        step_count = 0
        for timestep in agent:
            curr_pos = (timestep[0], timestep[1])
            if prev_pos is not None and prev_pos != curr_pos:
                step_count += 1
            prev_pos = curr_pos
        if step_count > max_step:
            max_step = step_count
    return max_step

def generate_steps(solution):
    steps = []
    for agent in solution:
        prev_pos = None
        step_count = 0
        for timestep in agent:
            curr_pos = (timestep[0], timestep[1])
            if prev_pos is not None and prev_pos != curr_pos:
                step_count += 1
            prev_pos = curr_pos
        steps.append(step_count)
    return steps

def convert_solution(ts_solution):
    n_agents = len(ts_solution[0])
    solution = [[] for _ in range(n_agents)]
    for timestep in range(len(ts_solution)):
        for agent in range(n_agents):
            solution[agent].append(ts_solution[timestep][agent] + (timestep,))
    return solution

def get_csv_logger(model_dir, default_model_name):
    """Create CSV logger for results."""
    model_dir_path = Path(model_dir)
    csv_path = model_dir_path / f"log-{default_model_name}.csv"
    create_folders_if_necessary(csv_path)
    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

def create_folders_if_necessary(path):
    """Create necessary folders for the given path."""
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)

def convert_to_tuple(positions):
    new_positions = ()
    for pos in positions:
        new_positions += (tuple(pos),)
    return new_positions

if __name__ == '__main__':

    # Base paths
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent.parent
    dataset_dir = project_dir / 'baselines/Dataset'
    model_save_name = "ODrMstar"
    
    # Map configurations for testing
    map_configurations = [
        {
            "map_name": "15_15_simple_warehouse",
            "size": 15,
            "n_tests": 200,
            "list_num_agents": [4, 8, 12, 16, 20, 22]
        },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128, 256]
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128, 256]
        },
        {
            "map_name": "50_55_open_space_warehouse_bottom",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32, 64, 128, 256]
        }
    ]

    # CSV header - matching EECBS format
    header = ["agents", "success_rate", 
              "time", "time_std", "time_min", "time_max",
              "episode_length", "episode_length_std", "episode_length_min", "episode_length_max",
              "total_steps", "total_steps_std", "total_steps_min", "total_steps_max",
              "avg_steps", "avg_steps_std", "avg_steps_min", "avg_steps_max",
              "max_steps", "max_steps_std", "max_steps_min", "max_steps_max",
              "min_steps", "min_steps_std", "min_steps_min", "min_steps_max",
              "total_costs", "total_costs_std", "total_costs_min", "total_costs_max",
              "avg_costs", "avg_costs_std", "avg_costs_min", "avg_costs_max",
              "max_costs", "max_costs_std", "max_costs_min", "max_costs_max",
              "min_costs", "min_costs_std", "min_costs_min", "min_costs_max",
              "agent_collision_rate", "agent_collision_rate_std", "agent_collision_rate_min", "agent_collision_rate_max",
              "obstacle_collision_rate", "obstacle_collision_rate_std", "obstacle_collision_rate_min", "obstacle_collision_rate_max",
              "total_collision_rate", "total_collision_rate_std", "total_collision_rate_min", "total_collision_rate_max"]

    # Process each map configuration
    for config in map_configurations:
        map_name = config["map_name"]
        size = config["size"]
        num_cases = config["n_tests"]
        list_num_agents = config["list_num_agents"]

        print(f"\nProcessing map: {map_name}")
        
        # Check if map exists
        map_path = dataset_dir / map_name
        if not map_path.exists():
            print(f"WARNING: Map path does not exist: {map_path}")
            continue

        # Load map
        map_file_path = map_path / "input" / "map" / f"{map_name}.npy"
        if not map_file_path.exists():
            print(f"WARNING: Map file does not exist: {map_file_path}")
            continue
        
        map = np.load(map_file_path)
            
        # Create output directory
        output_dir = map_path / "output" / model_save_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Setup CSV logger
        results_path = base_dir / "results"
        results_path.mkdir(parents=True, exist_ok=True)
        date = datetime.datetime.now().strftime("%y-%m-%d-%H-%M-%S")
        sanitized_map_name = map_name.replace("/", "_").replace("\\", "_")
        csv_filename_base = f'{model_save_name}_{sanitized_map_name}_{date}'
        csv_file, csv_logger = get_csv_logger(results_path, csv_filename_base)

        csv_logger.writerow(header)
        csv_file.flush()

        for num_agents in list_num_agents:
            print(f"Starting tests for {num_agents} agents on map {map_name}")
            
            agent_dir = map_path / "input" / "start_and_goal" / f"{num_agents}_agents"
            
            # Check if agent directory exists
            if not agent_dir.exists():
                print(f"WARNING: Agent directory does not exist: {agent_dir}")
                continue
            
            # Create output directory for the current number of agents
            agent_output_dir = output_dir / f"{num_agents}_agents"
            agent_output_dir.mkdir(parents=True, exist_ok=True)

            # Initialize result storage - matching EECBS structure
            results = {
                'finished': [], 'time': [], 'episode_length': [],
                'total_steps': [], 'avg_steps': [], 'max_steps': [], 'min_steps': [],
                'total_costs': [], 'avg_costs': [], 'max_costs': [], 'min_costs': [],
                'crashed': [], 'agent_coll_rate': [], 'obstacle_coll_rate': [], 'total_coll_rate': []
            }

            for case_id in tqdm(range(num_cases), desc=f"Map: {map_name}, Agents: {num_agents}"):
                case_filepath = agent_dir / f"{map_name}_{num_agents}_agents_ID_{str(case_id).zfill(3)}.npy"
                
                if not case_filepath.exists():
                    print(f"WARNING: Case file does not exist: {case_filepath}")
                    continue
                
                pos = np.load(case_filepath, allow_pickle=True)
                starts = pos[:, 0]
                goals = pos[:, 1]

                for agent_id in range(num_agents):
                    starts[agent_id] = [starts[agent_id][1], starts[agent_id][0]]
                    goals[agent_id] = [goals[agent_id][1], goals[agent_id][0]]

                # Convert start and goals to tuple
                starts = convert_to_tuple(starts)
                goals = convert_to_tuple(goals)

                # Run OD_M* with timing
                start_time = time.time()
                try:
                    # Transpose map
                    map_T = np.transpose(map)
                    mstar_path = cpp_mstar.find_path(map_T, starts, goals, 2, 10)
                    solved = True
                except OutOfTimeError:
                    solved = False
                except NoSolutionError:
                    solved = False
                
                actual_time = time.time() - start_time

                # Initialize metrics similar to EECBS
                metrics = {
                    'finished': solved,
                    'time': actual_time,
                    'episode_length': 0,
                    'total_steps': 0,
                    'avg_steps': 0,
                    'max_steps': 0,
                    'min_steps': 0,
                    'total_costs': 0,
                    'avg_costs': 0,
                    'max_costs': 0,
                    'min_costs': 0,
                    'agent_collisions': 0,
                    'obstacle_collisions': 0,
                    'crashed': False,
                    'agent_coll_rate': 0,
                    'obstacle_coll_rate': 0,
                    'total_coll_rate': 0
                }
                
                solution = []
                if solved:
                    solution = convert_solution(mstar_path)
                    
                    # Calculate episode length
                    episode_length = max(len(path) for path in solution if len(path) > 0) - 1
                    metrics['episode_length'] = max(episode_length, 0)
                    
                    # Calculate steps and costs for each agent
                    agent_steps = []
                    agent_costs = []
                    
                    for agent_path in solution:
                        if len(agent_path) > 0:
                            # Count actual movements (steps)
                            steps = 0
                            last_move_timestep = 0
                            
                            for i in range(1, len(agent_path)):
                                current_pos = (agent_path[i][0], agent_path[i][1])
                                previous_pos = (agent_path[i-1][0], agent_path[i-1][1])
                                if current_pos != previous_pos:
                                    steps += 1
                                    last_move_timestep = i  # Update when agent last moved
                            
                            # Cost is the timestep when the agent stops moving (reaches goal and stays there)
                            # This is the last timestep where the agent moved + 1, or the total path length if never stopped
                            cost = last_move_timestep
                            
                            agent_steps.append(steps)
                            agent_costs.append(cost)
                        else:
                            agent_steps.append(0)
                            agent_costs.append(0)
                    
                    if agent_steps:
                        metrics['total_steps'] = sum(agent_steps)
                        metrics['avg_steps'] = np.mean(agent_steps)
                        metrics['max_steps'] = max(agent_steps)
                        metrics['min_steps'] = min(agent_steps)
                        
                        metrics['total_costs'] = sum(agent_costs)
                        metrics['avg_costs'] = np.mean(agent_costs)
                        metrics['max_costs'] = max(agent_costs)
                        metrics['min_costs'] = min(agent_costs)
                    
                    # Count collisions
                    agent_coll, obs_coll = count_collisions(solution, map)
                    metrics['agent_collisions'] = agent_coll
                    metrics['obstacle_collisions'] = obs_coll
                    metrics['crashed'] = (agent_coll + obs_coll) > 0
                    
                    if episode_length > 0 and num_agents > 0:
                        total_agent_timesteps = (episode_length + 1) * num_agents
                        metrics['agent_coll_rate'] = agent_coll / total_agent_timesteps
                        metrics['obstacle_coll_rate'] = obs_coll / total_agent_timesteps
                        metrics['total_coll_rate'] = (agent_coll + obs_coll) / total_agent_timesteps

                # Store results
                results['finished'].append(metrics['finished'])
                if metrics['finished']:
                    results['time'].append(metrics['time'])
                    results['episode_length'].append(metrics['episode_length'])
                    results['total_steps'].append(metrics['total_steps'])
                    results['avg_steps'].append(metrics['avg_steps'])
                    results['max_steps'].append(metrics['max_steps'])
                    results['min_steps'].append(metrics['min_steps'])
                    results['total_costs'].append(metrics['total_costs'])
                    results['avg_costs'].append(metrics['avg_costs'])
                    results['max_costs'].append(metrics['max_costs'])
                    results['min_costs'].append(metrics['min_costs'])
                    results['agent_coll_rate'].append(metrics['agent_coll_rate'])
                    results['obstacle_coll_rate'].append(metrics['obstacle_coll_rate'])
                    results['total_coll_rate'].append(metrics['total_coll_rate'])
                    results['crashed'].append(metrics['crashed'])

                # Save solution to file - matching EECBS format
                solution_filepath = agent_output_dir / f"solution_{model_save_name}_{map_name}_{num_agents}_agents_ID_{str(case_id).zfill(3)}.txt"
                with open(solution_filepath, 'w') as f:
                    f.write("Metrics:\n")
                    serializable_metrics = convert_numpy_to_native(metrics)
                    json.dump(serializable_metrics, f, indent=4)
                    f.write("\n\nSolution:\n")
                    if solution:
                        for agent_idx, agent_path in enumerate(solution):
                            f.write(f"Agent {agent_idx}: {agent_path}\n")
                    else:
                        f.write("No solution found.\n")

            # Calculate aggregated metrics - matching EECBS approach
            final_results = {}
            final_results['finished'] = sum(results['finished']) / len(results['finished']) if len(results['finished']) > 0 else 0

            # Calculate statistics for metrics when available
            metric_keys = ['time', 'episode_length', 'total_steps', 'avg_steps', 'max_steps', 'min_steps', 
                         'total_costs', 'avg_costs', 'max_costs', 'min_costs', 
                         'agent_coll_rate', 'obstacle_coll_rate', 'total_coll_rate']
            
            for key in metric_keys:
                if results[key]:
                    final_results[key] = np.mean(results[key])
                else:
                    final_results[key] = 0

            final_results['crashed'] = sum(results['crashed']) / len(results['crashed']) if len(results['crashed']) > 0 else 0

            print(f'SR: {final_results["finished"] * 100:.2f}%, Time: {final_results["time"]:.2f}s, '
                  f'Episode Length: {final_results["episode_length"]:.2f}, Total Collision Rate: {final_results["total_coll_rate"] * 100:.2f}%')

            # Write results to CSV - matching EECBS format
            data = [num_agents,
                    final_results['finished'] * 100,  # convert to percentage
                    final_results['time'],
                    np.std(results['time']) if results['time'] else 0,
                    np.min(results['time']) if results['time'] else 0,
                    np.max(results['time']) if results['time'] else 0,
                    final_results['episode_length'],
                    np.std(results['episode_length']) if results['episode_length'] else 0,
                    np.min(results['episode_length']) if results['episode_length'] else 0,
                    np.max(results['episode_length']) if results['episode_length'] else 0,
                    final_results['total_steps'],
                    np.std(results['total_steps']) if results['total_steps'] else 0,
                    np.min(results['total_steps']) if results['total_steps'] else 0,
                    np.max(results['total_steps']) if results['total_steps'] else 0,
                    final_results['avg_steps'],
                    np.std(results['avg_steps']) if results['avg_steps'] else 0,
                    np.min(results['avg_steps']) if results['avg_steps'] else 0,
                    np.max(results['avg_steps']) if results['avg_steps'] else 0,
                    final_results['max_steps'],
                    np.std(results['max_steps']) if results['max_steps'] else 0,
                    np.min(results['max_steps']) if results['max_steps'] else 0,
                    np.max(results['max_steps']) if results['max_steps'] else 0,
                    final_results['min_steps'],
                    np.std(results['min_steps']) if results['min_steps'] else 0,
                    np.min(results['min_steps']) if results['min_steps'] else 0,
                    np.max(results['min_steps']) if results['min_steps'] else 0,
                    final_results['total_costs'],
                    np.std(results['total_costs']) if results['total_costs'] else 0,
                    np.min(results['total_costs']) if results['total_costs'] else 0,
                    np.max(results['total_costs']) if results['total_costs'] else 0,
                    final_results['avg_costs'],
                    np.std(results['avg_costs']) if results['avg_costs'] else 0,
                    np.min(results['avg_costs']) if results['avg_costs'] else 0,
                    np.max(results['avg_costs']) if results['avg_costs'] else 0,
                    final_results['max_costs'],
                    np.std(results['max_costs']) if results['max_costs'] else 0,
                    np.min(results['max_costs']) if results['max_costs'] else 0,
                    np.max(results['max_costs']) if results['max_costs'] else 0,
                    final_results['min_costs'],
                    np.std(results['min_costs']) if results['min_costs'] else 0,
                    np.min(results['min_costs']) if results['min_costs'] else 0,
                    np.max(results['min_costs']) if results['min_costs'] else 0,
                    final_results['agent_coll_rate'] * 100,  # convert to percentage
                    np.std(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.min(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.max(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    final_results['obstacle_coll_rate'] * 100,  # convert to percentage
                    np.std(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.min(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.max(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    final_results['total_coll_rate'] * 100,  # convert to percentage
                    np.std(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.min(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.max(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0]
            
            csv_logger.writerow(data)
            csv_file.flush()

        csv_file.close()
        print(f"Completed testing for map: {map_name}")

    print("Finished all tests!")













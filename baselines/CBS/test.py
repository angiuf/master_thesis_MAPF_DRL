import os
import numpy as np
import yaml
import subprocess
import csv
import datetime
import time
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
                positions_at_timestep.append(solution[agent_idx][timestep][:2])  # (x, y)
            else:
                positions_at_timestep.append(solution[agent_idx][-1][:2])  # Stay at last position
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
                x, y = int(pos[0]), int(pos[1])
                if 0 <= x < obstacle_map['dimensions'][1] and 0 <= y < obstacle_map['dimensions'][0]:
                    if obstacle_map['obstacles'][y][x] == 1:  # obstacle
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

def generate_steps(solution, goals):
    steps = []
    costs = []
    for agent_idx, agent in enumerate(solution):
        prev_pos = None
        step_count = 0
        cost_count = 0
        for timestep in agent:
            curr_pos = (timestep[0], timestep[1])
            if prev_pos is not None and prev_pos != curr_pos:
                step_count += 1
            if curr_pos[0] != goals[agent_idx][0] or curr_pos[1] != goals[agent_idx][1]:
                cost_count = timestep[2]
            prev_pos = curr_pos
        steps.append(step_count)
        costs.append(cost_count)
    return steps, costs

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

if __name__ == '__main__':

    # Base paths
    base_dir = Path(__file__).resolve().parent
    project_dir = base_dir.parent.parent
    dataset_dir = project_dir / 'baselines/Dataset'
    model_save_name = "CBS"
    
    # Map configurations for testing
    map_configurations = [
        {
            "map_name": "15_15_simple_warehouse",
            "size": 15,
            "n_tests": 200,
            "list_num_agents": [4, 8, 12, 16, 20]
        },
        {
            "map_name": "50_55_simple_warehouse",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32]
        },
        {
            "map_name": "50_55_long_shelves",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16]
        },
        {
            "map_name": "50_55_open_space_warehouse_bottom",
            "size": 50,
            "n_tests": 200,
            "list_num_agents": [4, 8, 16, 32]
        }
    ]

    cbs_timeout = 60  # seconds

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
        map_file_path = map_path / "input" / "map" / f"{map_name}.yaml"
        if not map_file_path.exists():
            print(f"WARNING: Map file does not exist: {map_file_path}")
            continue
        
        with open(map_file_path, 'r') as map_file:
            map_data = yaml.load(map_file, Loader=yaml.FullLoader)

        obstacle_matrix = [[0 for j in range(map_data["map"]["dimensions"][1])] for i in range(map_data["map"]["dimensions"][0])]
        for obstacle in map_data["map"]["obstacles"]:
            row, col = obstacle  # obstacle[0] is row, obstacle[1] is column
            if 0 <= col < map_data["map"]["dimensions"][1] and 0 <= row < map_data["map"]["dimensions"][0]:
                obstacle_matrix[row][col] = 1

        map_data["map"]["obstacles"] = obstacle_matrix
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
                case_filepath = agent_dir / f"{map_name}_{num_agents}_agents_ID_{case_id:03d}.npy"
                
                if not case_filepath.exists():
                    print(f"WARNING: Case file does not exist: {case_filepath}")
                    # Add failure metrics
                    results['finished'].append(0)
                    results['crashed'].append(1)
                    results['agent_coll_rate'].append(0)
                    results['obstacle_coll_rate'].append(0)
                    results['total_coll_rate'].append(0)
                    continue
                
                pos = np.load(case_filepath, allow_pickle=True)
                starts = pos[:,0]
                goals = pos[:,1]

                # No coordinate swapping needed - keep original format
                # starts[agent_id] and goals[agent_id] are [row, col] from the file

                # Create the input yaml file
                case_dict = {"map": map_data["map"]}
                case_dict["agents"] = []
                for agent_id in range(num_agents):
                    agent_name = "agent" + str(agent_id)
                    # Convert [row, col] to [col, row] for CBS which expects [x, y]
                    agent_start = np.array([starts[agent_id][1], starts[agent_id][0]])  # [col, row] = [x, y]
                    agent_goal = np.array([goals[agent_id][1], goals[agent_id][0]])     # [col, row] = [x, y]
                    agent_dict = {"start": agent_start, "goal": agent_goal, "name": agent_name}
                    case_dict["agents"].append(agent_dict)
                
                # Convert numpy arrays to lists
                for agent in case_dict["agents"]:
                    agent["start"] = agent["start"].tolist()
                    agent["goal"] = agent["goal"].tolist()
                
                # Write the yaml file in the current directory
                case_yaml_filepath = "cbs_input.yaml"
                with open(case_yaml_filepath, 'w') as case_yaml_file:
                    yaml.dump(case_dict, case_yaml_file, default_flow_style=False)

                # Run CBS
                start_time = time.time()
                python_cbs_path = project_dir / "baselines/CBS/cbs.py"
                cbs_output_path = project_dir / "baselines/CBS/cbs_output.yaml"
                subprocess.run(["python3", str(python_cbs_path), str(case_yaml_filepath), str(cbs_output_path), "--timeout", str(cbs_timeout)])
                end_time = time.time()
                solving_time = end_time - start_time

                # Load the output yaml file
                with open(cbs_output_path, 'r') as output_yaml_file:
                    output_dict = yaml.load(output_yaml_file, Loader=yaml.FullLoader)
                
                out = dict()
                out["finished"] = output_dict["finished"]
                solution = [[] for _ in range(num_agents)]
                
                if out["finished"]:
                    for agent in range(num_agents):
                        key = "agent" + str(agent)
                        for timestep in output_dict["schedule"][key]:
                            solution[agent].append((timestep["x"], timestep["y"], timestep["t"]))

                    # Calculate metrics
                    tot_step = output_dict["cost"]
                    avg_step = float(output_dict["cost"]) / num_agents
                    max_step = compute_max_step(solution)
                    ep_len = compute_episode_length(solution)
                    steps, costs = generate_steps(solution, goals)
                    tot_step = sum(steps) if steps else 0
                    avg_step = sum(steps) / num_agents if num_agents > 0 else 0
                    max_step = max(steps) if steps else 0
                    min_step = min(steps) if steps else 0
                    tot_cost = sum(costs) if costs else 0
                    avg_cost = sum(costs) / num_agents if num_agents > 0 else 0
                    max_cost = max(costs) if costs else 0
                    min_cost = min(costs) if costs else 0
                    
                    # Calculate collision metrics
                    agent_collisions, obstacle_collisions = count_collisions(solution, map_data["map"])
                    total_collisions = agent_collisions + obstacle_collisions
                    
                    # Store metrics
                    results['finished'].append(1)
                    results['time'].append(solving_time)
                    results['episode_length'].append(ep_len)
                    results['total_steps'].append(tot_step)
                    results['avg_steps'].append(avg_step)
                    results['max_steps'].append(max_step)
                    results['min_steps'].append(min_step)
                    results['total_costs'].append(tot_cost)
                    results['avg_costs'].append(avg_cost)
                    results['max_costs'].append(max_cost)
                    results['min_costs'].append(min_cost)
                    results['crashed'].append(0)
                    results['agent_coll_rate'].append(agent_collisions / max(1, tot_step))
                    results['obstacle_coll_rate'].append(obstacle_collisions / max(1, tot_step))
                    results['total_coll_rate'].append(total_collisions / max(1, tot_step))

                    out["total_step"] = tot_step
                    out["avg_step"] = avg_step
                    out["max_step"] = max_step
                    out["episode_length"] = ep_len
                    out["time"] = solving_time
                    out["collisions"] = total_collisions
                else:
                    # Failed to solve
                    results['finished'].append(0)
                    results['crashed'].append(1)
                    results['agent_coll_rate'].append(0)
                    results['obstacle_coll_rate'].append(0)
                    results['total_coll_rate'].append(0)
                
                out["crashed"] = not out["finished"]
                
                # Save the output file
                case_output_filepath = agent_output_dir / f"solution_{model_save_name}_{map_name}_{num_agents}_agents_ID_{case_id:03d}.npy"
                save_dict = {"metrics": out, "solution": solution}
                np.save(case_output_filepath, save_dict)

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
                    final_results['finished'] * 100,
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
                    final_results['agent_coll_rate'] * 100,
                    np.std(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.min(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    np.max(results['agent_coll_rate']) * 100 if results['agent_coll_rate'] else 0,
                    final_results['obstacle_coll_rate'] * 100,
                    np.std(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.min(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    np.max(results['obstacle_coll_rate']) * 100 if results['obstacle_coll_rate'] else 0,
                    final_results['total_coll_rate'] * 100,
                    np.std(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.min(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0,
                    np.max(results['total_coll_rate']) * 100 if results['total_coll_rate'] else 0]

            csv_logger.writerow(data)
            csv_file.flush()

        csv_file.close()
        print(f"Completed testing for map: {map_name}")

    print("Finished all tests!")













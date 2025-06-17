import numpy as np
import os
import random
import sys
import glob
import importlib.util
import yaml
import argparse

# Add the parent directory of 'PRIMAL' to the Python path
# This allows importing from 'warehouse_environments'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This should be the 'Dataset' directory

def load_env_definition(filepath):
    """Loads the ENV_DEFINITION dictionary from a given python file."""
    spec = importlib.util.spec_from_file_location("env_def_module", filepath)
    if spec and spec.loader:
        env_def_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(env_def_module)
        if hasattr(env_def_module, 'ENV_DEFINITION'):
            return env_def_module.ENV_DEFINITION
    return None

def generate_random_cases(n_agents, n_cases, map_name, grid_size, obstacles, open_list):
    """
    Generates n_cases random MAPF instances for a given number of agents.

    Args:
        n_agents (int): The number of agents in each instance.
        n_cases (int): The number of instances (cases) to generate.
        map_name (str): The name of the map (e.g., 'simple_warehouse').
        grid_size (str): The size of the grid (e.g., '15_15').
        obstacles (np.ndarray): The obstacle map.
        open_list (list): A list of valid [row, col] coordinates for starts/goals.
    """
    base_output_dir = os.path.dirname(__file__) # Directory of the current script
    output_dir = os.path.join(base_output_dir, grid_size + "_" + map_name, "input", "start_and_goal", f"{n_agents}_agents")
    warehouse_dir = os.path.join(base_output_dir, grid_size + "_" + map_name, "input/map")

    # Check if the warehouse environment file exists
    warehouse_env_path = os.path.join(warehouse_dir, grid_size + "_" + map_name + ".npy")
    if not os.path.exists(warehouse_dir):
        os.makedirs(warehouse_dir)
    np.save(warehouse_env_path, obstacles)
    print(f"Saved warehouse environment to {warehouse_env_path}")

    map_yaml_filename = os.path.join(warehouse_dir, f"{grid_size}_{map_name}.yaml")
    if not os.path.exists(map_yaml_filename):
        write_yaml_file(obstacles, file_path=map_yaml_filename)
        print(f"Saved map and obstacles to {map_yaml_filename}")
    else:
        print(f"Map and obstacles already saved to {map_yaml_filename}")

    cpp_map_filename = os.path.join(warehouse_dir, f"{grid_size}_{map_name}.map")
    if not os.path.exists(cpp_map_filename):
        if write_cpp_map_file(obstacles, cpp_map_filename, map_name):
            print(f"Saved .map file to {cpp_map_filename}")
        else:
            print(f"Failed to create .map file at {cpp_map_filename}")
    else:
        print(f".map file already exists at {cpp_map_filename}")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Convert open_list elements to tuples for hashability
    open_list_tuples = [tuple(pos) for pos in open_list]

    # Check if we have enough positions - we need at least n_agents unique starts + n_agents unique goals
    # But goals can overlap with starts, so minimum is n_agents + (n_agents - max_overlap)
    # In worst case, we need 2*n_agents positions if no overlap is possible
    if len(open_list_tuples) < n_agents * 2:
        print(f"Warning: Limited positions ({len(open_list_tuples)}) for {n_agents} agents. Will try to generate with possible overlaps.")
    elif len(open_list_tuples) < n_agents:
        print(f"Error: Not enough open spots ({len(open_list_tuples)}) to place {n_agents} agents (need at least {n_agents} for unique starts).")
        return

    print(f"Generating {n_cases} cases for {n_agents} agents...")
    print(f"Available open positions: {len(open_list_tuples)}, Required minimum: {n_agents} unique starts + {n_agents} unique goals")

    successful_cases = 0
    max_attempts_per_case = 1000  # Increased for better success rate

    for i in range(n_cases):
        case_generated = False
        
        for attempt in range(max_attempts_per_case):
            try:
                # More efficient approach: pre-select unique positions for both starts and goals
                if len(open_list_tuples) < n_agents * 2:
                    # If we don't have enough positions for completely separate starts and goals,
                    # we need to allow some overlap (goals can be other agents' starts)
                    available_positions = open_list_tuples[:]
                    random.shuffle(available_positions)
                    
                    # Assign unique starts first
                    starts = available_positions[:n_agents]
                    
                    # For goals, we can use any position except the same agent's start
                    goals = []
                    available_goal_positions = open_list_tuples[:]
                    
                    for j in range(n_agents):
                        # Find positions that can be goals for agent j
                        valid_positions = [pos for pos in available_goal_positions if pos != starts[j]]
                        if not valid_positions:
                            raise ValueError(f"No valid goal for agent {j+1}")
                        
                        goal = random.choice(valid_positions)
                        goals.append(goal)
                        # Remove this goal from available positions to ensure unique goals
                        available_goal_positions.remove(goal)
                        
                        if len(available_goal_positions) == 0 and j < n_agents - 1:
                            # We've run out of goal positions, need to restart
                            raise ValueError("Ran out of unique goal positions")
                else:
                    # We have enough positions - use a simpler approach
                    available_positions = open_list_tuples[:]
                    random.shuffle(available_positions)
                    
                    # Take first 2*n_agents positions
                    selected_positions = available_positions[:n_agents * 2]
                    starts = selected_positions[:n_agents]
                    potential_goals = selected_positions[n_agents:]
                    
                    # Assign goals ensuring no agent gets its start as goal
                    goals = []
                    available_goals = potential_goals[:]
                    
                    for j in range(n_agents):
                        # Find goals that are not this agent's start
                        valid_goals = [g for g in available_goals if g != starts[j]]
                        if not valid_goals:
                            raise ValueError(f"No valid goal for agent {j+1}")
                        
                        goal = random.choice(valid_goals)
                        goals.append(goal)
                        available_goals.remove(goal)
                
                # Validation should always pass with this logic, but let's double-check
                assert len(set(starts)) == n_agents, "Duplicate starts"
                assert len(set(goals)) == n_agents, "Duplicate goals"
                for j in range(n_agents):
                    assert starts[j] != goals[j], f"Agent {j+1} starts at goal"
                
                # Save the case
                case_data = [[list(s), list(g)] for s, g in zip(starts, goals)]
                filename = f"{grid_size}_{map_name}_{n_agents}_agents_ID_{i:03d}.npy"
                filepath = os.path.join(output_dir, filename)
                np.save(filepath, np.array(case_data), allow_pickle=True)
                
                case_generated = True
                successful_cases += 1
                
                # Debug info for first few cases
                if i < 3:
                    print(f"  Case {i+1}: Generated successfully on attempt {attempt+1}")
                    print(f"    Sample starts: {starts[:3]}")
                    print(f"    Sample goals: {goals[:3]}")
                
                break
                        
            except (ValueError, AssertionError) as e:
                # Continue to next attempt, don't break
                if attempt % 100 == 0:  # Only print every 100th attempt to avoid spam
                    print(f"  Case {i+1}, attempt {attempt+1}: {e}")
                continue
            except Exception as e:
                print(f"  Case {i+1}, attempt {attempt+1}: Unexpected error - {e}")
                continue
                
        if not case_generated:
            print(f"Failed to generate case {i+1} after {max_attempts_per_case} attempts. Skipping.")
        
        # Progress update
        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{n_cases} cases processed, {successful_cases} successful")

    print(f"Finished generating {successful_cases}/{n_cases} cases for {n_agents} agents in {grid_size}_{map_name}.")
    
    if successful_cases < n_cases:
        print(f"Warning: Only generated {successful_cases} out of {n_cases} requested cases.")
        print(f"Consider reducing the number of agents or increasing the open space.")
        
    return successful_cases

def write_yaml_file(matrix, file_path='custom_map.yaml'):
    obstacles = []
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] == 1:
                obstacles.append((i, j))
    dimensions = [matrix.shape[0], matrix.shape[1]]
    data = {'map': {'dimensions': dimensions, 'obstacles': obstacles}}
    with open(file_path, 'w') as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

def write_cpp_map_file(obstacles, output_path, map_name):
    """Create a .map file from obstacle matrix for cpp_mstar."""
    try:
        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with open(output_path, 'w') as f:
            f.write(f"type octile\n")
            f.write(f"height {obstacles.shape[0]}\n")
            f.write(f"width {obstacles.shape[1]}\n")
            f.write("map\n")

            for row in obstacles:
                line = ''.join(['@' if cell == 1 else '.' for cell in row])
                f.write(line + '\n')

        print(f"Created .map file: {output_path}")
        return True
    except Exception as e:
        print(f"Error creating .map file {output_path}: {e}")
        return False

def validate_existing_cases(grid_size, map_name, n_agents, max_cases_to_check=10):
    """
    Validates existing case files for common issues like duplicate positions 
    or agents starting at their goals.
    
    Args:
        grid_size (str): The size of the grid (e.g., '50_55').
        map_name (str): The name of the map.
        n_agents (int): The number of agents.
        max_cases_to_check (int): Maximum number of cases to validate.
    
    Returns:
        dict: Summary of validation results.
    """
    base_output_dir = os.path.dirname(__file__)
    cases_dir = os.path.join(base_output_dir, grid_size + "_" + map_name, "input", "start_and_goal", f"{n_agents}_agents")
    
    if not os.path.exists(cases_dir):
        return {"error": f"Cases directory does not exist: {cases_dir}"}
    
    case_files = glob.glob(os.path.join(cases_dir, f"{grid_size}_{map_name}_{n_agents}_agents_ID_*.npy"))
    case_files = sorted(case_files)[:max_cases_to_check]  # Limit number of cases to check
    
    results = {
        "total_checked": 0,
        "valid_cases": 0,
        "issues": {
            "duplicate_starts": 0,
            "duplicate_goals": 0,
            "start_at_goal": 0,
            "load_errors": 0
        },
        "sample_issues": []
    }
    
    print(f"Validating up to {len(case_files)} existing cases for {n_agents} agents...")
    
    for case_file in case_files:
        case_id = os.path.basename(case_file).split("_ID_")[1].split(".")[0]
        results["total_checked"] += 1
        
        try:
            pos = np.load(case_file, allow_pickle=True)
            
            if pos.shape != (n_agents, 2, 2):
                results["issues"]["load_errors"] += 1
                results["sample_issues"].append(f"Case {case_id}: Invalid shape {pos.shape}")
                continue
            
            starts = [tuple(p[0]) for p in pos]
            goals = [tuple(p[1]) for p in pos]
            case_valid = True
            
            # Check for duplicate starts
            if len(set(starts)) != len(starts):
                results["issues"]["duplicate_starts"] += 1
                results["sample_issues"].append(f"Case {case_id}: Duplicate start positions")
                case_valid = False
            
            # Check for duplicate goals
            if len(set(goals)) != len(goals):
                results["issues"]["duplicate_goals"] += 1
                results["sample_issues"].append(f"Case {case_id}: Duplicate goal positions")
                case_valid = False
            
            # Check for agents starting at their goals
            start_at_goal_count = sum(1 for i in range(len(pos)) if tuple(pos[i][0]) == tuple(pos[i][1]))
            if start_at_goal_count > 0:
                results["issues"]["start_at_goal"] += 1
                results["sample_issues"].append(f"Case {case_id}: {start_at_goal_count} agents start at goal")
                case_valid = False
            
            if case_valid:
                results["valid_cases"] += 1
                
        except Exception as e:
            results["issues"]["load_errors"] += 1
            results["sample_issues"].append(f"Case {case_id}: Load error - {str(e)[:50]}")
    
    return results

def clean_invalid_cases(grid_size, map_name, n_agents, backup=True):
    """
    Removes invalid case files and optionally backs them up.
    
    Args:
        grid_size (str): The size of the grid.
        map_name (str): The name of the map.
        n_agents (int): The number of agents.
        backup (bool): Whether to backup invalid files before deletion.
    
    Returns:
        dict: Summary of cleaning results.
    """
    base_output_dir = os.path.dirname(__file__)
    cases_dir = os.path.join(base_output_dir, grid_size + "_" + map_name, "input", "start_and_goal", f"{n_agents}_agents")
    
    if not os.path.exists(cases_dir):
        return {"error": f"Cases directory does not exist: {cases_dir}"}
    
    case_files = glob.glob(os.path.join(cases_dir, f"{grid_size}_{map_name}_{n_agents}_agents_ID_*.npy"))
    
    results = {
        "total_checked": 0,
        "removed": 0,
        "backed_up": 0,
        "errors": []
    }
    
    backup_dir = None
    if backup:
        backup_dir = os.path.join(cases_dir, "invalid_backup")
        if not os.path.exists(backup_dir):
            os.makedirs(backup_dir)
    
    print(f"Checking {len(case_files)} cases for cleanup...")
    
    for case_file in case_files:
        case_id = os.path.basename(case_file).split("_ID_")[1].split(".")[0]
        results["total_checked"] += 1
        
        try:
            pos = np.load(case_file, allow_pickle=True)
            
            # Check for various issues
            invalid = False
            issues = []
            
            if pos.shape != (n_agents, 2, 2):
                invalid = True
                issues.append("invalid_shape")
            else:
                starts = [tuple(p[0]) for p in pos]
                goals = [tuple(p[1]) for p in pos]
                
                if len(set(starts)) != len(starts):
                    invalid = True
                    issues.append("duplicate_starts")
                
                if len(set(goals)) != len(goals):
                    invalid = True
                    issues.append("duplicate_goals")
                
                start_at_goal_count = sum(1 for i in range(len(pos)) if tuple(pos[i][0]) == tuple(pos[i][1]))
                if start_at_goal_count > 0:
                    invalid = True
                    issues.append("start_at_goal")
            
            if invalid:
                print(f"  Removing invalid case {case_id}: {', '.join(issues)}")
                
                if backup and backup_dir:
                    backup_path = os.path.join(backup_dir, os.path.basename(case_file))
                    import shutil
                    shutil.copy2(case_file, backup_path)
                    results["backed_up"] += 1
                
                os.remove(case_file)
                results["removed"] += 1
                
        except Exception as e:
            results["errors"].append(f"Case {case_id}: {str(e)}")
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate MAPF dataset cases')
    parser.add_argument('--cases', type=int, default=200, help='Number of cases to generate per configuration')
    parser.add_argument('--agents', type=int, nargs='+', default=[64], help='Agent counts to generate (e.g., --agents 32 64 128)')
    parser.add_argument('--env-filter', type=str, default='open_space', help='Environment name pattern filter')
    parser.add_argument('--validate-only', action='store_true', help='Only validate existing cases, do not generate new ones')
    parser.add_argument('--clean-invalid', action='store_true', help='Remove invalid cases after validation')
    parser.add_argument('--max-validate', type=int, default=20, help='Maximum number of cases to validate per environment')
    
    args = parser.parse_args()

    # --- Configuration from arguments ---
    N_CASES = args.cases
    AGENT_COUNTS_FILTER = args.agents
    ENV_NAME_FILTER = args.env_filter
    VALIDATE_ONLY = args.validate_only
    CLEAN_INVALID = args.clean_invalid
    MAX_VALIDATE = args.max_validate
    # --- End Configuration ---

    print(f"Configuration:")
    print(f"  Cases per config: {N_CASES}")
    print(f"  Agent counts: {AGENT_COUNTS_FILTER}")
    print(f"  Environment filter: {ENV_NAME_FILTER}")
    print(f"  Validate only: {VALIDATE_ONLY}")
    print(f"  Clean invalid: {CLEAN_INVALID}")
    print()

    # Find all environment definition files in the envs_definitions subdirectory
    script_dir = os.path.dirname(__file__)
    definitions_dir = os.path.join(script_dir, "envs_definitions") # Look inside the subdirectory
    definition_files = glob.glob(os.path.join(definitions_dir, "*_def.py"))

    if not definition_files:
        print(f"Error: No environment definition files (*_def.py) found in the '{definitions_dir}' directory.")
        sys.exit(1)

    print(f"Found definition files: {definition_files}")

    for def_file_path in definition_files:
        # Apply environment name filter if specified
        if ENV_NAME_FILTER and ENV_NAME_FILTER not in def_file_path:
            print(f"Skipping file '{def_file_path}' as it does not match the environment filter '{ENV_NAME_FILTER}'.")
            continue
            
        env_details = load_env_definition(def_file_path)
        if not env_details:
            print(f"Warning: Could not load ENV_DEFINITION from '{def_file_path}'. Skipping.")
            continue

        env_name = os.path.basename(def_file_path).replace('_def.py', '')
        print(f"\n--- Processing Environment: {env_name} ---")

        grid_size = env_details["grid_size"]
        map_name = env_details["map_name"]
        obstacles = env_details["obstacles"]
        open_list = env_details["open_list"] # Original list of lists
        agent_counts = env_details["agent_counts"]

        for n_agents in agent_counts:
            # Apply agent count filter if specified
            if AGENT_COUNTS_FILTER and n_agents not in AGENT_COUNTS_FILTER:
                print(f"Skipping {n_agents} agents as it is not in the specified agent counts filter.")
                continue
            
            # Validate existing cases if any
            print(f"\nValidating existing cases for {n_agents} agents...")
            validation_results = validate_existing_cases(grid_size, map_name, n_agents, max_cases_to_check=MAX_VALIDATE)
            
            if "error" not in validation_results:
                print(f"Validation results: {validation_results['valid_cases']}/{validation_results['total_checked']} cases are valid")
                if validation_results["sample_issues"]:
                    print("Sample issues found:")
                    for issue in validation_results["sample_issues"][:5]:  # Show first 5 issues
                        print(f"  - {issue}")
                        
                # If many issues found, consider regenerating
                issue_rate = (validation_results["total_checked"] - validation_results["valid_cases"]) / max(1, validation_results["total_checked"])
                if issue_rate > 0.1:  # More than 10% issues
                    print(f"Warning: High issue rate ({issue_rate:.1%}) detected in existing cases!")
                    
                # Clean invalid cases if requested
                if CLEAN_INVALID and issue_rate > 0:
                    print(f"Cleaning invalid cases...")
                    clean_results = clean_invalid_cases(grid_size, map_name, n_agents, backup=True)
                    if "error" not in clean_results:
                        print(f"Cleaned {clean_results['removed']} invalid cases (backed up: {clean_results['backed_up']})")
                        if clean_results['errors']:
                            print(f"Errors during cleaning: {clean_results['errors']}")
                    else:
                        print(f"Cleaning error: {clean_results['error']}")
            else:
                print(f"Validation error: {validation_results['error']}")
            
            # Generate new cases unless validate-only mode
            if not VALIDATE_ONLY:
                print(f"Generating cases for {n_agents} agents...")
                successful = generate_random_cases(
                    n_agents=n_agents,
                    n_cases=N_CASES,
                    map_name=map_name,
                    grid_size=grid_size,
                    obstacles=obstacles,
                    open_list=open_list # Pass the original list of lists
                )
                
                if successful < N_CASES:
                    print(f"Warning: Only generated {successful} out of {N_CASES} cases for {n_agents} agents.")
            else:
                print(f"Skipping generation (validate-only mode)")

    print("\nScript finished.")

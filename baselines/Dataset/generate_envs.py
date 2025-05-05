import numpy as np
import os
import random
import sys
import glob
import importlib.util

# Add the parent directory of 'PRIMAL' to the Python path
# This allows importing from 'warehouse_environments'
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir) # This should be the 'Dataset' directory
primal_dir = os.path.join(os.path.dirname(parent_dir), 'PRIMAL') # Go up one more level to 'baselines' then into 'PRIMAL'
sys.path.append(primal_dir)

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

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Convert open_list elements to tuples for hashability
    open_list_tuples = [tuple(pos) for pos in open_list]

    if len(open_list_tuples) < n_agents * 2:
        print(f"Error: Not enough open spots ({len(open_list_tuples)}) to place {n_agents} agents with unique starts and goals.")
        return

    print(f"Generating {n_cases} cases for {n_agents} agents...")

    for i in range(n_cases):
        available_positions = open_list_tuples[:] # Use the tuple version
        random.shuffle(available_positions)

        # Sample unique positions for all starts and goals first
        try:
            # sampled_positions will now be a list of tuples
            sampled_positions = random.sample(available_positions, n_agents * 2)
        except ValueError:
             print(f"Error: Could not sample {n_agents * 2} unique positions from {len(available_positions)} available spots for case {i+1}.")
             continue # Skip this case

        # current_starts and current_goals are lists of tuples
        current_starts = sampled_positions[:n_agents]
        current_goals = sampled_positions[n_agents:] # Initial goal assignments

        # Keep track of positions not used for starts or initial goals (list of tuples)
        remaining_open = [pos for pos in available_positions if pos not in sampled_positions]
        # Keep track of goals actually assigned to agents to ensure uniqueness (set of tuples)
        assigned_goal_positions = set()
        final_starts = [None] * n_agents
        final_goals = [None] * n_agents
        possible = True

        # Convert current_starts to a set of tuples for faster lookups
        current_starts_set = set(current_starts)

        for j in range(n_agents):
            start_pos = current_starts[j] # tuple
            goal_pos = current_goals[j] # tuple

            # --- Conflict Resolution ---
            # Reason 1: Start position is the same as the initial goal position
            if start_pos == goal_pos:
                resolved = False
                # Try 1: Use a remaining open spot
                if remaining_open:
                    idx_to_pop = random.randrange(len(remaining_open))
                    new_goal = remaining_open.pop(idx_to_pop) # tuple
                    # Check if this new goal is already assigned or is another agent's start
                    if new_goal not in assigned_goal_positions and new_goal not in current_starts_set:
                         goal_pos = new_goal
                         resolved = True
                    else:
                        remaining_open.insert(idx_to_pop, new_goal) # Put it back if unusable

                # Try 2: Swap with another agent's goal (conceptually)
                if not resolved:
                    swap_candidates = [k for k in range(n_agents) if k != j]
                    random.shuffle(swap_candidates)
                    for k in swap_candidates:
                        potential_new_goal = current_goals[k] # tuple
                        potential_k_original_goal = current_goals[j] # tuple
                        start_k = current_starts[k] # tuple

                        # Check if swapping works:
                        # 1. Agent j's start != Agent k's goal
                        # 2. Agent k's start != Agent j's original goal
                        # 3. Agent k's goal is not already assigned (might have been taken by another agent's conflict resolution)
                        if start_pos != potential_new_goal and \
                           start_k != potential_k_original_goal and \
                           potential_new_goal not in assigned_goal_positions:
                            goal_pos = potential_new_goal
                            resolved = True
                            # Note: We don't actually swap in current_goals here, just assign the new goal to agent j.
                            # Agent k will handle its own assignment later.
                            break

                # If still not resolved, conflict is unresolvable with simple methods
                if not resolved:
                    print(f"Error: Cannot resolve start==goal conflict for case {i+1}, agent {j+1}. Skipping case.")
                    possible = False
                    break # Break agent loop

            # Reason 2: The determined goal (original or resolved) is already assigned to a previous agent
            attempts = 0
            max_attempts_find_unique = len(available_positions) # Safety break
            # Check using tuples and the set
            while goal_pos in assigned_goal_positions and attempts < max_attempts_find_unique:
                # This goal is taken, find an alternative for agent j
                found_alternative = False
                # Try 1: Use remaining open spots
                if remaining_open:
                    idx_to_pop = random.randrange(len(remaining_open))
                    new_goal = remaining_open.pop(idx_to_pop) # tuple
                    if new_goal != start_pos and new_goal not in assigned_goal_positions and new_goal not in current_starts_set:
                        goal_pos = new_goal
                        found_alternative = True
                    else:
                        remaining_open.insert(idx_to_pop, new_goal) # Put back

                # Try 2: Use any available position not used as start/goal yet
                if not found_alternative:
                    # fallback_options will be list of tuples
                    fallback_options = [p for p in available_positions if p != start_pos and p not in assigned_goal_positions and p not in current_starts_set]
                    if fallback_options:
                        goal_pos = random.choice(fallback_options) # tuple
                        found_alternative = True
                    # else: No fallback options left

                if not found_alternative:
                    # If we couldn't find an alternative, it might be impossible or require complex chain reactions.
                    # For simplicity, we'll fail the case generation here.
                    print(f"Warning: Could not find unique alternative goal for agent {j+1} in case {i+1} (goal {goal_pos} already assigned).")
                    # Break the while loop, the final check below will catch it.
                    break

                attempts += 1


            # Final check: If goal is still assigned or resolution failed, skip case
            # Check using tuples and the set
            if goal_pos in assigned_goal_positions or not possible:
                 if possible: # Only print error if not already printed
                     print(f"Error: Failed to assign unique goal for agent {j+1} in case {i+1} after {attempts} attempts. Skipping case.")
                 possible = False
                 break # Break agent loop
            # --- End Conflict Resolution ---

            # Assign final positions for agent j (as tuples)
            final_starts[j] = start_pos
            final_goals[j] = goal_pos
            assigned_goal_positions.add(goal_pos) # Add the tuple to the set

        # --- End Agent Loop ---

        if not possible: # Case generation failed for one of the agents
            print(f"Skipping case {i+1} due to assignment errors.")
            continue # Skip to the next case

        # Save the case
        # Format: list of [start_row, start_col], [goal_row, goal_col] for each agent
        # Convert tuples back to lists for saving
        case_data = [[list(s), list(g)] for s, g in zip(final_starts, final_goals)] # New format: [[start], [goal]]
        filename = f"{map_name}_{grid_size}_{n_agents}_agents_ID_{i:03d}.npy" # More descriptive filename
        filepath = os.path.join(output_dir, filename)
        np.save(filepath, np.array(case_data, dtype=object)) # Save as object array to handle lists

    print(f"Finished generating {n_cases} cases for {n_agents} agents in {grid_size}_{map_name}.")


if __name__ == "__main__":
    # --- Configuration Variables ---
    N_CASES = 200 # Set the desired number of cases per configuration
    # --- End Configuration ---

    # Find all environment definition files in the envs_definitions subdirectory
    script_dir = os.path.dirname(__file__)
    definitions_dir = os.path.join(script_dir, "envs_definitions") # Look inside the subdirectory
    definition_files = glob.glob(os.path.join(definitions_dir, "*_def.py"))

    if not definition_files:
        print(f"Error: No environment definition files (*_def.py) found in the '{definitions_dir}' directory.")
        sys.exit(1)

    print(f"Found definition files: {definition_files}")

    for def_file_path in definition_files:
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
            print(f"Generating cases for {n_agents} agents...")
            generate_random_cases(
                n_agents=n_agents,
                n_cases=N_CASES,
                map_name=map_name,
                grid_size=grid_size,
                obstacles=obstacles,
                open_list=open_list # Pass the original list of lists
            )

    print("\nScript finished.")

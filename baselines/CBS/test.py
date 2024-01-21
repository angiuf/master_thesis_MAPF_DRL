import os
import numpy as np
import yaml
import subprocess
import csv
from tqdm import tqdm

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

def get_csv_logger(model_dir, default_model_name):
    csv_dir = os.path.join(model_dir, default_model_name)
    
    if not os.path.exists(csv_dir):
        os.makedirs(csv_dir)
    
    csv_path = os.path.join(csv_dir, default_model_name + ".csv")

    csv_file = open(csv_path, "a")
    return csv_file, csv.writer(csv_file)

if __name__ == '__main__':

    dataset_dir = "/home/andrea/Thesis/baselines/Dataset/"
    map_name = "50_55_simple_warehouse"
    model_save_name = "CBS"
    results_path = "/home/andrea/Thesis/results/50_55_simple_warehouse/"
    
    list_num_agents = [4, 8, 16, 32, 64, 128, 256]
    num_cases = 200
    cbs_timeout = 5 * 60  # seconds

    # Load map
    map_path = os.path.join(dataset_dir, map_name, "input", "map/")
    with open(map_path + map_name + ".yaml", 'r') as map_file:
        map = yaml.load(map_file, Loader=yaml.FullLoader)
    # print(map)
        
    # Create output directory
    output_dir = os.path.join(dataset_dir, map_name, "output", model_save_name + "/")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Initialize csv file
    csv_file, csv_logger = get_csv_logger(results_path, model_save_name)


    for num_agents in list_num_agents:
        agent_dir = os.path.join(dataset_dir, map_name, "input", "start_and_goal", str(num_agents) + "_agents/")
        
        # Create output directory for the current number of agents
        agent_output_dir = os.path.join(output_dir, str(num_agents) + "_agents/")
        if not os.path.exists(agent_output_dir):
            os.makedirs(agent_output_dir)

        # Metrics arrays initialization
        success = []
        total_step_list = []
        avg_step_list = []
        max_step_list = []
        episode_length_list = []

        print(f"Start testing for {num_agents} agents on map {map_name}")
        for case_id in tqdm(range(num_cases)):
            case_filepath = agent_dir + map_name + "_" + str(num_agents) + "_agents_ID_" + str(case_id).zfill(5) + ".npy"
            starts, goals = np.load(case_filepath, allow_pickle=True)

            for agent_id in range(num_agents):
                starts[agent_id] = [starts[agent_id][1], starts[agent_id][0]]
                goals[agent_id] = [goals[agent_id][1], goals[agent_id][0]]

            # print(starts)
            # print(goals)

            # Create the input yaml file
            case_dict = {"map": map["map"]}
            case_dict["agents"] = []
            for agent_id in range(num_agents):
                agent_name = "agent" + str(agent_id)
                agent_start = np.array([starts[agent_id][1], starts[agent_id][0]])
                agent_goal = np.array([goals[agent_id][1], goals[agent_id][0]])
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
            subprocess.run(["python3", "cbs.py", case_yaml_filepath, "cbs_output.yaml", "--timeout", str(cbs_timeout)])

            # Load the output yaml file
            with open("cbs_output.yaml", 'r') as output_yaml_file:
                output_dict = yaml.load(output_yaml_file, Loader=yaml.FullLoader)
            
            out = dict()
            out["finished"] = output_dict["finished"]
            success.append(out["finished"])
            solution = [[] for _ in range(num_agents)]
            if out["finished"]:
                for agent in range(num_agents):
                    key = "agent" + str(agent)
                    for timestep in output_dict["schedule"][key]:
                        solution[agent].append((timestep["x"], timestep["y"], timestep["t"]))

                tot_step = output_dict["cost"]
                avg_step = float(output_dict["cost"]) / num_agents
                max_step = compute_max_step(solution)
                ep_len = compute_episode_length(solution)
                
                total_step_list.append(tot_step)
                avg_step_list.append(avg_step)
                max_step_list.append(max_step)
                episode_length_list.append(ep_len)

                out["total_step"] = tot_step
                out["avg_step"] = avg_step
                out["max_step"] = max_step
                out["episode_length"] = ep_len
            out["crashed"] = False
            out["collisions"] = 0
            
            # print("Output:", out)
            # print("Solution:", solution)

            # Save the output file
            case_output_filepath = agent_output_dir + "solution_" + model_save_name + "_" + map_name + "_" + str(num_agents) + "_agents_ID_" + str(case_id).zfill(5) + ".npy"
            save_dict = {"metrics": out, "solution": solution}
            np.save(case_output_filepath, save_dict)
        
        # Write row in csv file
        header = ["num_agents", "success", "total_step", "avg_step", "max_step", "episode_length", "total_step_std", "avg_step_std", "max_step_std", "episode_length_std", "total_step_min", "avg_step_min", "max_step_min", "episode_length_min", "total_step_max", "avg_step_max", "max_step_max", "episode_length_max"]
        data = [num_agents, np.mean(success)*100, np.mean(total_step_list), np.mean(avg_step_list), np.mean(max_step_list), np.mean(episode_length_list), np.std(total_step_list), np.std(avg_step_list), np.std(max_step_list), np.std(episode_length_list), np.min(total_step_list), np.min(avg_step_list), np.min(max_step_list), np.min(episode_length_list), np.max(total_step_list), np.max(avg_step_list), np.max(max_step_list), np.max(episode_length_list)]
        if num_agents == 4:
            csv_logger.writerow(header)
        csv_logger.writerow(data)
        csv_file.flush()

            


            
            
            






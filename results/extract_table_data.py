#!/usr/bin/env python3
import pandas as pd
import os
import glob
import numpy as np

# Define the data directory
data_dir = "/home/andrea/CODE/master_thesis_MAPF_DRL/results/final_data"

# Define environments and their configurations
environments = {
    "15_15_simple_warehouse": {
        "name": "15x15 warehouse",
        "agents": [12, 16, 20]
    },
    "50_55_simple_warehouse": {
        "name": "50x55 warehouse", 
        "agents": [16, 32, 64]
    },
    "50_55_long_shelves": {
        "name": "50x55 long shelves warehouse",
        "agents": [16, 32, 64]
    },
    "50_55_open_space_warehouse_bottom": {
        "name": "50x55 open space warehouse",
        "agents": [16, 32, 64]
    }
}

# Define algorithms
algorithms = ["DCC", "CBS", "PRIMAL", "SCRIMP", "ODrMstar", "CBSH2-RTC", "EECBS", "SILLM"]

def extract_data(csv_file, n_agents):
    """Extract relevant data from CSV file for specific number of agents"""
    try:
        df = pd.read_csv(csv_file)
        # Filter for the specific number of agents
        agent_data = df[df['n_agents'] == n_agents]
        
        if agent_data.empty:
            return None
            
        # Get the first row (should be only one)
        row = agent_data.iloc[0]
        
        # Check success rate - if 0, return None values
        if row['success_rate'] == 0:
            return None
        
        # Handle different column names for total_steps
        total_steps_col = 'total_steps' if 'total_steps' in row else 'total_step'
            
        return {
            'episode_length': row['episode_length'],
            'total_steps': row[total_steps_col], 
            'total_costs': row['total_costs']
        }
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
        return None

def format_value(value):
    """Format value for table - return 'nan' if None, otherwise format to 2 decimal places"""
    if value is None or pd.isna(value):
        return "nan"
    return f"{value:.2f}"

# Process all data
results = {}

for env_key, env_info in environments.items():
    env_path = os.path.join(data_dir, env_key)
    if not os.path.exists(env_path):
        print(f"Warning: Environment path {env_path} does not exist")
        continue
        
    print(f"Processing {env_info['name']}...")
    
    for algorithm in algorithms:
        algo_path = os.path.join(env_path, algorithm)
        if not os.path.exists(algo_path):
            print(f"  Warning: Algorithm path {algo_path} does not exist")
            continue
            
        # Find CSV file
        csv_files = glob.glob(os.path.join(algo_path, "*.csv"))
        csv_files = [f for f in csv_files if not f.endswith('.Zone.Identifier')]
        
        if not csv_files:
            print(f"  Warning: No CSV file found in {algo_path}")
            continue
            
        csv_file = csv_files[0]  # Take the first (should be only one)
        
        for n_agents in env_info['agents']:
            data = extract_data(csv_file, n_agents)
            
            key = (env_key, algorithm, n_agents)
            results[key] = data

# Generate LaTeX table
print("\nGenerating LaTeX table...")

# Create table content
table_lines = []
table_lines.append("\\begin{table*}[h]")
table_lines.append("\\scriptsize")
table_lines.append("\\makebox[\\textwidth]{ % Center table, ignoring margins")
table_lines.append("\\begin{tabular}{|l|ccc|ccc|ccc|}")
table_lines.append("\\hline")
table_lines.append("\\multicolumn{1}{|c|}{\\textbf{algorithm/map}} & \\multicolumn{3}{c|}{\\textbf{episode length}} & \\multicolumn{3}{c|}{\\textbf{total steps}} & \\multicolumn{3}{c|}{\\textbf{sum of costs}} \\\\")
table_lines.append("\\hline")

for env_key, env_info in environments.items():
    # Add environment header
    agents_str = ", ".join([f" {a}" for a in env_info['agents']])
    table_lines.append(f"\\multicolumn{{10}}{{|c|}}{{\\textbf{{{env_info['name']} with{agents_str} agents}}}}\\\\")
    table_lines.append("\\hline")
    
    # Add data for each algorithm
    for algorithm in algorithms:
        # Map algorithm names for display
        display_name = algorithm
        if algorithm == "ODrMstar":
            display_name = "ODrM*"
        
        row_data = [display_name]
        
        # Get data for each metric and agent count
        for metric in ['episode_length', 'total_steps', 'total_costs']:
            for n_agents in env_info['agents']:
                key = (env_key, algorithm, n_agents)
                data = results.get(key)
                if data and metric in data:
                    value = data[metric]
                else:
                    value = None
                row_data.append(format_value(value))
        
        row_str = " & ".join(row_data) + " \\\\\\hline"
        table_lines.append(row_str)

table_lines.append("\\end{tabular}")
table_lines.append("}")
table_lines.append("\\caption{Experimental results, ``nan'' indicates that no run is successfully completed for that configuration.}")
table_lines.append("\\label{table:result_table}")
table_lines.append("\\end{table*}")

# Write to file
output_content = "\n".join(table_lines)
print(output_content)

# Save to file
with open("/home/andrea/CODE/master_thesis_MAPF_DRL/results/new_result_table.tex", "w") as f:
    f.write(output_content)

print(f"\nTable saved to new_result_table.tex")

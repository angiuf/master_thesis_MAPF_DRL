# filepath: /home/andrea/CODE/master_thesis_MAPF_DRL/results/plot_seaborn.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set up the plotting style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Define fixed colors for models
MODEL_COLORS = {
    'CBS': '#1f77b4',          # Blue
    'CBSH2-RTC': '#ff7f0e',    # Orange
    'EECBS': '#2ca02c',        # Green
    'ODrMstar': '#d62728',     # Red
    'DCC': '#9467bd',          # Purple
    'PRIMAL': '#8c564b',       # Brown
    'PICO': '#e377c2',         # Pink
    'SCRIMP': '#7f7f7f',       # Gray
    'AB_Mapper': '#bcbd22',    # Olive
    'SILLM': '#17becf',        # Cyan
}

# Define different line styles for each model to handle overlapping lines
MODEL_LINESTYLES = {
    'CBS': '-',                # Solid
    'CBSH2-RTC': '--',         # Dashed
    'EECBS': '-.',             # Dash-dot
    'ODrMstar': ':',           # Dotted
    'DCC': '-',                # Solid
    'PRIMAL': '--',            # Dashed
    'PICO': '-.',              # Dash-dot
    'SCRIMP': ':',             # Dotted
    'AB_Mapper': '-',          # Solid
    'SILLM': '--',             # Dashed
}

# Define different markers for each model
MODEL_MARKERS = {
    'CBS': 'o',                # Circle
    'CBSH2-RTC': 's',          # Square
    'EECBS': '^',              # Triangle up
    'ODrMstar': 'v',           # Triangle down
    'DCC': 'D',                # Diamond
    'PRIMAL': 'P',             # Plus (filled)
    'PICO': 'X',               # X (filled)
    'SCRIMP': '*',             # Star
    'AB_Mapper': 'h',          # Hexagon
    'SILLM': 'p',              # Pentagon
}

# Additional colors for models not in the predefined list
ADDITIONAL_COLORS = [
    '#1a9850', '#543005', '#8c6bb1', '#c51b8a', '#feb24c', 
    '#fd8d3c', '#fc4e2a', '#e31a1c', '#bd0026', '#800026',
    '#045a8d', '#2b8cbe', '#74a9cf', '#bdc9e1', '#f1eef6'
]

def ensure_model_colors(df):
    """Ensure all models in the dataset have assigned colors, line styles, and markers"""
    all_models = df['model'].unique()
    
    # Find models without predefined colors
    missing_models = [model for model in all_models if model not in MODEL_COLORS]
    
    # Default line styles and markers to cycle through
    default_linestyles = ['-', '--', '-.', ':']
    default_markers = ['o', 's', '^', 'v', 'D', 'P', 'X', '*', 'h', 'p']
    
    # Assign colors, line styles, and markers to missing models
    for i, model in enumerate(missing_models):
        # Assign color
        if i < len(ADDITIONAL_COLORS):
            MODEL_COLORS[model] = ADDITIONAL_COLORS[i]
        else:
            # Generate a random color if we run out of predefined additional colors
            import random
            MODEL_COLORS[model] = f"#{random.randint(0, 0xFFFFFF):06x}"
        
        # Assign line style
        MODEL_LINESTYLES[model] = default_linestyles[i % len(default_linestyles)]
        
        # Assign marker
        MODEL_MARKERS[model] = default_markers[i % len(default_markers)]
    
    if missing_models:
        print(f"Assigned colors to new models: {missing_models}")

def load_all_data(base_path):
    """Load all CSV data from the final_data directory"""
    all_data = []
    
    # Get all map directories
    final_data_path = Path(base_path) / "final_data"
    
    for map_dir in final_data_path.iterdir():
        if map_dir.is_dir():
            map_name = map_dir.name
            print(f"Processing map: {map_name}")
            
            # Get all model directories within this map
            for model_dir in map_dir.iterdir():
                if model_dir.is_dir():
                    model_name = model_dir.name
                    print(f"  Processing model: {model_name}")
                    
                    # Find CSV files in this model directory
                    csv_files = list(model_dir.glob("*.csv"))
                    # Filter out Zone.Identifier files
                    csv_files = [f for f in csv_files if not f.name.endswith("Zone.Identifier")]
                    
                    for csv_file in csv_files:
                        try:
                            df = pd.read_csv(csv_file)
                            df['map'] = map_name
                            df['model'] = model_name
                            
                            # Standardize column names
                            if 'n_agents' in df.columns:
                                df['agents'] = df['n_agents']
                            
                            all_data.append(df)
                        except Exception as e:
                            print(f"    Error reading {csv_file}: {e}")
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True, sort=False)
        print(f"\nTotal data points loaded: {len(combined_df)}")
        print(f"Maps: {combined_df['map'].unique()}")
        print(f"Models: {combined_df['model'].unique()}")
        return combined_df
    else:
        print("No data loaded!")
        return pd.DataFrame()

def get_metrics_to_plot(df):
    """Get list of main metrics to plot, excluding derived metrics and those that are all zeros"""
    # Exclude non-metric columns
    exclude_cols = ['map', 'model', 'agents', 'n_agents']
    
    # Exclude derived metrics (std, min, max, var, etc.)
    exclude_patterns = ['_std', '_min', '_max', '_var', '_variance', '_se', '_stderr', 
                       '_mean', '_median', '_mode', '_q1', '_q3', '_iqr', '_95th', 
                       '_99th', '_percentile', '_ci', '_confidence']
    
    metrics = []
    for col in df.columns:
        if col not in exclude_cols:
            # Check if column name contains any excluded patterns
            if any(pattern in col.lower() for pattern in exclude_patterns):
                continue
                
            # Check if the column has any non-zero values
            if df[col].notna().any() and (df[col] != 0).any():
                metrics.append(col)
    
    return metrics

def create_plot(df, metric, map_name, save_dir):
    """Create a plot for a specific metric and map"""
    # Filter data for this map
    map_data = df[df['map'] == map_name].copy()
    
    if map_data.empty:
        print(f"No data for map {map_name}")
        return
    
    # For success rate and collision rate, keep all data points (including 0s)
    # For other metrics, filter out 0 values
    if 'success' not in metric.lower():
        if 'collision' in metric.lower():
            map_data = map_data[map_data['success_rate'] > 0]  # Ensure we only keep rows with non-zero success rate
        else:
            # Remove rows where the metric is 0
            map_data = map_data[map_data[metric] != 0]

        
        # After filtering, check if we still have data
        if map_data.empty:
            print(f"Skipping {metric} for {map_name} - all non-zero values filtered out")
            return
    
    
    # Check if metric has any data after filtering
    if map_data[metric].isna().all():
        print(f"Skipping {metric} for {map_name} - all values are NaN")
        return
    
    # Check if std column exists
    std_metric = f"{metric}_std"
    has_std = std_metric in map_data.columns and not map_data[std_metric].isna().all()
    
    # Use the global color palette - this ensures consistency across all plots
    model_palette = MODEL_COLORS.copy()
    
    # Create main plot with sidebar layout
    fig = plt.figure(figsize=(16, 8))
    
    # Create main plot (takes up left 75% of the figure)
    ax1 = plt.subplot2grid((2, 10), (0, 0), colspan=7, rowspan=2)
    
    # Sort the data to ensure proper ordering
    map_data = map_data.sort_values('agents')
    
    # Convert agents to string to ensure categorical treatment
    map_data['agents_str'] = map_data['agents'].astype(str)
    
    # Get unique agent counts for categorical x-axis
    unique_agents = sorted(map_data['agents'].unique())
    unique_agents_str = [str(x) for x in unique_agents]
    
    # Create the main metric plot with different line styles and markers for each model
    # Instead of using seaborn lineplot, plot each model individually to apply custom styles
    
    # Add small horizontal offsets to prevent complete overlap of points
    offset_step = 0.02  # Small offset between models
    model_list = sorted(map_data['model'].unique())
    
    for i, model in enumerate(model_list):
        model_data = map_data[map_data['model'] == model].copy()
        
        # Get style attributes for this model
        color = MODEL_COLORS.get(model, '#000000')
        linestyle = MODEL_LINESTYLES.get(model, '-')
        marker = MODEL_MARKERS.get(model, 'o')
        
        # Create slight horizontal offset for each model to prevent overlap
        x_positions = range(len(unique_agents_str))
        offset = (i - len(model_list)/2) * offset_step
        x_offset = [x + offset for x in x_positions]
        
        # Map agents to their position indices
        agent_to_pos = {str(agent): pos for pos, agent in enumerate(unique_agents_str)}
        model_x_positions = [agent_to_pos[str(agent)] + offset for agent in model_data['agents']]
        
        # Plot the line for this model
        ax1.plot(model_x_positions, model_data[metric], 
                color=color, linestyle=linestyle, marker=marker,
                linewidth=2.5, markersize=8, label=model, alpha=0.8)
    
    # Set x-axis to show all agent counts as categorical with proper ordering
    ax1.set_xticks(range(len(unique_agents_str)))
    ax1.set_xticklabels(unique_agents_str)
    
    # Customize the main plot
    ax1.set_title(f'{metric.replace("_", " ").title()}\n{map_name.replace("_", " ").title()}', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Number of Agents', fontsize=14)
    ax1.set_ylabel(metric.replace("_", " ").title(), fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Create legend handles for ALL models (not just those in current dataset)
    # This ensures consistent legend across all plots
    all_models_in_data = df['model'].unique()  # Get all models from the entire dataset
    legend_handles = []
    
    for model in sorted(all_models_in_data):
        if model in MODEL_COLORS:  # Only include models that have defined colors
            from matplotlib.lines import Line2D
            line = Line2D([0], [0], 
                         color=MODEL_COLORS[model], 
                         linestyle=MODEL_LINESTYLES.get(model, '-'),
                         marker=MODEL_MARKERS.get(model, 'o'),
                         linewidth=2.5, markersize=8, label=model, alpha=0.8)
            legend_handles.append(line)
    
    # Position legend closer to main plot - left side and higher up
    legend = fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.75, 0.8), 
                       fontsize=11, title='Model', title_fontsize=12, frameon=True, 
                       fancybox=True, shadow=True)
    
    # Create the std subplot if data exists
    if has_std:
        # Filter out zero std values for better visualization
        std_data = map_data[map_data[std_metric] > 0] if 'success' not in metric.lower() and 'collision' not in metric.lower() else map_data
        
        if not std_data.empty:
            # Create std deviation subplot in the right bottom area
            ax2 = plt.subplot2grid((2, 10), (1, 7), colspan=3)
            
            # Create bar plot in std subplot
            sns.barplot(data=std_data, x='agents_str', y=std_metric, hue='model', ax=ax2, palette=model_palette)
            
            # Set reasonable Y-axis limit for std plot to avoid outlier scaling
            # Use 95th percentile or median + 3*IQR, whichever is smaller
            std_values = std_data[std_metric].dropna()
            if len(std_values) > 0:
                q95 = std_values.quantile(0.95)
                q75 = std_values.quantile(0.75)
                q25 = std_values.quantile(0.25)
                median = std_values.median()
                iqr_limit = median + 3 * (q75 - q25)
                
                # Use the smaller of 95th percentile or IQR-based limit
                y_max = min(q95, iqr_limit)
                
                # Ensure we have a reasonable minimum range
                y_min = 0
                if y_max <= y_min:
                    y_max = std_values.max()
                
                ax2.set_ylim(y_min, y_max * 1.1)  # Add 10% padding at top
            
            # Customize the std subplot
            ax2.set_title('Standard Deviation', fontsize=12, fontweight='bold')
            ax2.set_xlabel('')  # Remove x-axis label
            ax2.set_ylabel('')  # Remove y-axis label
            ax2.tick_params(axis='both', which='major', labelsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Remove legend from std subplot (main legend is in sidebar)
            if ax2.get_legend():
                ax2.get_legend().remove()
    
    # Overall layout
    plt.tight_layout()
    
    # Create map-specific directory
    safe_map_name = map_name.replace("/", "_").replace(" ", "_")
    map_dir = save_dir / safe_map_name
    map_dir.mkdir(exist_ok=True)
    
    # Save the plot in the map-specific directory
    safe_metric_name = metric.replace("/", "_").replace(" ", "_")
    filename = f"{safe_metric_name}.svg"
    filepath = map_dir / filename
    
    plt.savefig(filepath, format='svg', bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {safe_map_name}/{filename}")

def main():
    # Define paths
    base_path = Path("/home/andrea/CODE/master_thesis_MAPF_DRL/results")
    save_dir = base_path / "plots"
    save_dir.mkdir(exist_ok=True)
    
    # Load all data
    print("Loading data...")
    df = load_all_data(base_path)
    
    if df.empty:
        print("No data to plot!")
        return
    
    # Ensure all models have assigned colors
    ensure_model_colors(df)
    
    # Get metrics to plot
    metrics = get_metrics_to_plot(df)
    print(f"\nMetrics to plot: {metrics}")
    
    # Get unique maps
    maps = df['map'].unique()
    print(f"Maps to process: {maps}")
    
    # Print color assignments
    print(f"\nModel color assignments:")
    for model in sorted(df['model'].unique()):
        print(f"  {model}: {MODEL_COLORS[model]}")
    
    # Create plots for each metric and map combination
    total_plots = len(metrics) * len(maps)
    plot_count = 0
    
    print(f"\nCreating {total_plots} plots...")
    
    for map_name in maps:
        print(f"\nProcessing map: {map_name}")
        
        for metric in metrics:
            plot_count += 1
            print(f"  [{plot_count}/{total_plots}] Creating plot for {metric}")
            
            try:
                create_plot(df, metric, map_name, save_dir)
            except Exception as e:
                print(f"    Error creating plot for {metric} on {map_name}: {e}")
    
    print(f"\nCompleted! All plots saved in: {save_dir}")
    
    # Print summary statistics
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    for map_name in maps:
        print(f"\n{map_name}:")
        map_data = df[df['map'] == map_name]
        print(f"  Models: {map_data['model'].unique()}")
        print(f"  Agent counts: {sorted(map_data['agents'].unique())}")
        print(f"  Data points: {len(map_data)}")

if __name__ == "__main__":
    main()
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
    
    # Exclude specific models from collision rate metrics
    if 'collision' in metric.lower():
        models_to_exclude = ['CBS', 'CBSH2-RTC', 'EECBS', 'ODrMstar']
        map_data = map_data[~map_data['model'].isin(models_to_exclude)]
        if map_data.empty:
            print(f"Skipping {metric} for {map_name} - all models excluded for collision rate")
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
    filename = f"{safe_metric_name}.pdf"
    filepath = map_dir / filename
    
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()
    
    print(f"Saved plot: {safe_map_name}/{filename}")

def plot_metric_across_environments(df, metric, maps, save_dir):
    """Create a single figure with 4 subplots (one for each environment) for a given metric."""
    import math
    n_envs = len(maps)
    
    # Check if std column exists for this metric
    std_metric = f"{metric}_std"
    has_std = std_metric in df.columns and not df[std_metric].isna().all()
    
    # Create figure with variance subplots if std data exists
    if has_std:
        # Create a figure with main plots in single column and variance plots in the rightmost column
        fig = plt.figure(figsize=(16, 4 * n_envs))
        
        # Create main plots (left column) and variance plots (right column)
        main_axes = []
        var_axes = []
        
        for idx, map_name in enumerate(maps):
            # Main plot positioning - single column layout
            ax_main = plt.subplot2grid((n_envs, 10), (idx, 0), colspan=7)
            main_axes.append(ax_main)
            
            # Variance plot positioning - single rightmost column, stacked vertically for all environments
            ax_var = plt.subplot2grid((n_envs, 10), (idx, 7), colspan=3)
            var_axes.append(ax_var)
            
            # Plot main metric
            map_data = df[df['map'] == map_name].copy()
            if map_data.empty:
                ax_main.set_title(f"{map_name}\n(No data)")
                ax_main.axis('off')
                ax_var.axis('off')
                continue
            
            # Exclude specific models from collision rate metrics
            if 'collision' in metric.lower():
                models_to_exclude = ['CBS', 'CBSH2-RTC', 'EECBS', 'ODrMstar']
                map_data = map_data[~map_data['model'].isin(models_to_exclude)]
                if map_data.empty:
                    ax_main.set_title(f"{map_name}\n(Models excluded for collision rate)")
                    ax_main.axis('off')
                    ax_var.axis('off')
                    continue
                
            # Filter data similar to individual plots
            if 'success' not in metric.lower():
                if 'collision' in metric.lower():
                    map_data = map_data[map_data['success_rate'] > 0]
                else:
                    map_data = map_data[map_data[metric] != 0]
                    
                if map_data.empty:
                    ax_main.set_title(f"{map_name}\n(No data after filtering)")
                    ax_main.axis('off')
                    ax_var.axis('off')
                    continue
            
            map_data = map_data.sort_values('agents')
            map_data['agents_str'] = map_data['agents'].astype(str)
            unique_agents = sorted(map_data['agents'].unique())
            unique_agents_str = [str(x) for x in unique_agents]
            offset_step = 0.02
            model_list = sorted(map_data['model'].unique())
            
            # Plot main metric
            for i, model in enumerate(model_list):
                model_data = map_data[map_data['model'] == model].copy()
                color = MODEL_COLORS.get(model, '#000000')
                linestyle = MODEL_LINESTYLES.get(model, '-')
                marker = MODEL_MARKERS.get(model, 'o')
                agent_to_pos = {str(agent): pos for pos, agent in enumerate(unique_agents_str)}
                offset = (i - len(model_list)/2) * offset_step
                model_x_positions = [agent_to_pos[str(agent)] + offset for agent in model_data['agents']]
                ax_main.plot(model_x_positions, model_data[metric], color=color, linestyle=linestyle, marker=marker,
                        linewidth=2.5, markersize=8, label=model, alpha=0.8)
            
            map_title = map_name.replace('_', ' ')
            map_title = map_title + f" ({idx+1})"
            ax_main.set_xticks(range(len(unique_agents_str)))
            ax_main.set_xticklabels(unique_agents_str)
            ax_main.set_title(map_title.title(), fontsize=14, fontweight='bold')
            ax_main.set_xlabel('Number of Agents', fontsize=12)
            # Always show y-label for single column layout
            ax_main.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax_main.grid(True, alpha=0.3)
            
            # Plot variance subplot
            std_data = map_data[map_data[std_metric] > 0] if 'success' not in metric.lower() and 'collision' not in metric.lower() else map_data
            
            if not std_data.empty:
                model_palette = MODEL_COLORS.copy()
                sns.barplot(data=std_data, x='agents_str', y=std_metric, hue='model', ax=ax_var, palette=model_palette)
                
                # Set reasonable Y-axis limit for std plot
                std_values = std_data[std_metric].dropna()
                if len(std_values) > 0:
                    q95 = std_values.quantile(0.95)
                    q75 = std_values.quantile(0.75)
                    q25 = std_values.quantile(0.25)
                    median = std_values.median()
                    iqr_limit = median + 3 * (q75 - q25)
                    y_max = min(q95, iqr_limit)
                    y_min = 0
                    if y_max <= y_min:
                        y_max = std_values.max()
                    ax_var.set_ylim(y_min, y_max * 1.1)
                
                ax_var.set_title(f'Std Dev ({idx+1})', fontsize=12, fontweight='bold')
                ax_var.set_xlabel('')
                ax_var.set_ylabel('')
                ax_var.tick_params(axis='both', which='major', labelsize=9)
                ax_var.grid(True, alpha=0.3)
                
                # Remove legend from variance subplot
                if ax_var.get_legend():
                    ax_var.get_legend().remove()
            else:
                ax_var.set_title('Std Dev\n(No data)', fontsize=12)
                ax_var.axis('off')
        
        # Get legend from the first main plot that has data
        handles, labels = None, None
        for ax in main_axes:
            if ax.get_legend_handles_labels()[0]:
                handles, labels = ax.get_legend_handles_labels()
                break
        
        if handles and labels:
            fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(6, len(labels)), fontsize=11, title='Model', title_fontsize=12)
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
                
    else:
        # Single column layout without variance plots
        fig, axes = plt.subplots(n_envs, 1, figsize=(12, 4 * n_envs), sharey=True)
        if n_envs == 1:
            axes = [axes]  # Ensure axes is always a list

        for idx, map_name in enumerate(maps):
            ax = axes[idx]
            map_data = df[df['map'] == map_name].copy()
            if map_data.empty:
                ax.set_title(f"{map_name}\n(No data)")
                ax.axis('off')
                continue
            
            # Exclude specific models from collision rate metrics
            if 'collision' in metric.lower():
                models_to_exclude = ['CBS', 'CBSH2-RTC', 'EECBS', 'ODrMstar']
                map_data = map_data[~map_data['model'].isin(models_to_exclude)]
                if map_data.empty:
                    ax.set_title(f"{map_name}\n(Models excluded for collision rate)")
                    ax.axis('off')
                    continue
            map_data = map_data.sort_values('agents')
            map_data['agents_str'] = map_data['agents'].astype(str)
            unique_agents = sorted(map_data['agents'].unique())
            unique_agents_str = [str(x) for x in unique_agents]
            offset_step = 0.02
            model_list = sorted(map_data['model'].unique())
            for i, model in enumerate(model_list):
                model_data = map_data[map_data['model'] == model].copy()
                color = MODEL_COLORS.get(model, '#000000')
                linestyle = MODEL_LINESTYLES.get(model, '-')
                marker = MODEL_MARKERS.get(model, 'o')
                agent_to_pos = {str(agent): pos for pos, agent in enumerate(unique_agents_str)}
                offset = (i - len(model_list)/2) * offset_step
                model_x_positions = [agent_to_pos[str(agent)] + offset for agent in model_data['agents']]
                ax.plot(model_x_positions, model_data[metric], color=color, linestyle=linestyle, marker=marker,
                        linewidth=2.5, markersize=8, label=model, alpha=0.8)
            ax.set_xticks(range(len(unique_agents_str)))
            ax.set_xticklabels(unique_agents_str)
            ax.set_title(map_name.replace('_', ' ').title(), fontsize=14, fontweight='bold')
            ax.set_xlabel('Number of Agents', fontsize=12)
            # Always show y-label for single column layout
            ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # No need to remove unused axes for single column layout
        
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.02), ncol=min(4, len(labels)), fontsize=11, title='Model', title_fontsize=12)
        plt.tight_layout(rect=[0, 0.05, 1, 0.93])
    
    fig.suptitle(f'{metric.replace("_", " ").title()} Across Environments', fontsize=18, fontweight='bold')
    safe_metric_name = metric.replace('/', '_').replace(' ', '_')
    filename = f"{safe_metric_name}_across_envs.pdf"
    filepath = save_dir / filename
    plt.savefig(filepath, format='pdf', bbox_inches='tight')
    plt.close()
    print(f"Saved multi-environment plot: {filename}")

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
    
    # Get unique maps and arrange them in the desired order
    maps = df['map'].unique()
    
    # Define the desired order for subplot arrangement
    desired_order = [
        '15_15_simple_warehouse',        # first
        '50_55_simple_warehouse',        # second  
        '50_55_long_shelves',            # third
        '50_55_open_space_warehouse_bottom'  # fourth
    ]
    
    # Reorder maps according to desired arrangement, keeping any additional maps at the end
    ordered_maps = []
    for map_name in desired_order:
        if map_name in maps:
            ordered_maps.append(map_name)
    
    # Add any maps not in the desired order list
    for map_name in maps:
        if map_name not in desired_order:
            ordered_maps.append(map_name)
    
    maps = ordered_maps
    print(f"Maps to process (in order): {maps}")
    
    # Print color assignments
    print(f"\nModel color assignments:")
    for model in sorted(df['model'].unique()):
        print(f"  {model}: {MODEL_COLORS[model]}")
    
    # Create plots for each metric and map combination
    total_plots = len(metrics) * len(maps)
    plot_count = 0
    
    print(f"\nCreating {total_plots} individual plots...")
    
    for map_name in maps:
        print(f"\nProcessing map: {map_name}")
        
        for metric in metrics:
            plot_count += 1
            print(f"  [{plot_count}/{total_plots}] Creating plot for {metric}")
            
            try:
                create_plot(df, metric, map_name, save_dir)
            except Exception as e:
                print(f"    Error creating plot for {metric} on {map_name}: {e}")
    
    # Create multi-environment plots for each metric
    print(f"\nCreating {len(metrics)} multi-environment plots...")
    
    for i, metric in enumerate(metrics):
        print(f"[{i+1}/{len(metrics)}] Creating multi-environment plot for {metric}")
        try:
            plot_metric_across_environments(df, metric, maps, save_dir)
        except Exception as e:
            print(f"    Error creating multi-environment plot for {metric}: {e}")
    
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
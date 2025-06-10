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
    'magat_pathplanning': '#17becf',  # Cyan
}

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
    
    # For success rate, keep all data points (including 0s)
    # For other metrics, filter out 0 values
    if 'success' not in metric.lower():
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
    
    # Create color palette for the models in this dataset
    unique_models = sorted(map_data['model'].unique())
    colors = [MODEL_COLORS.get(model, '#000000') for model in unique_models]  # Default to black if model not found
    model_palette = dict(zip(unique_models, colors))
    
    # Create main plot with sidebar layout
    fig = plt.figure(figsize=(16, 8))
    
    # Create main plot (takes up left 75% of the figure)
    ax1 = plt.subplot2grid((1, 10), (0, 0), colspan=7)
    
    # Sort the data by agents to ensure proper ordering
    map_data = map_data.sort_values('agents')
    
    # Convert agents to string to ensure categorical treatment
    map_data['agents_str'] = map_data['agents'].astype(str)
    
    # Get unique agent counts for categorical x-axis
    unique_agents = sorted(map_data['agents'].unique())
    unique_agents_str = [str(x) for x in unique_agents]
    
    # Create the main metric plot (line plot) without legend
    sns.lineplot(data=map_data, x='agents_str', y=metric, hue='model', marker='o', 
                linewidth=2, markersize=8, ax=ax1, palette=model_palette, legend=False)
    
    # Set x-axis to show all agent counts as categorical with proper ordering
    ax1.set_xticks(range(len(unique_agents_str)))
    ax1.set_xticklabels(unique_agents_str)
    
    # Customize the main plot
    ax1.set_title(f'{metric.replace("_", " ").title()} vs Number of Agents\n{map_name.replace("_", " ").title()}', 
                  fontsize=16, fontweight='bold')
    ax1.set_xlabel('Number of Agents', fontsize=14)
    ax1.set_ylabel(metric.replace("_", " ").title(), fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Create legend handles
    legend_handles = []
    for model in sorted(map_data['model'].unique()):
        from matplotlib.lines import Line2D
        line = Line2D([0], [0], color=model_palette[model], linewidth=2, marker='o', markersize=8, label=model)
        legend_handles.append(line)
    
    # Position legend in right sidebar
    legend = fig.legend(handles=legend_handles, loc='center', bbox_to_anchor=(0.85, 0.6), 
                       fontsize=11, title='Model', title_fontsize=12, frameon=True, 
                       fancybox=True, shadow=True)
    
    # Create the std overlay plot if data exists
    if has_std:
        # Filter out zero std values for better visualization
        std_data = map_data[map_data[std_metric] > 0] if 'success' not in metric.lower() else map_data
        
        if not std_data.empty:
            # Determine overlay position based on metric type
            if 'success' in metric.lower():
                # Top right for success rate (values typically decrease with more agents)
                overlay_pos = [0.52, 0.65, 0.2, 0.15]  # [left, bottom, width, height] - much smaller
            else:
                # Top left for other metrics (values typically increase with more agents)
                overlay_pos = [0.15, 0.65, 0.2, 0.15]  # [left, bottom, width, height] - much smaller
            
            # Create overlay axes on the main plot
            ax2 = fig.add_axes(overlay_pos)
            
            # Create bar plot in overlay
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
            
            # Customize the overlay plot - remove axis labels
            ax2.set_title('Std Dev', fontsize=9, fontweight='bold')
            ax2.set_xlabel('')  # Remove x-axis label
            ax2.set_ylabel('')  # Remove y-axis label
            ax2.tick_params(axis='both', which='major', labelsize=7)
            ax2.grid(True, alpha=0.3)
            
            # Remove legend from overlay (main legend is in sidebar)
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
    
    # Get metrics to plot
    metrics = get_metrics_to_plot(df)
    print(f"\nMetrics to plot: {metrics}")
    
    # Get unique maps
    maps = df['map'].unique()
    print(f"Maps to process: {maps}")
    
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
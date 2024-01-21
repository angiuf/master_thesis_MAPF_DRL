import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from math import ceil
import numpy as np

colors = ['r', 'g', 'b', 'orange', 'y', 'm', 'c', 'k', 'w']

def get_coordinates(n):
    coordinates = []
    for i in range(ceil(n/2)):
        for j in range(ceil(n/2)):
            coordinates.append((i, j))
    return coordinates


def plot_models(model_names, features, map_name):
    n_agents = features[0]
    features = features[1:]
    num_features = len(features)
    fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
    coords = get_coordinates(num_features)

    print(coords)
    if num_features == 1:
        ax = [ax]
    for i, feature in enumerate(features):
        for j, model_name in enumerate(model_names):
            file_path = os.path.join(map_name, model_name, f"{model_name}.csv")
            print(file_path)
            if os.path.exists(file_path):
                print(file_path)
                df = pd.read_csv(file_path)
                
                # check if feature exists
                if feature not in df.columns:
                    print(f"Feature {feature} not found in {model_name}")
                    continue

                x_ticks = np.arange(len(df[n_agents]))
                ax[coords[i]].plot(x_ticks, df[feature], 'o-', label=model_name, color=colors[j], markersize=3)
                # ax[coords[i]].plot(df[n_agents], df[feature], 'o-', label=model_name, color=colors[j])
                x_labels = [0] + df[n_agents].tolist()
                print(x_labels)
                ax[coords[i]].set_xticklabels(x_labels)
        ax[coords[i]].set_xlabel(n_agents)
        ax[coords[i]].set_ylabel(feature)

        # ax[coords[i]].set_xticks(x_ticks)

        ax[coords[i]].set_title(f"{feature} Comparison")
        ax[coords[i]].legend()
    plt.show()

if __name__ == "__main__":
    map_name = "50_55_simple_warehouse"
    # models = ['DCC', 'PRIMAL_256', 'SCRIMP', 'AB-MAPPER', 'CBS', 'ODrMstar']
    models = ['DCC', 'CBS', 'AB-MAPPER', 'PRIMAL', 'SCRIMP', 'ODrMstar']
    features = ['n_agents', 'success_rate', 'episode_length', 'total_step', 'collision_rate', 'avg_step', 'max_step']
    plot_models(models, features, map_name)
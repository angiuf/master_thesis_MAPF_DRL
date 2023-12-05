import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('SCRIMP.csv')

# Create a figure with 3 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot the data for success_rate
axs[0][0].plot(df['n_agents'], df['success_rate'], 'ro-')
axs[0][0].set_ylabel('success_rate')
axs[0][0].set_ylim(90, 100)
axs[0][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for episode_length
axs[0][1].plot(df['n_agents'], df['episode_length'], 'bo-')
axs[0][1].fill_between(df['n_agents'], df['episode_length'] - df['el_std'], df['episode_length'] + df['el_std'], alpha=0.2, color='blue')
axs[0][1].set_ylabel('episode_length')
axs[0][1].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for maximum_number_of_goals_reached
axs[1][0].plot(df['n_agents'], df['maximum_number_of_goals_reached'], 'go-')
axs[1][0].fill_between(df['n_agents'], df['maximum_number_of_goals_reached'] - df['mr_std'], df['maximum_number_of_goals_reached'] + df['mr_std'], alpha=0.2, color='green')
axs[1][0].set_ylabel('maximum_number_of_goals_reached')
axs[1][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for collision_ratio_static_obstacle
axs[1][1].plot(df['n_agents'], df['collision_ratio_static_obstacle'], 'yo-')
axs[1][1].fill_between(df['n_agents'], df['collision_ratio_static_obstacle'] - df['co_std'], df['collision_ratio_static_obstacle'] + df['co_std'], alpha=0.2, color='yellow')
axs[1][1].set_ylabel('collision_ratio_static_obstacle')
axs[1][1].set_xticks([4, 8, 12, 16, 20, 22])

# Set the x-axis label
fig.text(0.5, 0.04, 'n_agents', ha='center')

# Show the plot
plt.show()

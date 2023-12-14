import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('SCRIMP.csv')

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 2, figsize=(12, 8))

# Plot the data for success_rate
axs[0][0].plot(df['n_agents'], df['success_rate'], 'ro-')
axs[0][0].set_ylabel('success_rate')
axs[0][0].set_ylim(90, 100)
axs[0][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for episode_length
axs[0][1].plot(df['n_agents'], df['episode_length'], 'bo-')
axs[0][1].fill_between(df['n_agents'], df['episode_length'] - df['episode_length_std'], df['episode_length'] + df['episode_length_std'], alpha=0.2, color='blue')
axs[0][1].set_ylabel('episode_length')
axs[0][1].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for max_goals
axs[1][0].plot(df['n_agents'], df['max_goals'], 'go-')
axs[1][0].fill_between(df['n_agents'], df['max_goals'] - df['max_goals_std'], df['max_goals'] + df['max_goals_std'], alpha=0.2, color='green')
axs[1][0].set_ylabel('max_goals')
axs[1][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for collision_rate
axs[1][1].plot(df['n_agents'], df['collision_rate'], 'yo-')
# axs[1][1].fill_between(df['n_agents'], df['collision_rate'] - df['co_std'], df['collision_rate'] + df['co_std'], alpha=0.2, color='yellow')
axs[1][1].set_ylabel('collision_rate')
axs[1][1].set_xticks([4, 8, 12, 16, 20, 22])

axs[2][0].plot(df['n_agents'], df['total_step'], 'mo-')
axs[2][0].fill_between(df['n_agents'], df['total_step'] - df['total_step_std'], df['total_step'] + df['total_step_std'], alpha=0.2, color='magenta')
axs[2][0].set_ylabel('total_step')
axs[2][0].set_xticks([4, 8, 12, 16, 20, 22])

axs[2][1].plot(df['n_agents'], df['avg_step'], 'co-')
axs[2][1].fill_between(df['n_agents'], df['avg_step'] - df['avg_step_std'], df['avg_step'] + df['avg_step_std'], alpha=0.2, color='cyan')
axs[2][1].set_ylabel('avg_step')
axs[2][1].set_xticks([4, 8, 12, 16, 20, 22])

# Set the x-axis label
fig.text(0.5, 0.04, 'n_agents', ha='center')

# Show the plot
plt.show()

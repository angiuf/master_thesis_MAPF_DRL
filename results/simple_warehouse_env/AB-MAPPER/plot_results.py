import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('AB-MAPPER.csv')

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 2, figsize=(18, 8))

# Plot the data for success_rate
axs[0][0].plot(df['n_agents'], df['success_rate'], 'ro-')
axs[0][0].set_ylabel('success_rate')
axs[0][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for episode_length
axs[0][1].plot(df['n_agents'], df['episode_length'], 'go-')
axs[0][1].set_ylabel('episode_length')
axs[0][1].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for average_time
axs[1][0].plot(df['n_agents'], df['avg_step'], 'bo-')
axs[1][0].set_ylabel('avg_step')
axs[1][0].set_xticks([4, 8, 12, 16, 20, 22])


# Plot the data for total_step
axs[1][1].plot(df['n_agents'], df['total_step'], 'yo-')
axs[1][1].set_ylabel('total_step')
axs[1][1].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for collision_rate
axs[2][0].plot(df['n_agents'], df['collision_rate'], 'mo-')
axs[2][0].set_ylabel('collision_rate')
axs[2][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for extra_time
axs[2][1].plot(df['n_agents'], df['extra_time'], 'co-')
axs[2][1].set_ylabel('extra_time')
axs[2][1].set_xticks([4, 8, 12, 16, 20, 22])

# Set the x-axis label
fig.text(0.5, 0.04, 'n_agents', ha='center')

# Show the plot
plt.show()

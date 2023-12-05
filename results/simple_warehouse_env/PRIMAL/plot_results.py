import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('PRIMAL_256_step.csv')

# Create a figure with 3 subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Plot the data for success_rate
axs[0][0].plot(df['n_agents'], df['success_rate'], 'ro-')
axs[0][0].set_ylabel('success_rate')
axs[0][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for average_time
axs[0][1].plot(df['n_agents'], df['average_time'], 'bo-')
axs[0][1].set_ylabel('average_time')
axs[0][1].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for episode_length
axs[1][0].plot(df['n_agents'], df['episode_length'], 'go-')
axs[1][0].set_ylabel('episode_length')
axs[1][0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for crash_rate
axs[1][1].plot(df['n_agents'], df['crash_rate'], 'yo-')
axs[1][1].set_ylabel('crash_rate')
axs[1][1].set_ylim(0, 5)
axs[1][1].set_xticks([4, 8, 12, 16, 20, 22])

# Set the x-axis label
fig.text(0.5, 0.04, 'n_agents', ha='center')

# Show the plot
plt.show()

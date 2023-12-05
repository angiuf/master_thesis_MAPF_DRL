import pandas as pd
import matplotlib.pyplot as plt

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('DCC.csv')

# Create a figure with 3 subplots
fig, axs = plt.subplots(3, 1, figsize=(6, 8))

# Plot the data for success_rate
axs[0].plot(df['n_agents'], df['success_rate'], 'ro-')
axs[0].set_ylabel('success_rate')
axs[0].set_ylim(90, 100)
axs[0].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for average_step
axs[1].plot(df['n_agents'], df['average_step'], 'bo-')
axs[1].set_ylabel('average_step')
axs[1].set_xticks([4, 8, 12, 16, 20, 22])

# Plot the data for communication_times
axs[2].plot(df['n_agents'], df['communication_times'], 'go-')
axs[2].set_ylabel('communication_times')
axs[2].set_xticks([4, 8, 12, 16, 20, 22])

# Set the x-axis label
fig.text(0.5, 0.04, 'n_agents', ha='center')

# Show the plot
plt.show()

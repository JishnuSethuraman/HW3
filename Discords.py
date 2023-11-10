import numpy as np
import stumpy
import matplotlib.pyplot as plt
from SyntheticDatasetGeneration import generate_time_series, motif_length
import pandas as pd

# Load the CSV file
df = pd.read_csv('gasoline.csv', header=None)

# Extract the time series
time_series = df.iloc[:, 0].values

window_size = 50 

# Compute the matrix profile
matrix_profile = stumpy.stump(time_series, m=window_size)

# Find the index of the highest matrix profile value
discord_index = np.argmax(matrix_profile[:, 0])

# Extract the discord subsequence from the time series using the discord_index
discord = time_series[discord_index:discord_index+window_size]

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot the time series and highlight the discord
ax1.plot(time_series)
ax1.set_title('Gas Price with Discord Highlighted')
ax1.set_ylabel('Value')
ax1.axvspan(discord_index, discord_index+window_size, color='red', alpha=0.3)

# Plot the matrix profile
ax2.plot(matrix_profile[:, 0])
ax2.set_title('Matrix Profile')
ax2.set_xlabel('Time')
ax2.set_ylabel('Matrix Profile Value')
ax2.axvline(x=discord_index, color='red', linestyle='dashed')

plt.tight_layout()
plt.show()

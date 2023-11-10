import pandas as pd
import numpy as np
import stumpy
import matplotlib.pyplot as plt
from SyntheticDatasetGeneration import generate_time_series, motif_length

# Load the CSV file into a DataFrame
df = pd.read_csv('Stranger_Things_Ratings.csv')

# Extract the index and rating columns
index_rating_matrix = df[' Rating'].values

# Print the matrix
print(index_rating_matrix)

matrix_profile = stumpy.stump(index_rating_matrix, m=3)

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot the time series
ax1.plot(index_rating_matrix)
ax1.set_title('Time Series')
ax1.set_ylabel('Value')

# Plot the matrix profile
ax2.plot(matrix_profile[:, 0])
ax2.set_title('Matrix Profile')
ax2.set_xlabel('Time')
ax2.set_ylabel('Matrix Profile Value')

plt.tight_layout()
plt.show()
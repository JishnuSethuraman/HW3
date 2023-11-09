import numpy as np
import stumpy
import matplotlib.pyplot as plt
from SyntheticDatasetGeneration import generate_time_series, motif_length

window_size = motif_length

time_series = generate_time_series()

matrix_profile = stumpy.stump(time_series, m=window_size)

# Plo
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 6))

# Plot the time series
ax1.plot(time_series)
ax1.set_title('Time Series')
ax1.set_ylabel('Value')

# Plot the matrix profile
ax2.plot(matrix_profile[:, 0])
ax2.set_title('Matrix Profile')
ax2.set_xlabel('Time')
ax2.set_ylabel('Matrix Profile Value')

plt.tight_layout()
plt.show()
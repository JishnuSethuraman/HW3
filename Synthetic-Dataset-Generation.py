import numpy as np
import matplotlib.pyplot as plt

#parameters
time_series_length = 500
motif = np.array([0, 2, 3, 2, 0, -2, -3, -2, 0, 2, 3, 2, 0, -2, -3, -2, 0])
motif_length = len(motif)
noise_level = 0.2
num_motifs = 3

#sine function for base time series
time_series = np.sin(np.linspace(0, 10*np.pi, time_series_length))

#inject motifs at random intervals
for _ in range(num_motifs):
    start_index = np.random.randint(0, time_series_length - motif_length)
    time_series[start_index:start_index + motif_length] += 1.4*motif

#noise
time_series += np.random.normal(scale=noise_level, size=time_series_length)

plt.plot(time_series)
plt.title('Synthetic Time-Series with Embedded Motif')
plt.xlabel('Time')
plt.ylabel('Value')
plt.show()
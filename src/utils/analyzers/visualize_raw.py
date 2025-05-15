from scipy.io import loadmat

file_path = r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\raw\S1_A1_E1.mat"
mat_data = loadmat(file_path)

emg = mat_data['emg']
restimulus = mat_data['restimulus'].flatten()

relevant_window = []

for i in range(len(restimulus)):
    if restimulus[i] != 0:
        relevant_window.append(emg[i])
        if restimulus[i + 1] == 0:
            break

# print(relevant_window)

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

relevant_window = np.array(relevant_window)  # Shape: (samples, channels)
n_samples, n_channels = relevant_window.shape
time = np.arange(n_samples)

# 3D plot
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(111, projection='3d')

for ch in range(n_channels):
    y = np.ones_like(time) * ch  # Stack by channel
    z = relevant_window[:, ch]   # EMG amplitude for channel
    ax.plot(time, y, z, label=f'Ch {ch+1}')

# Labels and formatting
ax.set_xlabel('Time')
ax.set_ylabel('Channel')
ax.set_zlabel('Amplitude')
ax.set_title('Stacked 3D EMG Plot (Relevant Window)')

plt.tight_layout()
plt.show()

# Plot emg[0:441] (rest)
compare_window = emg[0:441]
t = np.arange(compare_window.shape[0])
n_channels = compare_window.shape[1]

fig2 = plt.figure(figsize=(12, 6))
ax2 = fig2.add_subplot(111, projection='3d')

for ch in range(n_channels):
    ax2.plot(t, np.ones_like(t) * ch, compare_window[:, ch], label=f'Ch {ch+1}')

ax2.set_xlabel('Time')
ax2.set_ylabel('Channel')
ax2.set_zlabel('Amplitude')
ax2.set_title('EMG Window: emg[0:441]')

plt.tight_layout()
plt.show()

# import numpy as np, matplotlib.pyplot as plt
# d = np.load(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed\s1.npz")
# X, y = d["X"], d["y"]
# print(X.shape, y.shape)          # e.g. (37420, 10, 20) (37420,)
# plt.plot(X[0][0])                # first channel of first window
# plt.title(f"Label {y[0]}")
# plt.show()
#

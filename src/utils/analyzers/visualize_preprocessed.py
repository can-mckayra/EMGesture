import numpy as np
import matplotlib.pyplot as plt

d = np.load(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed\s1.npz")
emg, label = d["X"], d["y"]    # emg: (Window, Channel, Time)

n_windows, n_channels, n_samples = emg.shape
win_idx = 28    # window to visualize

time = np.arange(n_samples)    # 0 … 19 if T = 20
fig = plt.figure(figsize=(12, 6))
ax  = fig.add_subplot(111, projection="3d")

for ch in range(n_channels):
    y = np.full_like(time, ch)    # constant “row” for this channel
    z = emg[win_idx, ch, :]    # amplitude of that channel in this window
    ax.plot(time, y, z, label=f"Ch {ch+1}")

ax.set_xlabel("Sample # within window")
ax.set_ylabel("Channel")
ax.set_zlabel("Amplitude")
ax.set_title(f"Stacked 3-D EMG plot — window {win_idx} (label {label[win_idx]})")

plt.tight_layout()
plt.show()

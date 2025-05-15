import numpy as np
import matplotlib.pyplot as plt

d = np.load(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed\s1.npz")
emg, label = d["X"], d["y"] # emg: (Windows, Channels, Time)

print(emg.shape)
print(label.shape)

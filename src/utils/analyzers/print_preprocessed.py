import numpy as np
import matplotlib.pyplot as plt

d = np.load(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed\s3.npz")
emg, label = d["X"], d["y"] # emg: (Windows, Channels, Time)

print(emg.shape)
print(label.shape)

# print(emg[1])
# print(label)

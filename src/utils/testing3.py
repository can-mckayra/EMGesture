# import numpy as np
#
# with np.load(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed\s3.npz") as npz:
#     print("Keys in the archive:", npz.files)      # or:  list(npz.keys())
#     for k in npz.files:
#         print(f"{k:>8}  shape = {npz[k].shape}  dtype = {npz[k].dtype}")

import numpy as np

d = np.load(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed\test.npz")
emg, label = d["emg"], d["label"]  # emg: (Windows, Channels, Time)

print(emg.shape)
print(label.shape)

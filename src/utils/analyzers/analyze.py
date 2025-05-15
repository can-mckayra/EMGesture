from scipy.io import loadmat

file_path = r"/raw/S1_A1_E1.mat"
mat_data = loadmat(file_path)

print(mat_data.keys())

emg = mat_data['emg']   # EMG signal
restimulus = mat_data['restimulus'].flatten()   # Labels
rerepetition = mat_data['rerepetition'].flatten()   # Trial numbers

print("EMG shape:", emg.shape)
print("restimulus shape:", restimulus.shape)
print("rerepetition shape:", rerepetition.shape)

repetitions = []
count = 0
this = rerepetition
for i in range(len(this) - 1):
    if this[i] != 0:
        count += 1
        if this[i + 1] == 0:
            repetitions.append(count)
            count = 0

print(repetitions)
print(f"Average: {sum(repetitions) / len(repetitions):.0f}")
print(emg[0])





# import numpy as np
#
# data = np.load("C:/Users/HP GAME/PycharmProjects/EMGesture/data/processed/s1.npz")
#
# X = data['X']
# y = data['y']
# print("X shape:", X.shape)
# print("y shape:", y.shape)
# print("X dtype:", X.dtype)
# print("y dtype:", y.dtype)
# print("First window of EMG data:\n", X[0])
# print("First label:", y[0])
#

from scipy.io import loadmat

file_path = r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\raw\S1_A1_E1.mat"
mat_data = loadmat(file_path)

emg = mat_data['emg']
restimulus = mat_data['restimulus'].flatten()

print(emg.shape)
print(restimulus.shape)

print(emg)
print(restimulus)

import os, glob, math, numpy as np
from scipy.io import loadmat

RAW_DIR = r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\raw"
OUT_DIR = r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\preprocessed"
WIN_SIZE_SAMP = 15  # 150 ms @ 100 Hz
N_CHANNELS = 10
REST_LABEL = 0

#os.makedirs(OUT_DIR, exist_ok=True)

# mat_file_paths = glob.glob(os.path.join(RAW_DIR, "*.mat"))
#
# for file in mat_file_paths:
#     print(file)

mat_file_paths = []

for s in range(1, 28):
    for e in range(1, 4):
        mat_file_paths.append(os.path.join(RAW_DIR, f"S{s}_A1_E{e}.mat"))



for file in mat_file_paths:
    print(file)

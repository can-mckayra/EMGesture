import os, glob, math, numpy as np
from scipy.io import loadmat

RAW_DIR       = "data/raw"
OUT_DIR       = "data/processed"
WIN_SIZE_SAMP = 50  # 500 ms @ 100 Hz
STRIDE_SAMP   = 25  # 50 % overlap
N_CHANNELS    = 10
REST_LABEL    = 0

os.makedirs(OUT_DIR, exist_ok=True)

def zscore(x, eps=1e-8):
    return (x - x.mean(axis=0)) / (x.std(axis=0) + eps)

def segment(emg, labels):
    # Return windows (Channels, T,me) and labels.
    X, y = [], []
    for start in range(0, len(emg) - WIN_SIZE_SAMP + 1, STRIDE_SAMP):
        end = start + WIN_SIZE_SAMP
        seg_lab = labels[start:end]
        if np.all(seg_lab == seg_lab[0]):
            X.append(emg[start:end].T)
            y.append(int(seg_lab[0]))
    return np.array(X).astype(np.float32), np.array(y, dtype=np.int64)

def process_subject(mat_paths, out_path):
    windows, labels = [], []
    for mp in mat_paths:
        m = loadmat(mp, squeeze_me=True)
        emg = m["emg"].astype(np.float32)            # (N, 10)
        signal = zscore(emg)                            # per-exercise norm
        restim = m["restimulus"].astype(np.int16)       # (N,)
        X, y = segment(signal, restim)
        windows.append(X)
        labels.append(y)
    X_all = np.concatenate(windows)
    y_all = np.concatenate(labels)
    np.savez_compressed(out_path, X=X_all, y=y_all)
    print(f"Saved {out_path}: {X_all.shape[0]} windows")

def main():

    mat_file_paths = glob.glob(os.path.join(RAW_DIR, "*.mat"))

    subject_names = set()
    for path in mat_file_paths:
        filename = os.path.basename(path)
        subject_prefix = filename.split("_")[0]
        subject_names.add(subject_prefix.lower())

    subjects = sorted(subject_names)

    for subj in subjects:
        paths = glob.glob(f"{RAW_DIR}/{subj}*.mat")
        process_subject(paths, f"{OUT_DIR}/{subj}.npz")

if __name__ == "__main__":
    main()

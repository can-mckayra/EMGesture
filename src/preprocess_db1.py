"""
Pre-process raw Ninapro DB-1 sEMG for the Atzori CNN.
Produces three NumPy arrays:
  windows.npy  (N, 1, 30, 10)
  labels.npy   (N,)
  reps.npy     (N,)
"""

import numpy as np
import scipy.io as sio
from scipy.signal import resample_poly, butter, filtfilt
from pathlib import Path
from tqdm import tqdm

# ---------------------------------------------------------------------
# 1. helpers
# ---------------------------------------------------------------------
def butter_lowpass(data, cutoff, fs, order=4):
    """4-th-order zero-phase Butterworth LPF."""
    b, a = butter(order, cutoff / (fs / 2), btype="low")
    return filtfilt(b, a, data, axis=0)

def majority_vote(vec):
    """Return the most frequent value (labels are piece-wise constant)."""
    return np.bincount(vec).argmax()

# ---------------------------------------------------------------------
# 2. per-subject processing
# ---------------------------------------------------------------------
def preprocess_subject(mat_path, win_len=30, step=30,
                       fs_orig=100, fs_target=200, lp_cut=1.0):
    """Return windows, labels, repetition-id for *one* .mat file."""
    m = sio.loadmat(mat_path.as_posix(), squeeze_me=True)

    emg      = m["s_emg"].astype(np.float32)        # (T, 10)
    stimulus = m["stimulus"].astype(np.int16)       # (T,)
    reps     = m["repetition"].astype(np.int8)      # (T,)

    # -----------------------------------------------------------------
    # upsample RMS signal 100 → 200 Hz  (factor = 2)
    # -----------------------------------------------------------------
    emg = resample_poly(emg, fs_target, fs_orig, axis=0)
    stimulus = np.repeat(stimulus, 2)
    reps     = np.repeat(reps, 2)

    # -----------------------------------------------------------------
    # 1 Hz low-pass on every channel
    # -----------------------------------------------------------------
    emg = butter_lowpass(emg, lp_cut, fs_target, order=4)

    # -----------------------------------------------------------------
    # sliding-window segmentation
    # -----------------------------------------------------------------
    n_samples = emg.shape[0]
    win, lab, repid = [], [], []
    for start in range(0, n_samples - win_len + 1, step):
        stop = start + win_len
        win.append(emg[start:stop][None, ...])          # (1, 30, 10)
        lab.append(majority_vote(stimulus[start:stop])) # int
        repid.append(reps[start])                       # repetition id

    return np.stack(win), np.array(lab), np.array(repid)

# ---------------------------------------------------------------------
# 3. iterate over every subject folder / .mat file
# ---------------------------------------------------------------------
def main(db_root="DB1_raw", out_dir="preproc_out"):
    db_root = Path(db_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True)

    all_w, all_y, all_r = [], [], []
    mat_files = sorted(db_root.glob("**/s[0-9]*.mat"))

    for f in tqdm(mat_files, desc="subjects"):
        w, y, r = preprocess_subject(f)
        all_w.append(w); all_y.append(y); all_r.append(r)

    X = np.concatenate(all_w).astype(np.float32)              # (N, 1, 30, 10)
    y = np.concatenate(all_y).astype(np.int64)
    reps = np.concatenate(all_r)

    np.save(out_dir / "windows.npy", X)
    np.save(out_dir / "labels.npy",  y)
    np.save(out_dir / "reps.npy",    reps)
    print(f"saved: {X.shape[0]} windows → {out_dir}")

if __name__ == "__main__":
    main()

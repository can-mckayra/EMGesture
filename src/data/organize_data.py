# split_merge_npz.py
from pathlib import Path
import numpy as np
import re

ROOT = Path(r"/processed")
KEYS = dict(emg="X", label="y")
TRAIN_RANGE = range(3, 25)
TEST_RANGE  = range(25, 28)
OUT_TRAIN = ROOT / "train.npz"
OUT_TEST  = ROOT / "test.npz"

def collect_subject_files(root: Path):
    """Return {subject_idx: Path} for every .npz in the folder."""
    patt = re.compile(r"s(\d+)\.npz", flags=re.I)
    mapping = {}
    for fp in root.glob("s*.npz"):
        m = patt.fullmatch(fp.name)
        if m:
            mapping[int(m.group(1))] = fp
    return mapping

def merge_subset(file_paths):
    """Load each .npz and concatenate `emg` and `label` along axis 0."""
    emg_list, label_list = [], []

    for fp in sorted(file_paths):
        with np.load(fp) as npz:
            emg = npz[KEYS["emg"]].astype(np.float32)
            label = npz[KEYS["label"]].astype(np.uint8)

            # gather
            emg_list.append(emg)
            label_list.append(label)

    emg_all   = np.concatenate(emg_list,   axis=0)
    label_all = np.concatenate(label_list, axis=0)
    return dict(emg=emg_all, label=label_all)

def main():
    subj_files = collect_subject_files(ROOT)

    train_files = [subj_files[i] for i in TRAIN_RANGE if i in subj_files]
    test_files  = [subj_files[i] for i in TEST_RANGE  if i in subj_files]

    if len(train_files) != len(TRAIN_RANGE):
        missing = set(TRAIN_RANGE) - subj_files.keys()
        raise FileNotFoundError(f"Missing train subjects: {sorted(missing)}")
    if len(test_files) != len(TEST_RANGE):
        missing = set(TEST_RANGE) - subj_files.keys()
        raise FileNotFoundError(f"Missing test subjects: {sorted(missing)}")

    train_data = merge_subset(train_files)
    test_data  = merge_subset(test_files)

    np.savez_compressed(OUT_TRAIN, **train_data)
    np.savez_compressed(OUT_TEST,  **test_data)

    print(f"Saved {OUT_TRAIN}  ->  {train_data['emg'].shape[0]} samples")
    print(f"Saved {OUT_TEST}  ->  {test_data['emg'].shape[0]} samples")

if __name__ == "__main__":
    main()

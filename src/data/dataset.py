from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class EMGestureDataset(Dataset):

    """Loads the pre-windowed .npz files as (N, 1, 15, 10) tensors + labels."""

    def __init__(self, npz_path: str | Path):
        z   = np.load(npz_path)
        X   = torch.from_numpy(z["emg"]).float()
        X   = X.permute(0, 2, 1)
        self.X = X.unsqueeze(1)
        self.y = torch.from_numpy(z["label"]).long()

    def __len__(self):
        return self.X.size(0)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

def make_loaders(root: Path, batch_sz: int = 256, val_split: float = 0.10):

    train_ds = EMGestureDataset(root / "train.npz")
    test_ds  = EMGestureDataset(root / "test.npz")

    if val_split:
        n_val   = int(len(train_ds) * val_split)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    else: val_ds = None

    def mk(dl_ds):
        return DataLoader(dl_ds, batch_size=batch_sz, pin_memory=True)

    return mk(train_ds), (mk(val_ds) if val_ds else None), mk(test_ds)

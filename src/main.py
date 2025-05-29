# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# class EMGesture(nn.Module):  # Renamed from SEMGCNNOtto
#     """
#     PyTorch implementation of the CNN architecture described in "Deep Learning with
#     Convolutional Neural Networks Applied to Electromyography Data: A Resource
#     for the Classification of Movements for Prosthetic Hands" by Atzori et al. (2016).
#
#     This version is specifically configured for the "Otto Bock" setup:
#     - num_electrodes = 10
#     - conv4_kernel_h = 5
#     - time_steps = 15 (150ms window at 100Hz)
#
#     The input is expected to be of shape (batch_size, 1, 15, 10).
#     """
#
#     def __init__(self, num_classes):
#         """
#         Initializes the EMGesture model.
#
#         Args:
#             num_classes (int): Number of output classes (hand movements).
#         """
#         super(EMGesture, self).__init__()  # Updated class name here
#
#         self.num_electrodes = 10
#         self.num_classes = num_classes
#         self.conv4_kernel_h = 5
#         self.time_steps = 15  # Expected input height (150ms at 100Hz)
#
#         # Common ReLU activation
#         self.relu = nn.ReLU()
#
#         # Block 1: Convolution + ReLU
#         # Input: (N, 1, 15, 10)
#         # Kernel: "a row of the length of number of electrodes" -> (1, 10)
#         # Output: (N, 32, 15, 1)
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, self.num_electrodes), padding=0)
#
#         # Block 2: Convolution + ReLU + Pooling
#         # Input: (N, 32, 15, 1)
#         # Conv kernel: 3x3 in paper, interpreted as (3,1) due to input width 1
#         # Output: (N, 32, 15, 1) after conv (due to padding)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1),
#                                padding=(1, 0))  # Padding to keep time_steps same
#         # Pool kernel: 3x3 in paper, interpreted as (3,1)
#         # Stride (2,1) to halve the time dimension
#         self.pool1 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))
#         # Output after pool1: (N, 32, 7, 1) for time_steps=15 input to pool
#
#         # Block 3: Convolution + ReLU + Pooling
#         # Input: (N, 32, 7, 1)
#         # Conv kernel: 5x5 in paper, interpreted as (5,1)
#         # Output: (N, 64, 7, 1) after conv (due to padding)
#         self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1),
#                                padding=(2, 0))  # Padding to keep time_steps same
#         # Pool kernel: 3x3 in paper, interpreted as (3,1)
#         # Stride (2,1) to halve the time dimension
#         self.pool2 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))
#         # Output after pool2: (N, 64, 3, 1) for time_steps=7 input to pool
#
#         # Block 4: Convolution + ReLU
#         # Input: (N, 64, 3, 1)
#         # Conv kernel: (5, 1) as conv4_kernel_h is 5
#         # Padding to keep time_steps same: ((5 - 1) // 2, 0) = (2,0)
#         self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(self.conv4_kernel_h, 1),
#                                padding=((self.conv4_kernel_h - 1) // 2, 0))
#         # Output: (N, 64, 3, 1) # Height remains same due to padding
#
#         # Block 5: Classifier (using Adaptive Pooling before 1x1 Conv)
#         # Adaptive Average Pooling to reduce spatial dimensions to (1,1)
#         # Input: (N, 64, 3, 1)
#         # Output: (N, 64, 1, 1)
#         self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
#
#         # Final 1x1 Convolutional layer acting as a classifier
#         # Input: (N, 64, 1, 1)
#         # Output: (N, num_classes, 1, 1)
#         self.conv5_classifier = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=(1, 1))
#
#     def forward(self, x):
#         """
#         Forward pass of the EMGesture model.
#
#         Args:
#             x (torch.Tensor): Input tensor of shape (batch_size, 1, 15, 10).
#
#         Returns:
#             torch.Tensor: Output logits of shape (batch_size, num_classes).
#         """
#         # Verify input shape (optional, for debugging)
#         # assert x.shape[2] == self.time_steps, f"Input time_steps {x.shape[2]} does not match model's expected {self.time_steps}"
#         # assert x.shape[3] == self.num_electrodes, f"Input num_electrodes {x.shape[3]} does not match model's expected {self.num_electrodes}"
#
#         # Block 1
#         x = self.relu(self.conv1(x))
#         # Shape: (N, 32, 15, 1)
#
#         # Block 2
#         x = self.relu(self.conv2(x))  # Shape: (N, 32, 15, 1)
#         x = self.pool1(x)  # Shape: (N, 32, 7, 1)
#         # Calculation for height after pool1: floor((15 - 3)/2 + 1) = floor(12/2 + 1) = 7
#
#         # Block 3
#         x = self.relu(self.conv3(x))  # Shape: (N, 64, 7, 1)
#         x = self.pool2(x)  # Shape: (N, 64, 3, 1)
#         # Calculation for height after pool2: floor((7 - 3)/2 + 1) = floor(4/2 + 1) = 3
#
#         # Block 4
#         x = self.relu(self.conv4(x))  # Shape: (N, 64, 3, 1)
#
#         # Block 5 (Classifier)
#         x = self.adaptive_pool(x)  # Reduces spatial dims to (1,1) -> (N, 64, 1, 1)
#         x = self.conv5_classifier(x)  # Classifier conv -> (N, num_classes, 1, 1)
#
#         # Squeeze to get (batch_size, num_classes) for classification
#         x = x.view(x.size(0), -1)  # More robust squeeze
#
#         return x
#
#
# if __name__ == '__main__':
#     # Example Usage:
#     batch_size = 4
#     num_classes_example = 53  # Example number of movements
#
#     # Expected input dimensions for this Otto-specific model
#     expected_time_steps = 15
#     expected_num_electrodes = 10
#
#     # Instantiate the model
#     model_EMGesture = EMGesture(num_classes=num_classes_example)  # Updated class name here
#
#     print("Model for Otto Bock Configuration (EMGesture):")  # Updated print statement
#     print(f"  Expected input time_steps: {model_EMGesture.time_steps}")
#     print(f"  Expected input num_electrodes: {model_EMGesture.num_electrodes}")
#     print(f"  Conv4 kernel height: {model_EMGesture.conv4_kernel_h}")
#     # print(model_otto) # You can uncomment to see the full model structure
#
#     # Create a dummy input tensor
#     # Shape: (batch_size, in_channels, height, width) -> (batch_size, 1, 15, 10)
#     dummy_input_otto = torch.randn(batch_size, 1, expected_time_steps, expected_num_electrodes)
#
#     # Perform a forward pass
#     output_otto = model_EMGesture(dummy_input_otto)
#     print(f"\nInput shape: {dummy_input_otto.shape}")
#     print(f"Output shape (Otto): {output_otto.shape}")  # Should be (batch_size, num_classes)
#
#     # You can trace the shapes through the network for a single sample if needed:
#     print("\n--- Tracing shapes for EMGesture model with one sample ---")  # Updated print statement
#     sample_input = torch.randn(1, 1, expected_time_steps, expected_num_electrodes)
#     print(f"Initial input: {sample_input.shape}")
#
#     x = model_EMGesture.relu(model_EMGesture.conv1(sample_input))
#     print(f"After Block 1 (conv1 + relu): {x.shape}")
#
#     x_conv2 = model_EMGesture.relu(model_EMGesture.conv2(x))
#     print(f"After Block 2 (conv2 + relu): {x_conv2.shape}")
#     x_pool1 = model_EMGesture.pool1(x_conv2)
#     print(f"After Block 2 (pool1): {x_pool1.shape}")
#
#     x_conv3 = model_EMGesture.relu(model_EMGesture.conv3(x_pool1))
#     print(f"After Block 3 (conv3 + relu): {x_conv3.shape}")
#     x_pool2 = model_EMGesture.pool2(x_conv3)
#     print(f"After Block 3 (pool2): {x_pool2.shape}")
#
#     x_conv4 = model_EMGesture.relu(model_EMGesture.conv4(x_pool2))
#     print(f"After Block 4 (conv4 + relu): {x_conv4.shape}")
#
#     x_adapt_pool = model_EMGesture.adaptive_pool(x_conv4)
#     print(f"After Adaptive Pool: {x_adapt_pool.shape}")
#
#     x_classifier = model_EMGesture.conv5_classifier(x_adapt_pool)
#     print(f"After Classifier Conv: {x_classifier.shape}")
#
#     final_output = x_classifier.view(x_classifier.size(0), -1)
#     print(f"Final Output (logits): {final_output.shape}")
#

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split

# ---------------------------------------------------------------------------#
# 1.  YOUR MODEL (unchanged)
# ---------------------------------------------------------------------------#
class EMGesture(nn.Module):  # Renamed from SEMGCNNOtto
    """
    PyTorch implementation of the CNN architecture described in "Deep Learning with
    Convolutional Neural Networks Applied to Electromyography Data: A Resource
    for the Classification of Movements for Prosthetic Hands" by Atzori et al. (2016).

    This version is specifically configured for the "Otto Bock" setup:
    - num_electrodes = 10
    - conv4_kernel_h = 5
    - time_steps = 15 (150ms window at 100Hz)

    The input is expected to be of shape (batch_size, 1, 15, 10).
    """

    def __init__(self, num_classes):
        """
        Initializes the EMGesture model.

        Args:
            num_classes (int): Number of output classes (hand movements).
        """
        super(EMGesture, self).__init__()  # Updated class name here

        self.num_electrodes = 10
        self.num_classes = num_classes
        self.conv4_kernel_h = 5
        self.time_steps = 15  # Expected input height (150ms at 100Hz)

        # Common ReLU activation
        self.relu = nn.ReLU()

        # Block 1: Convolution + ReLU
        # Input: (N, 1, 15, 10)
        # Kernel: "a row of the length of number of electrodes" -> (1, 10)
        # Output: (N, 32, 15, 1)
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(1, self.num_electrodes), padding=0)

        # Block 2: Convolution + ReLU + Pooling
        # Input: (N, 32, 15, 1)
        # Conv kernel: 3x3 in paper, interpreted as (3,1) due to input width 1
        # Output: (N, 32, 15, 1) after conv (due to padding)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 1),
                               padding=(1, 0))  # Padding to keep time_steps same
        # Pool kernel: 3x3 in paper, interpreted as (3,1)
        # Stride (2,1) to halve the time dimension
        self.pool1 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))
        # Output after pool1: (N, 32, 7, 1) for time_steps=15 input to pool

        # Block 3: Convolution + ReLU + Pooling
        # Input: (N, 32, 7, 1)
        # Conv kernel: 5x5 in paper, interpreted as (5,1)
        # Output: (N, 64, 7, 1) after conv (due to padding)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(5, 1),
                               padding=(2, 0))  # Padding to keep time_steps same
        # Pool kernel: 3x3 in paper, interpreted as (3,1)
        # Stride (2,1) to halve the time dimension
        self.pool2 = nn.AvgPool2d(kernel_size=(3, 1), stride=(2, 1))
        # Output after pool2: (N, 64, 3, 1) for time_steps=7 input to pool

        # Block 4: Convolution + ReLU
        # Input: (N, 64, 3, 1)
        # Conv kernel: (5, 1) as conv4_kernel_h is 5
        # Padding to keep time_steps same: ((5 - 1) // 2, 0) = (2,0)
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(self.conv4_kernel_h, 1),
                               padding=((self.conv4_kernel_h - 1) // 2, 0))
        # Output: (N, 64, 3, 1) # Height remains same due to padding

        # Block 5: Classifier (using Adaptive Pooling before 1x1 Conv)
        # Adaptive Average Pooling to reduce spatial dimensions to (1,1)
        # Input: (N, 64, 3, 1)
        # Output: (N, 64, 1, 1)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Final 1x1 Convolutional layer acting as a classifier
        # Input: (N, 64, 1, 1)
        # Output: (N, num_classes, 1, 1)
        self.conv5_classifier = nn.Conv2d(in_channels=64, out_channels=self.num_classes, kernel_size=(1, 1))

    def forward(self, x):
        """
        Forward pass of the EMGesture model.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 1, 15, 10).

        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes).
        """
        # Verify input shape (optional, for debugging)
        # assert x.shape[2] == self.time_steps, f"Input time_steps {x.shape[2]} does not match model's expected {self.time_steps}"
        # assert x.shape[3] == self.num_electrodes, f"Input num_electrodes {x.shape[3]} does not match model's expected {self.num_electrodes}"

        # Block 1
        x = self.relu(self.conv1(x))
        # Shape: (N, 32, 15, 1)

        # Block 2
        x = self.relu(self.conv2(x))  # Shape: (N, 32, 15, 1)
        x = self.pool1(x)  # Shape: (N, 32, 7, 1)
        # Calculation for height after pool1: floor((15 - 3)/2 + 1) = floor(12/2 + 1) = 7

        # Block 3
        x = self.relu(self.conv3(x))  # Shape: (N, 64, 7, 1)
        x = self.pool2(x)  # Shape: (N, 64, 3, 1)
        # Calculation for height after pool2: floor((7 - 3)/2 + 1) = floor(4/2 + 1) = 3

        # Block 4
        x = self.relu(self.conv4(x))  # Shape: (N, 64, 3, 1)

        # Block 5 (Classifier)
        x = self.adaptive_pool(x)  # Reduces spatial dims to (1,1) -> (N, 64, 1, 1)
        x = self.conv5_classifier(x)  # Classifier conv -> (N, num_classes, 1, 1)

        # Squeeze to get (batch_size, num_classes) for classification
        x = x.view(x.size(0), -1)  # More robust squeeze

        return x


# ---------------------------------------------------------------------------#
# 2.  DATA  →  PyTorch Dataset + loaders
# ---------------------------------------------------------------------------#
class EMGestureDataset(Dataset):
    """Loads one .npz (X, y) into RAM; adds channel dim for Conv2d model."""
    def __init__(self, npz_path: str | Path):
        z = np.load(npz_path)
        X = torch.from_numpy(z["emg"]).float()          # (N, 10, 15)
        y = torch.from_numpy(z["label"]).long()           # (N,)
        self.X = X.unsqueeze(1)                       # (N, 1, 10, 15)
        self.y = y

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def make_loaders(root: Path,
                 batch_sz: int = 256,
                 val_split: float = 0.1,
                 num_workers: int = 0,
                 pin_mem: bool = True):
    train_ds = EMGestureDataset(root / "train.npz")
    test_ds  = EMGestureDataset(root / "test.npz")

    # optional in-train validation split
    if val_split > 0:
        n_val   = int(len(train_ds) * val_split)
        n_train = len(train_ds) - n_val
        train_ds, val_ds = random_split(
            train_ds, [n_train, n_val],
            generator=torch.Generator().manual_seed(42))
    else:
        val_ds = None

    def make_loader(ds, shuffle):
        return DataLoader(ds,
                          batch_size=batch_sz,
                          shuffle=shuffle,
                          num_workers=num_workers,
                          pin_memory=pin_mem)

    tr_loader  = make_loader(train_ds, shuffle=True)
    te_loader  = make_loader(test_ds,  shuffle=False)
    val_loader = make_loader(val_ds,   shuffle=False) if val_ds else None
    return tr_loader, val_loader, te_loader


# ---------------------------------------------------------------------------#
# 3.  TRAIN / EVAL HELPERS
# ---------------------------------------------------------------------------#
@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    n_correct = n_total = 0
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = criterion(logits, yb)
        running_loss += loss.item() * yb.size(0)
        preds = logits.argmax(1)
        n_correct += (preds == yb).sum().item()
        n_total   += yb.size(0)
    mean_loss = running_loss / n_total
    acc       = n_correct / n_total
    return mean_loss, acc


def train_one_epoch(model, loader, optimiser, criterion, device):
    model.train()
    n_correct = n_total = 0
    running_loss = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimiser.zero_grad(set_to_none=True)
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward()
        optimiser.step()

        running_loss += loss.item() * yb.size(0)
        preds = logits.argmax(1)
        n_correct += (preds == yb).sum().item()
        n_total   += yb.size(0)

    mean_loss = running_loss / n_total
    acc       = n_correct / n_total
    return mean_loss, acc


# ---------------------------------------------------------------------------#
# 4.  MAIN SCRIPT
# ---------------------------------------------------------------------------#
if __name__ == "__main__":
    # ---- paths & hyper-params ------------------------------------------------
    DATA_ROOT   = Path(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed")
    NUM_CLASSES = 53
    EPOCHS      = 25
    BATCH_SIZE  = 512
    LR          = 3e-4

    # ---- data loaders --------------------------------------------------------
    train_loader, val_loader, test_loader = make_loaders(
        DATA_ROOT,
        batch_sz=BATCH_SIZE,
        val_split=0.1,
        num_workers=0,
        pin_mem=True,
    )

    # ---- model / loss / optimiser -------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model  = EMGesture(num_classes=NUM_CLASSES).to(device)
    criterion  = nn.CrossEntropyLoss()
    optimiser  = torch.optim.Adam(model.parameters(), lr=LR)

    # ---- training loop -------------------------------------------------------
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, train_loader,
                                          optimiser, criterion, device)

        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader,
                                         criterion, device)
            msg = (f"Epoch {epoch:02d}/{EPOCHS}  "
                   f"train  loss {tr_loss:.4f}  acc {tr_acc:.3%}   "
                   f"val  loss {val_loss:.4f}  acc {val_acc:.3%}   "
                   f"[{time.perf_counter()-t0:.1f}s]")
        else:
            msg = (f"Epoch {epoch:02d}/{EPOCHS}  "
                   f"train loss {tr_loss:.4f}  acc {tr_acc:.3%}   "
                   f"[{time.perf_counter()-t0:.1f}s]")
        print(msg)

    # ---- final test ----------------------------------------------------------
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTEST set  loss {test_loss:.4f}  acc {test_acc:.3%}")

    # ---- (optional) save checkpoint -----------------------------------------
    torch.save(model.state_dict(), "emgesture_otto.pth")
    print("Model saved to emgesture_otto.pth")

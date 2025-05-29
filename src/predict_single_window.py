# predict_single_window.py
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------#
# 1.  model definition – paste EXACTLY the class you trained with
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
# 2.  helper
# ---------------------------------------------------------------------------#
def load_single_window(npz_path: Path, index: int):
    """
    Returns (window_tensor, true_label, idx) ready for Conv2d model.
    `window_tensor` shape → (1, 1, 15, 10)
    """
    z = np.load(npz_path)
    X = z["X"]      # (N, 10, 15)
    y = z["y"]      # (N,)

    if not (0 <= index < X.shape[0]):
        raise IndexError(f"index {index} out of range (0–{X.shape[0]-1})")

    win = X[index]                      # (10, 15)
    win_t = torch.from_numpy(win).float().unsqueeze(0).unsqueeze(0)
    # -> (1, 1, 10, 15)  … but model expects (1, 1, 15, 10)
    win_t = win_t.permute(0, 1, 3, 2)   # (1, 1, 15, 10)
    label = int(y[index])
    return win_t, label, index


def plot_window(win_np, title=None):
    """win_np shape (10, 15) → quick heat-map."""
    plt.imshow(win_np, aspect="auto", origin="lower", cmap="viridis")
    plt.colorbar(label="Amplitude (a.u.)")
    plt.xlabel("Time step")
    plt.ylabel("Electrode")
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------#
# 3.  main
# ---------------------------------------------------------------------------#
def main():
    parser = argparse.ArgumentParser(description="EMGesture single-window predictor")
    parser.add_argument("--split", choices=["train", "test"], default="test",
                        help="Which .npz file to read (default=test)")
    parser.add_argument("--idx", type=int, default=0,
                        help="Window index to inspect (default=0)")
    parser.add_argument("--show", action="store_true",
                        help="Visualize the window as an image")
    args = parser.parse_args([])  # ← remove `[]` to enable CLI parsing outside PyCharm

    # --- paths ----------------------------------------------------------------
    DATA_ROOT = Path(r"/processed")
    NPZ_PATH  = DATA_ROOT / f"{args.split}.npz"
    CKPT_PATH = Path("emgesture_otto.pth")     # adjust if you saved elsewhere

    # --- load window ----------------------------------------------------------
    window, true_label, _ = load_single_window(NPZ_PATH, args.idx)

    # --- model ----------------------------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    NUM_CLASSES = 53
    model = EMGesture(num_classes=NUM_CLASSES).to(device)
    model.load_state_dict(torch.load(CKPT_PATH, map_location=device))
    model.eval()

    # --- predict --------------------------------------------------------------
    with torch.no_grad():
        logits = model(window.to(device))
        probs  = F.softmax(logits, dim=1).cpu().squeeze(0)
        pred   = int(probs.argmax())
        confid = float(probs.max())

    # --- report ---------------------------------------------------------------
    print(f"\nFile : {args.split}.npz")
    print(f"Index: {args.idx}")
    print(f"True label: {true_label}")
    print(f"Pred label: {pred}   (confidence {confid:.2%})")
    print("\nTop-5 probabilities:")
    top5 = torch.topk(probs, k=min(5, NUM_CLASSES))
    for rank, (cls, p) in enumerate(zip(top5.indices, top5.values), 1):
        print(f"  #{rank}: class {int(cls):>3}  prob {float(p):.2%}")

    # --- optional viz ---------------------------------------------------------
    if args.show:
        win_np = window.squeeze().permute(1, 0).numpy()  # back to (10, 15)
        plot_window(win_np, title=f"Window #{args.idx} (true {true_label}, pred {pred})")


if __name__ == "__main__":
    main()

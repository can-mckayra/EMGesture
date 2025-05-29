import torch
import torch.nn as nn
import time
from pathlib import Path

from model import EMGesture
from data.dataset import make_loaders
from utils.train_utils import train_one_epoch, evaluate

DATA_ROOT   = Path(r"C:\Users\HP GAME\PycharmProjects\EMGesture\data\processed")
NUM_CLASSES = 53
EPOCHS      = 15
BATCH_SIZE  = 512
LR          = 3e-4

def main():
    # data
    tr_loader, val_loader, te_loader = make_loaders(DATA_ROOT, batch_sz=BATCH_SIZE, val_split=0.10)

    # model / loss / optimizer
    device   = torch.device("cuda:0")
    model    = EMGesture(NUM_CLASSES).to(device)
    crit     = nn.CrossEntropyLoss()
    optimizer    = torch.optim.Adam(model.parameters(), lr=LR)

    # training loop
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        tr_loss, tr_acc = train_one_epoch(model, tr_loader, optimizer, crit, device)
        if val_loader:
            v_loss, v_acc = evaluate(model, val_loader, crit, device)
            print(f"Ep {epoch:02d}/{EPOCHS}  " f"train {tr_loss:.4f}/{tr_acc:.2%}  " f"val {v_loss:.4f}/{v_acc:.2%}  " f"[{time.perf_counter()-t0:.1f}s]")
        else:
            print(f"Ep {epoch:02d}/{EPOCHS}  " f"train {tr_loss:.4f}/{tr_acc:.2%}  " f"[{time.perf_counter()-t0:.1f}s]")

    # final test
    te_loss, te_acc = evaluate(model, te_loader, crit, device)
    print(f"\nTEST loss {te_loss:.4f} acc {te_acc:.2%}")

    # save
    torch.save(model.state_dict(), "emgesture.pth")
    print("Model saved to emgesture.pth")

if __name__ == "__main__":
    main()

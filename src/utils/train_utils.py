import torch

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss_sum += loss.item() * yb.size(0)
        pred     = logits.argmax(dim=1)
        correct += (pred == yb).sum().item()
        total   += yb.size(0)
    return loss_sum / total, correct / total

def train_one_epoch(model, loader, optim, criterion, device):
    model.train()
    loss_sum = correct = total = 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optim.zero_grad(set_to_none=True)
        logits = model(xb)
        loss   = criterion(logits, yb)
        loss.backward(); optim.step()

        loss_sum += loss.item() * yb.size(0)
        pred      = logits.argmax(dim=1)
        correct  += (pred == yb).sum().item()
        total    += yb.size(0)
    return loss_sum / total, correct / total

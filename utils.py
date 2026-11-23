import random
import numpy as np
import torch
import torch.nn.functional as F


def set_seed(seed: int = 123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def overfit_one_batch(model, loader, criterion, device, steps=300, lr=1e-3):
    x, y = next(iter(loader))
    x, y = x.to(device), y.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        y_pred = model(x)
        loss = criterion(y_pred, y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % 50 == 0:
            print(f"step {step}: loss = {loss.item():.4f}")
    return losses


@torch.no_grad()
def run_inference(model, loader, criterion, device):
    """Полный инференс модели по даталоадеру: возвращает метрики, предсказания и вероятности."""
    model.eval()
    total_loss = 0.0
    n = 0
    all_preds = []
    all_targets = []
    all_probs = []

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        bs = x.size(0)
        total_loss += loss.item() * bs
        n += bs

        probs = F.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.append(probs.cpu())
        all_preds.append(preds.cpu())
        all_targets.append(y.cpu())

    preds_cat = torch.cat(all_preds)
    targets_cat = torch.cat(all_targets)
    probs_cat = torch.cat(all_probs)

    avg_loss = total_loss / max(1, n)
    acc = (preds_cat == targets_cat).float().mean().item()

    return {
        "loss": avg_loss,
        "acc": acc,
        "preds": preds_cat.numpy(),
        "targets": targets_cat.numpy(),
        "probs": probs_cat.numpy(),
    }


def confusion_matrix(targets, preds, num_classes=10):
    """строки - истина, столбцы - предсказание"""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm
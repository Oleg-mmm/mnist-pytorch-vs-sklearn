import math
import torch
import torch.nn as nn


def make_optimizer(cfg, model):
    if cfg.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    if cfg.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay,
        )
    raise ValueError(cfg.optimizer)


def make_scheduler(cfg, optimizer, total_steps):
    warmup = int(cfg.warmup_frac * total_steps)
    if cfg.schedule is None:
        return None
    if cfg.schedule == "cosine":
        def fn(step):
            if step < warmup:
                return step / max(1, warmup)
            p = (step - warmup) / max(1, total_steps - warmup)
            return 0.5 * (1.0 + math.cos(math.pi * p))
        return torch.optim.lr_scheduler.LambdaLR(optimizer, fn)
    if cfg.schedule == "step":
        return torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_steps // 5), gamma=0.3
        )
    raise ValueError(cfg.schedule)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, n, correct = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        total_loss += criterion(logits, y).item() * x.size(0)
        n += x.size(0)
        correct += (logits.argmax(1) == y).sum().item()
    return total_loss / n, correct / n


def train(cfg, model, train_loader, val_loader, device):
    model = model.to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = make_optimizer(cfg, model)
    total_steps = cfg.epochs * len(train_loader)
    scheduler = make_scheduler(cfg, optimizer, total_steps)

    history = []
    bad = 0
    best_loss = float("inf")
    for epoch in range(cfg.epochs):
        model.train()
        epoch_loss, n = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()
            if scheduler:
                scheduler.step()
            epoch_loss += loss.item() * x.size(0)
            n += x.size(0)

        epoch_loss /= n
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        history.append((epoch + 1, epoch_loss, val_loss, val_acc))

        if val_loss < best_loss - cfg.delta:
            best_loss = val_loss
            bad = 0
            torch.save(model.state_dict(), cfg.ckpt_path)
            print(
                f"epoch {epoch+1:2d}  train {epoch_loss:.4f}  val {val_loss:.4f}  "
                f"acc {val_acc:.4f}  *NEW BEST MODEL: LOSS = {best_loss:.4f}"
            )
        else:
            bad += 1
            print(
                f"epoch {epoch+1:2d}  train {epoch_loss:.4f}  val {val_loss:.4f}  "
                f"acc {val_acc:.4f}  best {best_loss:.4f}"
            )
            if bad >= cfg.patience:
                print(f"EARLY STOPPING ON {epoch+1} EPOCH")
                break

    model.load_state_dict(torch.load(cfg.ckpt_path, map_location=device))
    return model, history
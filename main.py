import argparse
import numpy as np
import torch
import torch.nn as nn

from config import Config
from data import get_loaders
from model import build_model
from train import train
from utils import confusion_matrix, overfit_one_batch, run_inference, set_seed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlp", choices=["mlp", "conv"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--optim", default=None, choices=["adamw", "sgd"])
    ap.add_argument("--no_residual", action="store_true")
    ap.add_argument("--no_sanity", action="store_true")
    args = ap.parse_args()

    cfg = Config()
    cfg.model_name = args.model
    if args.epochs is not None:
        cfg.epochs = args.epochs
    if args.lr is not None:
        cfg.lr = args.lr
    if args.optim is not None:
        cfg.optimizer = args.optim
    if args.no_residual:
        cfg.use_residual = False

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    print("device:", device)
    set_seed(cfg.seed)

    train_loader, val_loader, test_loader = get_loaders(cfg)

    if not args.no_sanity:
        print("\noverfit one batch")
        tmp_model = build_model(cfg).to(device)
        crit = nn.CrossEntropyLoss()
        overfit_one_batch(tmp_model, train_loader, crit, device, steps=300)
        del tmp_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    model = build_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nmodel: {cfg.model_name}  params: {n_params:,}")

    model, history = train(cfg, model, train_loader, val_loader, device)

    print("\nrunning best model on test_dataset")
    criterion = nn.CrossEntropyLoss()
    test_res = run_inference(model, test_loader, criterion, device)
    print(f"FINAL test_loss {test_res['loss']:.4f}  test_acc {test_res['acc']:.4f}")

    cm = confusion_matrix(test_res["targets"], test_res["preds"], num_classes=10)
    print("\nconfusion matrix (rows=true, cols=pred):")
    print(cm)

    off = cm.copy()
    np.fill_diagonal(off, 0)
    top = np.dstack(np.unravel_index(np.argsort(off.ravel())[-5:][::-1], off.shape))[0]
    print("\nтоп-5 ошибок (true -> pred, count):")
    for t, p in top:
        print(f"  {t} -> {p} : {off[t, p]}")


if __name__ == "__main__":
    main()
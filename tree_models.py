import argparse
import gzip
import os
import struct
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, log_loss


def load_mnist_images(path):
    with gzip.open(path, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
        return data.reshape(num, rows * cols).astype(np.float32) / 255.0


def load_mnist_labels(path):
    with gzip.open(path, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        return np.frombuffer(f.read(), dtype=np.uint8).astype(np.int64)


def load_dataset(data_dir="./data", val_size=6000, seed=123):
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    X_train_full = load_mnist_images(os.path.join(raw_dir, "train-images-idx3-ubyte.gz"))
    y_train_full = load_mnist_labels(os.path.join(raw_dir, "train-labels-idx1-ubyte.gz"))
    X_test = load_mnist_images(os.path.join(raw_dir, "t10k-images-idx3-ubyte.gz"))
    y_test = load_mnist_labels(os.path.join(raw_dir, "t10k-labels-idx1-ubyte.gz"))

    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(X_train_full))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    return (
        X_train_full[train_idx],
        y_train_full[train_idx],
        X_train_full[val_idx],
        y_train_full[val_idx],
        X_test,
        y_test,
    )


def build_model(name, seed=123):
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=seed,
        )
    if name == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(
            n_estimators=150,
            learning_rate=0.05,
            num_leaves=31,
            n_jobs=-1,
            random_state=seed,
            verbose=-1,
        )
    raise ValueError(f"Unknown model: {name}. Use 'rf' or 'lgbm'")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="rf", choices=["rf", "lgbm"])
    ap.add_argument("--data_dir", default="./data")
    ap.add_argument("--val_size", type=int, default=6000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(
        data_dir=args.data_dir, val_size=args.val_size, seed=args.seed
    )

    model = build_model(args.model, seed=args.seed)
    model.fit(X_train, y_train)

    val_preds = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_preds)
    print(f"val_acc {val_acc:.4f}")

    test_preds = model.predict(X_test)
    test_probs = model.predict_proba(X_test)
    test_loss = log_loss(y_test, test_probs)
    test_acc = accuracy_score(y_test, test_preds)
    print(f"\nFINAL  test_loss {test_loss:.4f}  test_acc {test_acc:.4f}")

    cm = confusion_matrix(y_test, test_preds)
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
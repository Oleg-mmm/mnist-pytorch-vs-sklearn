import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_loaders(cfg):
    train_tfm = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08), scale=(0.95, 1.05)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.13), std=(0.3))
    ])
    eval_tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.13), std=(0.3))
    ])

    train_full_aug = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=train_tfm)
    train_full_clean = datasets.MNIST(cfg.data_dir, train=True, download=True, transform=eval_tfm)
    test_ds = datasets.MNIST(cfg.data_dir, train=False, download=True, transform=eval_tfm)

    total_train = len(train_full_clean)
    val_size = min(cfg.val_size, total_train - 1)
    generator = torch.Generator().manual_seed(cfg.seed)
    perm = torch.randperm(total_train, generator=generator).tolist()

    val_indices = perm[:val_size]
    train_indices = perm[val_size:]

    train_ds = Subset(train_full_aug, train_indices)
    val_ds = Subset(train_full_clean, val_indices)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=256, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader = DataLoader(
        test_ds, batch_size=256, shuffle=False, num_workers=cfg.num_workers
    )
    return train_loader, val_loader, test_loader
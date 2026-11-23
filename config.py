from dataclasses import dataclass


@dataclass
class Config:
    # data
    data_dir: str = "./data"
    num_workers: int = 2
    batch_size: int = 64
    val_size: int = 6000

    # model
    model_name: str = "mlp"
    hidden_dim: int = 256
    n_layers: int = 3
    dropout: float = 0.2
    use_residual: bool = True

    # optimizer
    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    momentum: float = 0.9

    # schedule
    epochs: int = 15
    warmup_frac: float = 0.05
    schedule: str = "cosine"

    # early stopping
    patience: int = 5
    delta: float = 0.01

    # other
    seed: int = 123
    device: str = "cuda"
    ckpt_path: str = "best.pt"
    batch_count: int = 100
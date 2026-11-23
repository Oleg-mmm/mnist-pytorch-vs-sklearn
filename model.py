import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, dim, dropout, use_residual):
        super().__init__()
        self.use_residual = use_residual
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        if use_residual:
            nn.init.zeros_(self.block[-1].weight)
            nn.init.zeros_(self.block[-1].bias)

    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_residual else out


class MLPModel(nn.Module):
    def __init__(self, input, hidden, n_classes, dropout, use_residual=True, num_blocks=3):
        super().__init__()
        self.use_residual = use_residual
        self.stem = nn.Linear(input, hidden)
        self.backbone = nn.Sequential(
            *[MLPBlock(hidden, dropout, use_residual) for _ in range(num_blocks)]
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.Dropout(0.35),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.stem(x)
        x = self.backbone(x)
        x = self.head(x)
        return x


class ConvModel(nn.Module):
    def __init__(self, in_channels=1, n_classes=10, dropout=0.1):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(32 * 2 * 2, n_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def build_model(cfg):
    if cfg.model_name == "mlp":
        return MLPModel(
            input=28 * 28,
            hidden=cfg.hidden_dim,
            n_classes=10,
            num_blocks=cfg.n_layers,
            dropout=cfg.dropout,
            use_residual=cfg.use_residual,
        )
    if cfg.model_name == "conv":
        return ConvModel(
            in_channels=1,
            n_classes=10,
            dropout=cfg.dropout,
        )
    raise ValueError(f"Unknown model name: {cfg.model_name}. Use 'mlp' or 'conv'")
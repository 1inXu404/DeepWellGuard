"""Pure CNN baseline for oil well event classification.

The model keeps the same contract as the rest of the project:
input ``(batch, N_FEATURES, 120)`` and output ``(batch, 7)`` logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import CNN_CHANNELS, CNN_KERNEL, N_CLASSES, N_FEATURES


class CNNModel(nn.Module):
    """Temporal 1D-CNN baseline without recurrent or attention layers."""

    def __init__(self):
        super().__init__()

        channels = [N_FEATURES, *CNN_CHANNELS]
        padding = CNN_KERNEL // 2
        conv_blocks = []
        for in_channels, out_channels in zip(channels[:-1], channels[1:]):
            conv_blocks.extend(
                [
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=CNN_KERNEL,
                        padding=padding,
                    ),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2),
                ]
            )

        self.feature_extractor = nn.Sequential(*conv_blocks)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(CNN_CHANNELS[-1], 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input shaped ``(batch, N_FEATURES, 120)``."""
        x = self.feature_extractor(x)
        x = self.global_pool(x).squeeze(-1)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        return F.softmax(self.forward(x), dim=1)

"""CNN baseline model for oil well event detection (time series classification).

Input:  (batch, 22 sensors, 120 time steps)
Output: (batch, 7) logits for 7 event classes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNModel(nn.Module):
    """3-block 1D CNN for time series classification.

    Architecture:
        Input:  (batch, 22, 120)
        Block1: Conv1d(22→64, k=3, p=1) → BatchNorm1d(64) → ReLU → MaxPool1d(2)
        Block2: Conv1d(64→128, k=3, p=1) → BatchNorm1d(128) → ReLU → MaxPool1d(2)
        Block3: Conv1d(128→256, k=3, p=1) → BatchNorm1d(256) → ReLU → MaxPool1d(2)
        Pooling: AdaptiveAvgPool1d(1) → Flatten → (batch, 256)
        FC: Linear(256, 128) → ReLU → Dropout(0.3) → Linear(128, 7)
        Output: (batch, 7) — raw logits for 7 classes

    Parameter count: ~500K–1M
    """

    def __init__(self):
        super().__init__()

        # Conv blocks
        self.block1 = nn.Sequential(
            nn.Conv1d(22, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block2 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )
        self.block3 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        self.pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns raw logits.

        Args:
            x: Input tensor of shape (batch, 22, 120).

        Returns:
            Logits tensor of shape (batch, 7).
        """
        x = self.block1(x)  # (batch, 64, 60)
        x = self.block2(x)  # (batch, 128, 30)
        x = self.block3(x)  # (batch, 256, 15)
        x = self.pool(x)    # (batch, 256, 1)
        x = x.squeeze(-1)   # (batch, 256)
        x = self.fc(x)      # (batch, 7)
        return x

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (indices).

        Args:
            x: Input tensor of shape (batch, 22, 120).

        Returns:
            LongTensor of shape (batch,) with predicted class indices.
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities.

        Args:
            x: Input tensor of shape (batch, 22, 120).

        Returns:
            Tensor of shape (batch, 7) with class probabilities.
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

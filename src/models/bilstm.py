"""Pure bidirectional LSTM baseline for oil well event classification."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import LSTM_HIDDEN, LSTM_LAYERS, N_CLASSES, N_FEATURES


class BiLSTMModel(nn.Module):
    """
    2-layer bidirectional LSTM for time series classification.

    Input: ``(batch, N_FEATURES, 120)`` and output: ``(batch, 7)`` logits.
    """

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=LSTM_HIDDEN,
            num_layers=LSTM_LAYERS,
            bidirectional=True,
            batch_first=True,
        )

        self.fc = nn.Sequential(
            nn.Linear(LSTM_HIDDEN * 2, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits for input shaped ``(batch, N_FEATURES, 120)``."""
        x = x.permute(0, 2, 1)
        _, (h_n, _) = self.lstm(x)
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)
        return self.fc(hidden)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils.config import (
    LSTM_HIDDEN,
    LSTM_LAYERS,
    LSTM_BIDIRECTIONAL,
    N_FEATURES,
    N_CLASSES,
)


class LSTMModel(nn.Module):
    """
    2-layer Bidirectional LSTM for time series classification (Pure Baseline).

    Architecture:
        Input:  (batch, 22, 120) → permute → (batch, 120, 22)
        LSTM: LSTM(22, 128, num_layers=2, bidirectional=True, batch_first=True)
        Hidden extraction: concat last forward/backward h_n → (batch, 256)
        FC: Linear(256, 64) → ReLU → Linear(64, 7)
        Output: (batch, 7)  — logits for 7 classes
    """

    def __init__(self):
        super().__init__()

        lstm_hidden = LSTM_HIDDEN
        lstm_layers = LSTM_LAYERS
        lstm_bidirectional = LSTM_BIDIRECTIONAL

        self.lstm = nn.LSTM(
            input_size=N_FEATURES,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            bidirectional=lstm_bidirectional,
            batch_first=True,
        )

        lstm_output_dim = lstm_hidden * 2 if lstm_bidirectional else lstm_hidden

        # FC without Dropout (Pure Baseline)
        self.fc = nn.Sequential(
            nn.Linear(lstm_output_dim, 64),
            nn.ReLU(),
            nn.Linear(64, N_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, 22, 120) — [batch, features, time_steps]

        Returns:
            logits: (batch, 7)
        """
        # Permute from (batch, features, time) → (batch, time, features)
        x = x.permute(0, 2, 1)  # (batch, 120, 22)

        # LSTM forward: output is (batch, seq_len, hidden*D), h_n is (D*layers, batch, hidden)
        _, (h_n, _) = self.lstm(x)  # h_n: (4, batch, 128) for 2-layer bidirectional

        # Concatenate last forward and last backward hidden states
        # h_n[-2]: last forward layer, h_n[-1]: last backward layer
        hidden = torch.cat([h_n[-2], h_n[-1]], dim=1)  # (batch, 256)

        return self.fc(hidden)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class indices.

        Args:
            x: (batch, 22, 120)

        Returns:
            (batch,) LongTensor of class predictions
        """
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities.

        Args:
            x: (batch, 22, 120)

        Returns:
            (batch, 7) softmax probabilities
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

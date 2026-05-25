"""CNN-LSTM-Channel-Attention hybrid model for oil well event detection.

Input:  (batch, 22 sensors, 120 time steps)
Output: (batch, 7) logits for 7 event classes.

Architecture:
- 1st Derivative concatenation (captures sudden changes)
- Multi-Scale CNN blocks (k=3, 7, 11) for different temporal scales
- BiLSTM Temporal Encoder
- Channel Attention over post-BiLSTM features
- Global Max Pooling to capture anomaly peaks
- FC classifier
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention1d(nn.Module):
    """Channel attention using average and max temporal descriptors."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        reduced_channels = max(channels // reduction, 1)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.shared_mlp = nn.Sequential(
            nn.Linear(channels, reduced_channels, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduced_channels, channels, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply channel-wise reweighting to ``(batch, channels, time)``."""
        avg_descriptor = self.avg_pool(x).squeeze(-1)
        max_descriptor = self.max_pool(x).squeeze(-1)
        weights = self.shared_mlp(avg_descriptor) + self.shared_mlp(max_descriptor)
        weights = self.sigmoid(weights).unsqueeze(-1)
        return x * weights


class MultiScaleConv1d(nn.Module):
    """Multi-Scale Convolution block to capture both sudden and slow events."""

    def __init__(self, in_channels, out_channels):
        super(MultiScaleConv1d, self).__init__()
        assert out_channels % 3 == 0, "out_channels should be divisible by 3"
        branch_out = out_channels // 3

        self.branch1 = nn.Conv1d(in_channels, branch_out, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, branch_out, kernel_size=7, padding=3)
        self.branch3 = nn.Conv1d(in_channels, branch_out, kernel_size=11, padding=5)

        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(2)

    def forward(self, x):
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat([x1, x2, x3], dim=1)
        out = self.bn(out)
        out = self.relu(out)
        out = self.pool(out)
        return out


class CNNLSTMAttention(nn.Module):
    """
    CNN-LSTM-Channel-Attention model.

    Pipeline: 1st derivative -> multi-scale CNN -> BiLSTM ->
    channel attention -> global max pooling -> classifier.
    """

    def __init__(self):
        super().__init__()

        # We concatenate original input (22) with its 1st derivative (22) -> 44 channels
        in_channels = 44

        # CNN Part: Block 1
        self.ms_block1 = MultiScaleConv1d(in_channels, 96)

        # CNN Part: Block 2
        self.ms_block2 = MultiScaleConv1d(96, 192)

        # LSTM Part: Temporal Encoder
        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=96,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Channel Attention over post-BiLSTM high-level temporal features
        self.channel_attention = ChannelAttention1d(192)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass, returns raw logits.

        Args:
            x: Input tensor of shape (batch, 22, 120).

        Returns:
            Logits tensor of shape (batch, 7).
        """
        # Step 0: Calculate 1st derivative (rate of change)
        diff = torch.zeros_like(x)
        diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
        x = torch.cat([x, diff], dim=1)  # (batch, 44, 120)

        # Step 1: Multi-scale CNN
        x = self.ms_block1(x)  # (batch, 96, 60)
        x = self.ms_block2(x)  # (batch, 192, 30)

        # Step 2: LSTM Temporal Encoding
        x = x.permute(0, 2, 1)  # (batch, 30, 192)
        lstm_out, _ = self.lstm(x)  # (batch, 30, 192)

        # Step 3: Channel Attention over BiLSTM features
        features = lstm_out.permute(0, 2, 1)  # (batch, 192, 30)
        features = self.channel_attention(features)

        # Step 4: Global Max Pooling
        pooled = features.max(dim=2)[0]  # (batch, 192)

        # Step 5: Classifier
        logits = self.classifier(pooled)  # (batch, 7)
        return logits

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class predictions (indices)."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities."""
        logits = self.forward(x)
        return F.softmax(logits, dim=1)

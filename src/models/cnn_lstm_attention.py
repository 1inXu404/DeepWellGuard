"""CNN-LSTM-Attention hybrid model for oil well event detection.

Input:  (batch, 22 sensors, 120 time steps)
Output: (batch, 7) logits for 7 event classes.

Architecture: CNN feature extractor → LSTM temporal encoder →
              Multi-head self-attention → Global pooling → FC classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNLSTMAttention(nn.Module):
    """CNN-LSTM-Attention hybrid model for time series classification.

    Architecture:
        Input:  (batch, 22, 120)

        CNN Feature Extractor (2 blocks, lightweight):
        Block1: Conv1d(22→32, k=3, p=1) → BatchNorm1d → ReLU → MaxPool1d(2)
                (batch, 32, 60)
        Block2: Conv1d(32→64, k=3, p=1) → BatchNorm1d → ReLU → MaxPool1d(2)
                (batch, 64, 30)

        LSTM Temporal Encoder:
        permute: (batch, 64, 30) → (batch, 30, 64)
        BiLSTM: LSTM(64, 64, 1, bidirectional=True, batch_first=True)
                output: (batch, 30, 128)

        Multi-head Self-Attention:
        attn = MultiheadAttention(embed_dim=128, num_heads=4, dropout=0.3, batch_first=True)
        Residual + LayerNorm(128)

        Global Pooling:
        pool = attn_out.mean(dim=1)  (batch, 128)

        Classifier:
        Linear(128, 64) → BatchNorm1d → ReLU → Dropout(0.4) → Linear(64, 7)
        Output: (batch, 7)  logits for 7 classes
    """

    def __init__(self):
        super().__init__()

        # CNN feature extractor — Block 1
        self.block1 = nn.Sequential(
            nn.Conv1d(22, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # CNN feature extractor — Block 2
        self.block2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
        )

        # BiLSTM temporal encoder
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            dropout=0.3,
            batch_first=True,
        )
        self.layer_norm = nn.LayerNorm(128)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
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
        # Step 1: CNN feature extraction
        x = self.block1(x)  # (batch, 32, 60)
        x = self.block2(x)  # (batch, 64, 30)

        # Step 2: LSTM temporal encoding
        x = x.permute(0, 2, 1)  # (batch, 30, 64)
        lstm_out, _ = self.lstm(x)  # (batch, 30, 128)

        # Step 3: Multi-head self-attention with residual
        attn_out, _ = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # (batch, 30, 128)
        
        # Add residual connection & LayerNorm (crucial for stabilization)
        lstm_out = self.layer_norm(lstm_out + attn_out)

        # Step 4: Global mean pooling
        pooled = lstm_out.mean(dim=1)  # (batch, 128)

        # Step 5: Classifier
        logits = self.classifier(pooled)  # (batch, 7)
        return logits

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

"""Ablation variants for the CNN-LSTM-Attention model.

The variants keep the same input/output contract as the main model:
input ``(batch, 22, 120)`` and output ``(batch, 7)`` logits.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.cnn_lstm_attention import MultiScaleConv1d, SEBlock


@dataclass(frozen=True)
class AblationConfig:
    """Feature switches for one ablation run."""

    name: str
    use_derivative: bool = True
    use_multiscale: bool = True
    use_se: bool = True
    use_temporal_attention: bool = True


ABLATION_CONFIGS = {
    "full": AblationConfig(name="full"),
    "no_derivative": AblationConfig(name="no_derivative", use_derivative=False),
    "single_scale": AblationConfig(name="single_scale", use_multiscale=False),
    "no_se": AblationConfig(name="no_se", use_se=False),
    "no_temporal_attention": AblationConfig(
        name="no_temporal_attention",
        use_temporal_attention=False,
    ),
}


class SingleScaleConv1d(nn.Module):
    """Single-kernel convolution block used to replace multi-scale CNN."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class AblationCNNLSTMAttention(nn.Module):
    """Configurable CNN-LSTM-Attention model for ablation experiments."""

    def __init__(self, config: AblationConfig):
        super().__init__()
        self.config = config

        in_channels = 44 if config.use_derivative else 22
        conv_cls = MultiScaleConv1d if config.use_multiscale else SingleScaleConv1d

        self.conv_block1 = conv_cls(in_channels, 96)
        self.se_block1 = SEBlock(96) if config.use_se else nn.Identity()

        self.conv_block2 = conv_cls(96, 192)
        self.se_block2 = SEBlock(192) if config.use_se else nn.Identity()

        self.lstm = nn.LSTM(
            input_size=192,
            hidden_size=96,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )

        if config.use_temporal_attention:
            self.attention = nn.MultiheadAttention(
                embed_dim=192,
                num_heads=4,
                dropout=0.3,
                batch_first=True,
            )
        else:
            self.attention = None

        self.layer_norm = nn.LayerNorm(192)
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 7),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.config.use_derivative:
            diff = torch.zeros_like(x)
            diff[:, :, 1:] = x[:, :, 1:] - x[:, :, :-1]
            x = torch.cat([x, diff], dim=1)

        x = self.conv_block1(x)
        x = self.se_block1(x)
        x = self.conv_block2(x)
        x = self.se_block2(x)

        x = x.permute(0, 2, 1)
        lstm_out, _ = self.lstm(x)

        if self.attention is not None:
            attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)
            lstm_out = self.layer_norm(lstm_out + attn_out)
        else:
            lstm_out = self.layer_norm(lstm_out)

        pooled = lstm_out.max(dim=1)[0]
        return self.classifier(pooled)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return torch.argmax(self.forward(x), dim=1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=1)


def get_ablation_config(name: str) -> AblationConfig:
    """Return an ablation config by name."""
    try:
        return ABLATION_CONFIGS[name]
    except KeyError as exc:
        available = ", ".join(sorted(ABLATION_CONFIGS))
        raise ValueError(f"Unknown ablation variant '{name}'. Available: {available}") from exc

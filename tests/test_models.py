"""Pytest tests for all models: CNN, Bi-LSTM, CNN-LSTM-Attention."""

import torch
import torch.nn as nn
import tempfile
from pathlib import Path

from src.models.cnn import CNNModel
from src.models.bilstm import BiLSTMModel
from src.models.cnn_lstm_attention import ChannelAttention1d, CNNLSTMAttention
from src.models.unilstm import UniLSTMModel
from src.models.ablation import ABLATION_CONFIGS, AblationCNNLSTMAttention
from src.utils.config import N_FEATURES, WINDOW_SIZE


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _overfit_single_batch(model: nn.Module, lr: float = 0.01,
                          steps: int = 20) -> list[float]:
    """Train a model on a single batch and check that loss decreases.

    Returns the list of losses over *steps* optimisation steps.
    """
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    x = torch.randn(8, N_FEATURES, WINDOW_SIZE)
    y = torch.randint(0, 7, (8,))
    losses: list[float] = []
    for _ in range(steps):
        opt.zero_grad()
        loss = criterion(model(x), y)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def _forward_shape_test(model: nn.Module):
    """Verify forward pass produces (batch, 7)."""
    out = model(torch.randn(4, N_FEATURES, WINDOW_SIZE))
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"


def _predict_test(model: nn.Module):
    """Verify predict returns valid class indices."""
    preds = model.predict(torch.randn(4, N_FEATURES, WINDOW_SIZE))
    assert preds.shape == (4,), f"Expected (4,), got {preds.shape}"
    assert all(0 <= x < 7 for x in preds), f"Predictions out of range: {preds}"


def _save_load_test(model: nn.Module):
    """Verify save/load round-trip preserves forward pass results."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "_test_model.pt"
        torch.save(model.state_dict(), path)
        model_cls = type(model)
        loaded = model_cls()
        loaded.load_state_dict(torch.load(path, weights_only=True))
        x = torch.randn(4, N_FEATURES, WINDOW_SIZE)
        # Switch to eval mode so Dropout/BatchNorm behave deterministically
        model.eval()
        loaded.eval()
        with torch.no_grad():
            expected = model(x)
            actual = loaded(x)
        assert torch.allclose(expected, actual, atol=1e-6), (
            "Loaded model forward differs from original"
        )


# ---------------------------------------------------------------------------
# CNN
# ---------------------------------------------------------------------------

class TestCNN:
    """Tests for CNNModel."""

    def test_forward_shape(self):
        _forward_shape_test(CNNModel())

    def test_overfit(self):
        losses = _overfit_single_batch(CNNModel())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_predict(self):
        _predict_test(CNNModel())

    def test_save_load(self):
        _save_load_test(CNNModel())


# ---------------------------------------------------------------------------
# Uni-LSTM
# ---------------------------------------------------------------------------

class TestUniLSTM:
    """Tests for UniLSTMModel."""

    def test_forward_shape(self):
        _forward_shape_test(UniLSTMModel())

    def test_overfit(self):
        losses = _overfit_single_batch(UniLSTMModel())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_predict(self):
        _predict_test(UniLSTMModel())

    def test_save_load(self):
        _save_load_test(UniLSTMModel())


# ---------------------------------------------------------------------------
# Bi-LSTM
# ---------------------------------------------------------------------------

class TestBiLSTM:
    """Tests for BiLSTMModel."""

    def test_forward_shape(self):
        _forward_shape_test(BiLSTMModel())

    def test_overfit(self):
        losses = _overfit_single_batch(BiLSTMModel())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_predict(self):
        _predict_test(BiLSTMModel())

    def test_save_load(self):
        _save_load_test(BiLSTMModel())


# ---------------------------------------------------------------------------
# CNN-LSTM-Attention
# ---------------------------------------------------------------------------

class TestCNNLSTMAttention:
    """Tests for CNNLSTMAttention."""

    def test_forward_shape(self):
        _forward_shape_test(CNNLSTMAttention())

    def test_overfit(self):
        losses = _overfit_single_batch(CNNLSTMAttention())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_predict(self):
        _predict_test(CNNLSTMAttention())

    def test_save_load(self):
        _save_load_test(CNNLSTMAttention())

    def test_params_count(self):
        n_attn = sum(p.numel() for p in CNNLSTMAttention().parameters())
        assert n_attn > 0, "Model should have parameters"

    def test_channel_attention_shape(self):
        module = ChannelAttention1d(192)
        x = torch.randn(4, 192, 30)
        out = module(x)
        assert out.shape == x.shape


class TestAblationCNNLSTMAttention:
    """Tests for all ablation variants."""

    def test_all_variants_forward_shape(self):
        for config in ABLATION_CONFIGS.values():
            _forward_shape_test(AblationCNNLSTMAttention(config))

    def test_all_variants_predict(self):
        for config in ABLATION_CONFIGS.values():
            _predict_test(AblationCNNLSTMAttention(config))

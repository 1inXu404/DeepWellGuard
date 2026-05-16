"""Pytest tests for all 3 models: CNN, LSTM, CNN-LSTM-Attention."""

import copy

import torch
import torch.nn as nn

from src.models.cnn import CNNModel
from src.models.lstm import LSTMModel
from src.models.cnn_lstm_attention import CNNLSTMAttention


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
    x = torch.randn(8, 22, 120)
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
    out = model(torch.randn(4, 22, 120))
    assert out.shape == (4, 7), f"Expected (4, 7), got {out.shape}"


def _predict_test(model: nn.Module):
    """Verify predict returns valid class indices."""
    preds = model.predict(torch.randn(4, 22, 120))
    assert preds.shape == (4,), f"Expected (4,), got {preds.shape}"
    assert all(0 <= x < 7 for x in preds), f"Predictions out of range: {preds}"


def _save_load_test(model: nn.Module, path: str = "/tmp/_test_model.pt"):
    """Verify save/load round-trip preserves forward pass results."""
    torch.save(model.state_dict(), path)
    model_cls = type(model)
    loaded = model_cls()
    loaded.load_state_dict(torch.load(path, weights_only=True))
    x = torch.randn(4, 22, 120)
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
# LSTM
# ---------------------------------------------------------------------------

class TestLSTM:
    """Tests for LSTMModel."""

    def test_forward_shape(self):
        _forward_shape_test(LSTMModel())

    def test_overfit(self):
        losses = _overfit_single_batch(LSTMModel())
        assert losses[-1] < losses[0], (
            f"Loss did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"
        )

    def test_predict(self):
        _predict_test(LSTMModel())

    def test_save_load(self):
        _save_load_test(LSTMModel())


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

    def test_largest_params(self):
        n_cnn = sum(p.numel() for p in CNNModel().parameters())
        n_lstm = sum(p.numel() for p in LSTMModel().parameters())
        n_attn = sum(p.numel() for p in CNNLSTMAttention().parameters())
        assert n_attn > n_cnn, (
            f"CNNLSTMAttention ({n_attn}) should be larger than CNN ({n_cnn})"
        )
        assert n_attn > n_lstm, (
            f"CNNLSTMAttention ({n_attn}) should be larger than LSTM ({n_lstm})"
        )

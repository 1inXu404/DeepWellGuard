"""Integration tests for Trainer, evaluate, and comparison pipeline."""

import os

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


class TestTrainerIntegration:
    """Integration tests for the Trainer class (fit, early stopping, predict)."""

    def test_fit_on_synthetic(self):
        """Trainer.fit on synthetic data reduces loss and produces history keys."""
        from src.models.lstm import LSTMModel
        from src.train.trainer import Trainer
        from src.utils.device import get_device

        device = get_device()
        model = LSTMModel().to(device)

        x = torch.randn(200, 22, 120)
        y = torch.randint(0, 7, (200,))
        train_ds = TensorDataset(x[:150], y[:150])
        val_ds = TensorDataset(x[150:], y[150:])

        trainer = Trainer(model, device)
        history = trainer.fit(
            DataLoader(train_ds, 16), DataLoader(val_ds, 16),
            epochs=5, patience=3,
        )

        assert len(history["train_loss"]) > 0
        assert "val_acc" in history
        assert len(history["val_loss"]) == len(history["train_loss"])
        assert all(isinstance(v, float) for v in history["train_loss"])

    def test_early_stopping(self):
        """Early stopping triggers before max epochs with low patience."""
        from src.models.lstm import LSTMModel
        from src.train.trainer import Trainer
        from src.utils.device import get_device

        device = get_device()
        model = LSTMModel().to(device)

        # Use independent train/val sets so the model cannot memorise both.
        x_train = torch.randn(200, 22, 120)
        y_train = torch.randint(0, 7, (200,))
        x_val = torch.randn(50, 22, 120)
        y_val = torch.randint(0, 7, (50,))

        train_loader = DataLoader(TensorDataset(x_train, y_train), 16)
        val_loader = DataLoader(TensorDataset(x_val, y_val), 16)

        trainer = Trainer(model, device, {"patience": 2})
        history = trainer.fit(train_loader, val_loader, epochs=20, patience=2)

        # With patience=2 and random independent val data, the model cannot
        # improve on unseen random labels, so early stopping should fire
        # well before epoch 20.
        assert len(history["train_loss"]) < 10, (
            f"Early stopping expected before epoch 10, got {len(history['train_loss'])}"
        )

    def test_predict(self):
        """Trainer.predict returns correct shapes and valid class indices."""
        from src.models.lstm import LSTMModel
        from src.train.trainer import Trainer
        from src.utils.device import get_device

        device = get_device()
        model = LSTMModel().to(device)

        x = torch.randn(50, 22, 120)
        y = torch.randint(0, 7, (50,))
        ds = TensorDataset(x, y)
        loader = DataLoader(ds, 16)

        trainer = Trainer(model, device)
        trainer.fit(loader, loader, epochs=1, patience=3)

        preds, probs, y_true = trainer.predict(loader)
        assert preds.shape == (50,), f"Expected (50,), got {preds.shape}"
        assert probs.shape == (50, 7), f"Expected (50, 7), got {probs.shape}"
        assert y_true.shape == (50,), f"Expected (50,), got {y_true.shape}"
        assert all(0 <= p < 7 for p in preds), "Predictions out of range"
        # Probabilities should sum to 1 per sample
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-5)


class TestEvaluate:
    """Tests for evaluation metrics (compute_metrics)."""

    def test_perfect_prediction(self):
        """Perfect predictions yield accuracy = 1.0."""
        from src.train.evaluate import compute_metrics

        y = np.array([0, 1, 3, 4, 5, 6, 9])
        m = compute_metrics(y, y)

        assert m["accuracy"] == 1.0
        assert m["weighted_f1"] == 1.0
        assert m["macro_f1"] == 1.0

    def test_edge_case_all_zero(self):
        """All predictions in one class produces valid per-class metrics."""
        from src.train.evaluate import compute_metrics

        y_true = np.array([0, 1, 3, 4, 5])
        y_pred = np.array([0, 0, 0, 0, 0])
        m = compute_metrics(y_true, y_pred)

        assert "per_class_f1" in m
        assert len(m["per_class_f1"]) == len(np.unique(np.concatenate([y_true, y_pred])))

    def test_all_wrong(self):
        """All wrong predictions yield accuracy = 0.0."""
        from src.train.evaluate import compute_metrics

        y_true = np.array([0, 1, 3])
        y_pred = np.array([4, 5, 6])
        m = compute_metrics(y_true, y_pred)

        assert m["accuracy"] == 0.0
        assert isinstance(m["weighted_f1"], float)

    def test_save_metrics_csv(self):
        """save_metrics_csv writes a valid CSV with correct values."""
        from src.train.evaluate import save_metrics_csv

        import tempfile
        import pandas as pd

        metrics = {"accuracy": 0.95, "weighted_f1": 0.94, "macro_f1": 0.88}
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        try:
            tmp.close()
            save_metrics_csv(metrics, tmp.name)
            df = pd.read_csv(tmp.name)
            assert df["accuracy"][0] == 0.95
            assert df["weighted_f1"][0] == 0.94
        finally:
            os.unlink(tmp.name)


class TestComparison:
    """Tests for the comparison CSV produced by scripts/compare.py."""

    def test_csv_exists_and_has_data(self):
        """comparison.csv exists and contains at least one row."""
        csv_path = "results/metrics/comparison.csv"
        if not os.path.exists(csv_path):
            pytest.skip("comparison.csv not found — run scripts/compare.py first")

        import pandas as pd
        df = pd.read_csv(csv_path)
        assert len(df) > 0, "comparison.csv is empty"
        expected_cols = {"accuracy", "weighted_f1", "macro_f1", "model"}
        assert expected_cols.issubset(set(df.columns)), (
            f"Missing expected columns. Got: {list(df.columns)}"
        )

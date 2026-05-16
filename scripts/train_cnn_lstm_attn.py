#!/usr/bin/env python3
"""
CNN-LSTM-Attention training script with 5-fold cross-validation.

Loads preprocessed cache files (fold_{0-4}_X/y.npy, fold_{0-4}_val_X/y.npy),
trains a CNNLSTMAttention model per fold via Trainer with early stopping,
ensembles the fold models via averaged softmax probabilities, and evaluates
on the holdout test set (test_X/y.npy).

Usage:
    python scripts/train_cnn_lstm_attn.py
    python scripts/train_cnn_lstm_attn.py --epochs 50 --patience 10
"""

import argparse
import os
import random
import sys
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OilWellDataset
from src.models.cnn_lstm_attention import CNNLSTMAttention
from src.train.evaluate import compute_metrics
from src.train.trainer import Trainer
from src.utils.config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, MAX_EPOCHS, RETAINED_CLASSES, SEED
from src.utils.device import get_device


def _fold_cache_exists(fold: int) -> bool:
    """Check whether all four cache files for a given fold exist."""
    cache_dir = "results/cache"
    required = [
        os.path.join(cache_dir, f"fold_{fold}_X.npy"),
        os.path.join(cache_dir, f"fold_{fold}_y.npy"),
        os.path.join(cache_dir, f"fold_{fold}_val_X.npy"),
        os.path.join(cache_dir, f"fold_{fold}_val_y.npy"),
    ]
    return all(os.path.isfile(p) for p in required)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train CNN-LSTM-Attention with 5-fold cross-validation on 3W oil well data."
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS, help="Max training epochs"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=EARLY_STOPPING_PATIENCE,
        help="Early-stopping patience",
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE, help="Batch size"
    )
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to use (0.0-1.0) for faster training")
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join("results", "models", run_id)
    metrics_dir = os.path.join("results", "metrics", run_id)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    print(f"Run ID: {run_id}")

    # ── Set all seeds for reproducibility ──────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = get_device()
    use_cuda = device.type == "cuda"
    label_map = {orig: new for new, orig in enumerate(RETAINED_CLASSES)}
    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")
    print(f"Epochs: {args.epochs}  |  Patience: {args.patience}  |  Batch size: {args.batch_size}")

    # ── Single Fold Training (was 5-fold CV) ───────────────────────────────
    fold_accs: list[float] = []
    fold_models: list[CNNLSTMAttention] = []

    for fold in range(1):
        if not _fold_cache_exists(fold):
            print(f"\n[SKIP] Fold {fold} — cache files not found.")
            continue

        print(f"\n{'=' * 50}")
        print(f"  Fold {fold}")
        print(f"{'=' * 50}")

        train_ds = OilWellDataset(
            f"results/cache/fold_{fold}_X.npy",
            f"results/cache/fold_{fold}_y.npy",
        )
        val_ds = OilWellDataset(
            f"results/cache/fold_{fold}_val_X.npy",
            f"results/cache/fold_{fold}_val_y.npy",
        )

        # Remap labels (uses global label_map)
        train_ds.labels = np.array([label_map[lbl] for lbl in train_ds.labels])
        val_ds.labels = np.array([label_map[lbl] for lbl in val_ds.labels])

        # Load labels for balanced sampling
        train_labels = train_ds.labels
        class_counts = np.bincount(train_labels, minlength=7)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # guard missing classes
        sample_weights = 1.0 / class_counts[train_labels]
        num_train_samples = int(len(train_ds) * args.subset)
        sampler = WeightedRandomSampler(sample_weights, num_train_samples, replacement=True)

        if args.subset < 1.0:
            import torch
            val_indices = np.random.choice(len(val_ds), int(len(val_ds) * args.subset), replace=False)
            val_ds = torch.utils.data.Subset(val_ds, val_indices)

        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size,
            sampler=sampler,
            pin_memory=use_cuda, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            pin_memory=use_cuda, num_workers=0,
        )

        print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

        model = CNNLSTMAttention()
        trainer = Trainer(model, device)
        _ = trainer.fit(
            train_loader, val_loader, epochs=args.epochs, patience=args.patience
        )

        # Save model weights
        
        ckpt_path = f"{models_dir}/cnn_lstm_attn_fold{fold}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Model saved → {ckpt_path}")

        fold_models.append(model)

        # Validation accuracy
        val_preds, _ = trainer.predict(val_loader)
        val_acc = float((val_preds == val_ds.labels).mean())
        fold_accs.append(val_acc)
        print(f"  Fold {fold} val accuracy: {val_acc:.4f}")

    # ── Summary ────────────────────────────────────────────────────────
    if fold_accs:
        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        print(f"\n{'=' * 50}")
        print(f"  CNN-LSTM-Attention Single fold CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"{'=' * 50}")
    else:
        print("\nNo folds were trained — nothing to evaluate.")
        return

    # ── Ensemble on holdout test set ───────────────────────────────────
    test_X_path = "results/cache/test_X.npy"
    test_y_path = "results/cache/test_y.npy"

    if not (os.path.isfile(test_X_path) and os.path.isfile(test_y_path)):
        print("\nTest cache files not found — skipping test evaluation.")
        return

    print("\n" + "=" * 50)
    print("  Ensemble evaluation on holdout test set")
    print("=" * 50)

    test_ds = OilWellDataset(test_X_path, test_y_path)
    # Remap test labels too
    test_ds.labels = np.array([label_map[lbl] for lbl in test_ds.labels])
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_cuda, num_workers=0,
    )
    print(f"  Test samples: {len(test_ds)}")

    all_probs: list[np.ndarray] = []
    for model in fold_models:
        trainer = Trainer(model, device)
        _, probs = trainer.predict(test_loader)
        all_probs.append(probs)

    # Ensemble: average softmax probabilities across fold models
    avg_probs = np.mean(all_probs, axis=0)  # (N, n_classes)
    test_preds = np.argmax(avg_probs, axis=1)  # (N,)

    # Save predictions
    
    np.savez(
        os.path.join(metrics_dir, "cnn_lstm_attn_predictions.npz"),
        preds=test_preds,
        probs=avg_probs,
        labels=test_ds.labels,
    )
    print("  Predictions saved → results/metrics/cnn_lstm_attn_predictions.npz")

    # Compute and display metrics
    metrics = compute_metrics(test_ds.labels, test_preds, avg_probs)
    print(f"\n  Test accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Per-class F1:     {[round(f, 4) for f in metrics['per_class_f1']]}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
LSTM training script for 3W oil well dataset.
5-fold cross-validation + ensemble voting + holdout evaluation.

Usage:
    python scripts/train_lstm.py
    python scripts/train_lstm.py --epochs 150 --batch-size 64 --lr 0.0005
"""
import sys
import os
import argparse
import time
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import (
    SEED,
    BATCH_SIZE,
    LEARNING_RATE,
    MAX_EPOCHS,
    EARLY_STOPPING_PATIENCE,
    N_CLASSES,
    RETAINED_CLASSES,
)
from src.utils.device import get_device
from src.data.dataset import OilWellDataset
from src.models.lstm import LSTMModel
from src.train.trainer import Trainer
from src.train.evaluate import compute_metrics, save_metrics_csv


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train LSTM model with 5-fold cross-validation"
    )
    parser.add_argument(
        "--epochs", type=int, default=MAX_EPOCHS,
        help="Maximum number of training epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=BATCH_SIZE,
        help="Batch size for DataLoader (default: %(default)s)"
    )
    parser.add_argument(
        "--lr", type=float, default=LEARNING_RATE,
        help="Learning rate for Adam optimizer (default: %(default)s)"
    )
    parser.add_argument(
        "--patience", type=int, default=EARLY_STOPPING_PATIENCE,
        help="Early stopping patience in epochs (default: %(default)s)"
    )
    parser.add_argument(
        "--device", type=str, default="auto",
        help="Device: 'auto', 'cpu', 'cuda', 'mps' (default: %(default)s)"
    )
    parser.add_argument(
        "--cache-dir", type=str, default="results/cache",
        help="Directory with preprocessed .npy cache files (default: %(default)s)"
    )
    parser.add_argument(
        "--models-dir", type=str, default="results/models",
        help="Directory to save trained model weights (default: %(default)s)"
    )
    parser.add_argument(
        "--metrics-dir", type=str, default="results/metrics",
        help="Directory to save metrics CSV files (default: %(default)s)"
    )
    parser.add_argument(
        "--seed", type=int, default=SEED,
        help="Random seed for reproducibility (default: %(default)s)"
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def discover_folds(cache_dir: str) -> list[int]:
    """Discover available fold indices from cache files.

    Scans ``cache_dir`` for ``fold_{idx}_X.npy`` files and returns
    the sorted list of integer fold indices found.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        Sorted list of fold indices.
    """
    fold_indices: list[int] = []
    for fname in os.listdir(cache_dir):
        # Match patterns like "fold_0_X.npy" or "fold_10_X.npy"
        if fname.startswith("fold_") and fname.endswith("_X.npy"):
            parts = fname.split("_")
            if len(parts) >= 2:
                try:
                    idx = int(parts[1])
                    fold_indices.append(idx)
                except ValueError:
                    continue
    return sorted(set(fold_indices))


def load_test_data(cache_dir: str) -> tuple[np.ndarray, np.ndarray]:
    """Load holdout test set from cache.

    Args:
        cache_dir: Path to the cache directory.

    Returns:
        Tuple of ``(X_test, y_test)`` numpy arrays.
    """
    X_test = np.load(os.path.join(cache_dir, "test_X.npy"))
    y_test = np.load(os.path.join(cache_dir, "test_y.npy"))
    return X_test, y_test


def main():
    """Main training routine: 5-fold CV → ensemble → holdout evaluation."""
    args = parse_args()
    set_seed(args.seed)

    # Resolve device
    if args.device == "auto":
        device = get_device()
    else:
        device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    label_map = {orig: new for new, orig in enumerate(RETAINED_CLASSES)}

    # Header
    print("=" * 60)
    print("3W Oil Well Dataset - LSTM Training")
    print("=" * 60)
    print(f"  Device:     {device}")
    print(f"  Seed:       {args.seed}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Patience:   {args.patience}")
    print(f"  Cache dir:  {args.cache_dir}")
    print()

    # Ensure output directories exist
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(args.metrics_dir, exist_ok=True)

    # Discover available folds from cache
    fold_indices = discover_folds(args.cache_dir)
    if not fold_indices:
        print(f"ERROR: No fold cache files found in '{args.cache_dir}'.")
        print("  Expected files: fold_0_X.npy, fold_0_y.npy, fold_0_val_X.npy, ...")
        print("  Run 'python scripts/preprocess.py' first to generate cache.")
        sys.exit(1)

    print(f"Discovered {len(fold_indices)} fold(s): {fold_indices}")
    print()

    # ======================================================================
    # 1. 5-Fold Cross-Validation Training
    # ======================================================================
    fold_metrics: dict[int, dict] = {}
    all_val_preds: list[np.ndarray] = []
    all_val_true: list[np.ndarray] = []

    for fold_idx in fold_indices:
        print(f"\n{'─' * 60}")
        print(f"  Fold {fold_idx} / {len(fold_indices) - 1}")
        print(f"{'─' * 60}")

        # Build cache paths
        train_features_path = os.path.join(args.cache_dir, f"fold_{fold_idx}_X.npy")
        train_labels_path = os.path.join(args.cache_dir, f"fold_{fold_idx}_y.npy")
        val_features_path = os.path.join(args.cache_dir, f"fold_{fold_idx}_val_X.npy")
        val_labels_path = os.path.join(args.cache_dir, f"fold_{fold_idx}_val_y.npy")

        # Quick shape info
        X_train = np.load(train_features_path)
        y_train = np.load(train_labels_path)
        X_val = np.load(val_features_path)
        y_val = np.load(val_labels_path)
        print(f"    Train: {X_train.shape[0]} windows  |  Val: {X_val.shape[0]} windows")
        print(f"    Train classes: {dict(Counter(y_train.tolist()))}")

        # Create datasets and dataloaders
        train_dataset = OilWellDataset(train_features_path, train_labels_path)
        val_dataset = OilWellDataset(val_features_path, val_labels_path)

        # Remap labels (uses global label_map)
        train_dataset.labels = np.array([label_map[l] for l in train_dataset.labels])
        val_dataset.labels = np.array([label_map[l] for l in val_dataset.labels])

        # Load labels for balanced sampling
        train_labels = train_dataset.labels
        class_counts = np.bincount(train_labels, minlength=7)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # guard missing classes
        sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_dataset), replacement=True)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=sampler,
            drop_last=False,
            pin_memory=use_cuda,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=use_cuda,
            num_workers=0,
        )

        # Instantiate model and trainer
        model = LSTMModel()
        trainer = Trainer(
            model,
            device,
            config={"lr": args.lr, "patience": args.patience},
        )

        # Train
        t0 = time.time()
        history = trainer.fit(train_loader, val_loader, epochs=args.epochs)
        elapsed = time.time() - t0

        # Evaluate on validation set
        val_preds, val_probs = trainer.predict(val_loader)
        metrics = compute_metrics(y_val, val_preds, val_probs)
        metrics["train_time_sec"] = round(elapsed, 2)
        fold_metrics[fold_idx] = metrics

        all_val_preds.append(val_preds)
        all_val_true.append(y_val)

        # Print fold results
        print(f"\n    ── Fold {fold_idx} Results ──")
        print(f"      Accuracy:        {metrics['accuracy']:.4f}")
        print(f"      Weighted F1:     {metrics['weighted_f1']:.4f}")
        print(f"      Macro F1:        {metrics['macro_f1']:.4f}")
        print(f"      Last val loss:   {history['val_loss'][-1]:.4f}  "
              f"(best: {min(history['val_loss']):.4f})")
        print(f"      Epochs trained:  {len(history['train_loss'])}")
        print(f"      Training time:   {elapsed:.1f}s")

        # Save model weights
        model_path = os.path.join(args.models_dir, f"lstm_fold{fold_idx}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"    Model saved: {model_path}")

        # Save fold metrics
        metrics_path = os.path.join(
            args.metrics_dir, f"lstm_fold{fold_idx}_metrics.csv"
        )
        save_metrics_csv(metrics, metrics_path)
        print(f"    Metrics saved: {metrics_path}")

    # ======================================================================
    # 2. Cross-Validation Summary
    # ======================================================================
    print(f"\n{'=' * 60}")
    print("  CROSS-VALIDATION SUMMARY (LSTM)")
    print(f"{'=' * 60}")

    accuracies = [fold_metrics[f]["accuracy"] for f in fold_indices]
    weighted_f1s = [fold_metrics[f]["weighted_f1"] for f in fold_indices]
    macro_f1s = [fold_metrics[f]["macro_f1"] for f in fold_indices]

    print(f"    Accuracy:       {np.mean(accuracies):.4f}  ±  {np.std(accuracies):.4f}")
    print(f"    Weighted F1:    {np.mean(weighted_f1s):.4f}  ±  {np.std(weighted_f1s):.4f}")
    print(f"    Macro F1:       {np.mean(macro_f1s):.4f}  ±  {np.std(macro_f1s):.4f}")

    # ======================================================================
    # 3. Ensemble Evaluation on Holdout Test Set
    # ======================================================================
    test_path = os.path.join(args.cache_dir, "test_X.npy")
    if not os.path.exists(test_path):
        print(f"\n  NOTE: Test cache '{test_path}' not found. Skipping holdout evaluation.")
        print("  Run 'python scripts/preprocess.py' to generate test cache.")
    else:
        print(f"\n{'=' * 60}")
        print("  ENSEMBLE EVALUATION ON HOLDOUT TEST SET (LSTM)")
        print(f"{'=' * 60}")

        X_test, y_test = load_test_data(args.cache_dir)
        print(f"    Test samples: {X_test.shape[0]}")
        print(f"    Test classes: {dict(Counter(y_test.tolist()))}")

        test_dataset = OilWellDataset(
            os.path.join(args.cache_dir, "test_X.npy"),
            os.path.join(args.cache_dir, "test_y.npy"),
        )
        # Remap test labels
        test_dataset.labels = np.array([label_map[l] for l in test_dataset.labels])
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=use_cuda,
            num_workers=0,
        )

        # Ensemble: sum softmax probabilities from all fold models
        ensemble_probs = None
        models_loaded = 0

        for fold_idx in fold_indices:
            model_path = os.path.join(args.models_dir, f"lstm_fold{fold_idx}.pt")
            if not os.path.exists(model_path):
                print(f"    WARNING: Model '{model_path}' not found — skipping fold {fold_idx}")
                continue

            model = LSTMModel()
            state = torch.load(model_path, map_location=device, weights_only=True)
            model.load_state_dict(state)
            trainer = Trainer(model, device)

            _, probs = trainer.predict(test_loader)
            if ensemble_probs is None:
                ensemble_probs = probs
            else:
                ensemble_probs += probs
            models_loaded += 1

        if ensemble_probs is None or models_loaded == 0:
            print("    ERROR: No fold models could be loaded for ensemble.")
        else:
            # Average probabilities and take argmax
            ensemble_probs /= models_loaded
            ensemble_preds = np.argmax(ensemble_probs, axis=1)

            ensemble_metrics = compute_metrics(y_test, ensemble_preds, ensemble_probs)

            print(f"\n    ── Ensemble Results ──")
            print(f"      Models in ensemble: {models_loaded}")
            print(f"      Accuracy:           {ensemble_metrics['accuracy']:.4f}")
            print(f"      Weighted F1:        {ensemble_metrics['weighted_f1']:.4f}")
            print(f"      Macro F1:           {ensemble_metrics['macro_f1']:.4f}")

            # Per-class metrics
            print(f"\n    Per-class F1:")
            for cls_idx, f1_val in enumerate(ensemble_metrics["per_class_f1"]):
                print(f"      Class {cls_idx}: {f1_val:.4f}")

            # Save predictions
            predictions_path = os.path.join(
                args.metrics_dir, "lstm_predictions.npz"
            )
            np.savez(
                predictions_path,
                y_true=y_test,
                y_pred=ensemble_preds,
                y_proba=ensemble_probs,
                fold_accuracies=np.array(accuracies),
                fold_weighted_f1=np.array(weighted_f1s),
                fold_macro_f1=np.array(macro_f1s),
            )
            print(f"\n    Predictions saved: {predictions_path}")

            # Save ensemble metrics as CSV
            ensemble_metrics_path = os.path.join(
                args.metrics_dir, "lstm_ensemble_metrics.csv"
            )
            save_metrics_csv(ensemble_metrics, ensemble_metrics_path)
            print(f"    Ensemble metrics:  {ensemble_metrics_path}")

    # ======================================================================
    # 4. Summary
    # ======================================================================
    print(f"\n{'=' * 60}")
    print("  LSTM TRAINING COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Models:      {args.models_dir}/lstm_fold{{fold}}.pt")
    print(f"  Metrics:     {args.metrics_dir}/lstm_fold{{fold}}_metrics.csv")
    print(f"  Predictions: {args.metrics_dir}/lstm_predictions.npz")
    print()


if __name__ == "__main__":
    main()

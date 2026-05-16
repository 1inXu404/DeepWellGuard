#!/usr/bin/env python3
"""
CNN training script with 5-fold cross-validation.

Loads preprocessed cache files (fold_{0-4}_X/y.npy, fold_{0-4}_val_X/y.npy),
trains a CNNModel per fold via Trainer with early stopping, ensembles the
fold models via averaged softmax probabilities, and evaluates on the holdout
test set (test_X/y.npy).

Handles label remapping: original class labels [0,1,3,4,5,6,9] are mapped to
contiguous [0,1,2,3,4,5,6] for the CNN's 7-class output. Excluded classes
[2,7,8] are filtered out.

Usage:
    python scripts/train_cnn.py
    python scripts/train_cnn.py --epochs 50 --patience 10
"""

import argparse
import os
import random
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OilWellDataset
from src.models.cnn import CNNModel
from src.train.evaluate import compute_metrics
from src.train.trainer import Trainer
from src.utils.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    RETAINED_CLASSES,
    SEED,
)
from src.utils.device import get_device

# ---------------------------------------------------------------------------
# Label remapping helpers
# ---------------------------------------------------------------------------


def _build_label_map(labels: np.ndarray) -> np.ndarray:
    """Build a vectorised label-map array spanning 0 … max(labels).

    Returns an array ``m`` where ``m[original] == contiguous`` for every
    retained class and ``-1`` for any other label value.
    """
    max_label = max(int(labels.max()), max(RETAINED_CLASSES))
    m = np.full(max_label + 1, -1, dtype=np.int64)
    for new, orig in enumerate(RETAINED_CLASSES):
        m[orig] = new
    return m


def _prepare_dataset(x_path: str, y_path: str, label_map: np.ndarray) -> Subset:
    """Load an ``OilWellDataset``, remap labels in-place, filter excluded classes.

    Returns a ``Subset`` that only exposes samples whose original label was
    one of ``RETAINED_CLASSES``. The underlying ``ds.labels`` is remapped
    so that ``__getitem__`` returns the correct contiguous label.
    """
    ds = OilWellDataset(x_path, y_path)

    # Filter mask: keep only retained classes
    keep_mask = np.isin(ds.labels, RETAINED_CLASSES)
    keep_indices = np.where(keep_mask)[0]

    # Remap labels (creates a new array in RAM, avoiding read-only mmap modification)
    ds.labels = label_map[ds.labels]

    return Subset(ds, keep_indices)


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
        description="Train CNN with 5-fold cross-validation on 3W oil well data."
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
    args = parser.parse_args()

    # ── Set all seeds for reproducibility ──────────────────────────────
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = get_device()
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")
    print(f"Epochs: {args.epochs}  |  Patience: {args.patience}  |  Batch size: {args.batch_size}")
    print(f"Retained classes: {RETAINED_CLASSES} → contiguous 0..{len(RETAINED_CLASSES) - 1}")

    # Build label map from the first available fold (to determine max label)
    label_map: np.ndarray | None = None
    for fold in range(5):
        train_y_path = f"results/cache/fold_{fold}_y.npy"
        if os.path.isfile(train_y_path):
            label_map = _build_label_map(np.load(train_y_path))
            break
    if label_map is None:
        print("No cached fold data found. Run preprocess.py first.")
        sys.exit(1)

    # ── Single Fold Training (was 5-fold CV) ───────────────────────────────
    fold_accs: list[float] = []
    fold_models: list[CNNModel] = []

    for fold in range(1):
        if not _fold_cache_exists(fold):
            print(f"\n[SKIP] Fold {fold} — cache files not found.")
            continue

        print(f"\n{'=' * 50}")
        print(f"  Fold {fold}")
        print(f"{'=' * 50}")

        train_ds = _prepare_dataset(
            f"results/cache/fold_{fold}_X.npy",
            f"results/cache/fold_{fold}_y.npy",
            label_map,
        )
        val_ds = _prepare_dataset(
            f"results/cache/fold_{fold}_val_X.npy",
            f"results/cache/fold_{fold}_val_y.npy",
            label_map,
        )

        # Load labels for balanced sampling
        train_labels = np.load(f"results/cache/fold_{fold}_y.npy")
        class_counts = np.bincount(train_labels, minlength=7)
        class_counts = np.where(class_counts == 0, 1, class_counts)  # guard missing classes
        sample_weights = 1.0 / class_counts[train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

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

        model = CNNModel()
        trainer = Trainer(model, device)
        _ = trainer.fit(
            train_loader, val_loader, epochs=args.epochs, patience=args.patience
        )

        # Save model weights
        os.makedirs("results/models", exist_ok=True)
        ckpt_path = f"results/models/cnn_fold{fold}.pt"
        torch.save(model.state_dict(), ckpt_path)
        print(f"  Model saved → {ckpt_path}")

        fold_models.append(model)

        # Validation accuracy (labels already remapped in OilWellDataset)
        val_preds, _ = trainer.predict(val_loader)
        # Retrieve remapped labels via the Subset's indices
        val_labels = np.array(
            [val_ds.dataset.labels[i] for i in val_ds.indices]
        )
        val_acc = float((val_preds == val_labels).mean())
        fold_accs.append(val_acc)
        print(f"  Fold {fold} val accuracy: {val_acc:.4f}")

    # ── Summary ────────────────────────────────────────────────────────
    if fold_accs:
        mean_acc = float(np.mean(fold_accs))
        std_acc = float(np.std(fold_accs))
        print(f"\n{'=' * 50}")
        print(f"  Single fold CV accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
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

    test_ds = _prepare_dataset(test_X_path, test_y_path, label_map)
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

    # Collect ground-truth labels from the test Subset (already remapped)
    test_labels = np.array(
        [test_ds.dataset.labels[i] for i in test_ds.indices]
    )

    # Save predictions
    os.makedirs("results/metrics", exist_ok=True)
    np.savez(
        "results/metrics/cnn_predictions.npz",
        preds=test_preds,
        probs=avg_probs,
        labels=test_labels,
    )
    print("  Predictions saved → results/metrics/cnn_predictions.npz")

    # Compute and display metrics
    metrics = compute_metrics(test_labels, test_preds, avg_probs)
    print(f"\n  Test accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Per-class F1:     {[round(f, 4) for f in metrics['per_class_f1']]}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()

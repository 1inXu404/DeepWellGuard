"""Shared training/evaluation flow for pure baseline models."""

from dataclasses import dataclass
import argparse
import os
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

from src.data.dataset import OilWellDataset
from src.models.bilstm import BiLSTMModel
from src.models.cnn import CNNModel
from src.models.unilstm import UniLSTMModel
from src.train.evaluate import compute_metrics
from src.train.trainer import Trainer
from src.utils.config import (
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    RETAINED_CLASSES,
    SEED,
    set_global_seed,
)
from src.utils.device import get_device
from src.visualize.plots import plot_training_curves


@dataclass(frozen=True)
class BaselineSpec:
    """Metadata needed to train and persist one baseline."""

    factory: Callable[[], torch.nn.Module]
    display_name: str
    file_stem: str


BASELINE_MODELS = {
    "cnn": BaselineSpec(CNNModel, "CNNModel", "cnnmodel"),
    "unilstm": BaselineSpec(UniLSTMModel, "UniLSTMModel", "unilstmmodel"),
    "bilstm": BaselineSpec(BiLSTMModel, "BiLSTMModel", "bilstmmodel"),
}


def add_baseline_args(parser: argparse.ArgumentParser) -> None:
    """Add common baseline training arguments to ``parser``."""
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early-stopping patience")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to use")


def build_train_val_loaders(args: argparse.Namespace, use_cuda: bool):
    """Build a balanced training loader and fixed validation loader."""
    label_map = {orig: new for new, orig in enumerate(RETAINED_CLASSES)}

    train_ds = OilWellDataset("results/cache/fold_train_X.npy", "results/cache/fold_train_y.npy")
    val_ds = OilWellDataset("results/cache/fold_val_X.npy", "results/cache/fold_val_y.npy")

    train_ds.labels = np.array([label_map[lbl] for lbl in train_ds.labels])
    val_ds.labels = np.array([label_map[lbl] for lbl in val_ds.labels])

    if args.subset < 1.0:
        rng = np.random.default_rng(args.seed)
        train_indices = rng.choice(
            len(train_ds),
            max(1, int(len(train_ds) * args.subset)),
            replace=False,
        )
        val_indices = rng.choice(
            len(val_ds),
            max(1, int(len(val_ds) * args.subset)),
            replace=False,
        )
        train_ds = Subset(train_ds, train_indices)
        val_ds = Subset(val_ds, val_indices)

    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        generator=train_generator,
        pin_memory=use_cuda,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )
    return train_ds, val_ds, train_loader, val_loader


def build_test_loader(args: argparse.Namespace, use_cuda: bool):
    """Build the holdout test loader, or return ``None`` if cache is missing."""
    test_X_path = "results/cache/test_X.npy"
    test_y_path = "results/cache/test_y.npy"
    if not (os.path.isfile(test_X_path) and os.path.isfile(test_y_path)):
        return None

    label_map = {orig: new for new, orig in enumerate(RETAINED_CLASSES)}
    test_ds = OilWellDataset(test_X_path, test_y_path)
    test_ds.labels = np.array([label_map[lbl] for lbl in test_ds.labels])
    return DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )


def train_baseline(model_key: str, args: argparse.Namespace) -> None:
    """Train one baseline model using the shared project flow."""
    if model_key not in BASELINE_MODELS:
        available = ", ".join(sorted(BASELINE_MODELS))
        raise ValueError(f"Unknown baseline model '{model_key}'. Available: {available}")

    spec = BASELINE_MODELS[model_key]
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    models_dir = os.path.join("results", "models", run_id)
    metrics_dir = os.path.join("results", "metrics", run_id)
    figures_dir = os.path.join("results", "figures", run_id)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)
    print(f"Run ID: {run_id}")

    set_global_seed(args.seed)

    device = get_device()
    use_cuda = device.type == "cuda"
    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")
    print(f"Epochs: {args.epochs}  |  Patience: {args.patience}  |  Batch size: {args.batch_size}")

    print("\n" + "=" * 50)
    print(f"  Training {spec.display_name}")
    print("=" * 50)

    train_ds, val_ds, train_loader, val_loader = build_train_val_loaders(args, use_cuda)
    print(f"  Train samples: {len(train_ds)}  |  Val samples: {len(val_ds)}")

    model = spec.factory()
    trainer = Trainer(model, device)
    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    history_path = os.path.join(metrics_dir, f"{spec.file_stem}_training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index_label="epoch")
    curve_path = os.path.join(figures_dir, f"{spec.file_stem}_training_curves.png")
    plot_training_curves(history, spec.display_name, save_path=curve_path)
    print(f"  Training history saved -> {history_path}")
    print(f"  Training curves saved -> {curve_path}")

    ckpt_path = os.path.join(models_dir, f"{spec.file_stem}.pt")
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Model saved -> {ckpt_path}")

    print("\n" + "=" * 50)
    print("  Evaluation on holdout test set")
    print("=" * 50)

    test_loader = build_test_loader(args, use_cuda)
    if test_loader is None:
        print("\nTest cache files not found - skipping test evaluation.")
        return

    print(f"  Test samples: {len(test_loader.dataset)}")
    preds, probs, y_true = trainer.predict(test_loader)
    np.savez(
        os.path.join(metrics_dir, f"{spec.file_stem}_predictions.npz"),
        preds=preds,
        probs=probs,
        labels=y_true,
    )

    metrics = compute_metrics(y_true, preds, probs)
    print(f"\n  Test accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Per-class F1:     {[round(f, 4) for f in metrics['per_class_f1']]}")
    print("=" * 50)

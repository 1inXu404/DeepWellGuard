#!/usr/bin/env python3
"""Train and evaluate CNN-LSTM-Channel-Attention ablation variants.

Expected cache files:
    results/cache/fold_train_X.npy
    results/cache/fold_train_y.npy
    results/cache/fold_val_X.npy
    results/cache/fold_val_y.npy
    results/cache/test_X.npy
    results/cache/test_y.npy
"""

import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OilWellDataset  # noqa: E402
from src.models.ablation import (  # noqa: E402
    ABLATION_CONFIGS,
    AblationCNNLSTMAttention,
    get_ablation_config,
)
from src.train.evaluate import compute_metrics  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402
from src.visualize.plots import plot_comparison_bar, plot_training_curves  # noqa: E402
from src.utils.config import (  # noqa: E402
    BATCH_SIZE,
    EARLY_STOPPING_PATIENCE,
    MAX_EPOCHS,
    RETAINED_CLASSES,
    SEED,
    set_global_seed,
)
from src.utils.device import get_device  # noqa: E402


DEFAULT_VARIANTS = ["lstm_channel_attention", "cnn_lstm"]


def build_loaders(args, use_cuda: bool):
    """Build deterministic train/val/test DataLoaders."""
    label_map = {orig: new for new, orig in enumerate(RETAINED_CLASSES)}

    train_ds = OilWellDataset(
        "results/cache/fold_train_X.npy",
        "results/cache/fold_train_y.npy",
    )
    val_ds = OilWellDataset(
        "results/cache/fold_val_X.npy",
        "results/cache/fold_val_y.npy",
    )
    test_ds = OilWellDataset(
        "results/cache/test_X.npy",
        "results/cache/test_y.npy",
    )

    train_ds.labels = np.array([label_map[lbl] for lbl in train_ds.labels])
    val_ds.labels = np.array([label_map[lbl] for lbl in val_ds.labels])
    test_ds.labels = np.array([label_map[lbl] for lbl in test_ds.labels])

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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=use_cuda,
        num_workers=0,
    )
    return train_loader, val_loader, test_loader


def train_one_variant(variant: str, args, device: torch.device, run_dir: str) -> dict:
    """Train one ablation variant and return its holdout metrics."""
    set_global_seed(args.seed)
    use_cuda = device.type == "cuda"
    train_loader, val_loader, test_loader = build_loaders(args, use_cuda)

    config = get_ablation_config(variant)
    model = AblationCNNLSTMAttention(config)
    trainer = Trainer(model, device)

    print("\n" + "=" * 60)
    print(f"Ablation variant: {variant}")
    print("=" * 60)
    print(f"  Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    history = trainer.fit(
        train_loader,
        val_loader,
        epochs=args.epochs,
        patience=args.patience,
    )

    models_dir = os.path.join(run_dir, "models")
    metrics_dir = os.path.join(run_dir, "metrics")
    figures_dir = os.path.join(run_dir, "figures")
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(figures_dir, exist_ok=True)

    model_path = os.path.join(models_dir, f"{variant}.pt")
    torch.save(model.state_dict(), model_path)

    curve_path = os.path.join(figures_dir, f"{variant}_training_curves.png")
    plot_training_curves(history, variant, save_path=curve_path)

    preds, probs, y_true = trainer.predict(test_loader)
    npz_path = os.path.join(metrics_dir, f"{variant}_predictions.npz")
    np.savez(npz_path, preds=preds, probs=probs, labels=y_true)

    metrics = compute_metrics(y_true, preds, probs)
    metrics.update(
        {
            "variant": variant,
            "best_val_acc": max(history["val_acc"]) if history["val_acc"] else np.nan,
            "epochs_run": len(history["train_loss"]),
            "model_path": model_path,
            "predictions_path": npz_path,
        }
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for CNN-LSTM-Channel-Attention."
    )
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--subset", type=float, default=1.0)
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_VARIANTS,
        choices=sorted(ABLATION_CONFIGS.keys()),
        help="Ablation variants to train.",
    )
    args = parser.parse_args()

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("results", "ablation", run_id)
    os.makedirs(run_dir, exist_ok=True)

    set_global_seed(args.seed)
    device = get_device()
    print(f"Run ID: {run_id}")
    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")
    print(f"Variants: {', '.join(args.variants)}")

    rows = []
    for variant in args.variants:
        rows.append(train_one_variant(variant, args, device, run_dir))

    summary_path = os.path.join(run_dir, "ablation_summary.csv")
    summary = pd.DataFrame(rows)
    summary.to_csv(summary_path, index=False)

    figures_dir = os.path.join(run_dir, "figures")
    plot_df = summary.rename(columns={"variant": "model"})
    plot_comparison_bar(
        plot_df,
        os.path.join(figures_dir, "ablation_f1_comparison.png"),
    )

    display_cols = [
        "variant",
        "accuracy",
        "weighted_f1",
        "macro_f1",
        "best_val_acc",
        "epochs_run",
    ]
    print("\nAblation summary:")
    print(summary[display_cols].to_string(index=False))
    print(f"\nSaved summary -> {summary_path}")


if __name__ == "__main__":
    main()

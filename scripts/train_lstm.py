#!/usr/bin/env python3
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.dataset import OilWellDataset  # noqa: E402
from src.models.lstm import LSTMModel  # noqa: E402
from src.train.evaluate import compute_metrics  # noqa: E402
from src.train.trainer import Trainer  # noqa: E402
from src.visualize.plots import plot_training_curves  # noqa: E402
from src.utils.config import BATCH_SIZE, EARLY_STOPPING_PATIENCE, MAX_EPOCHS, RETAINED_CLASSES, SEED, set_global_seed  # noqa: E402
from src.utils.device import get_device  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train LSTMModel on 3W oil well data.")
    parser.add_argument("--epochs", type=int, default=MAX_EPOCHS, help="Max training epochs")
    parser.add_argument("--patience", type=int, default=EARLY_STOPPING_PATIENCE, help="Early-stopping patience")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Batch size")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed")
    parser.add_argument("--subset", type=float, default=1.0, help="Fraction of data to use")
    args = parser.parse_args()

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
    label_map = {orig: new for new, orig in enumerate(RETAINED_CLASSES)}
    print(f"Device: {device}")
    print(f"Seed:   {args.seed}")
    print(f"Epochs: {args.epochs}  |  Patience: {args.patience}  |  Batch size: {args.batch_size}")

    print(f"\n==================================================")
    print("  Training")
    print(f"==================================================")

    train_ds = OilWellDataset("results/cache/fold_train_X.npy", "results/cache/fold_train_y.npy")
    val_ds = OilWellDataset("results/cache/fold_val_X.npy", "results/cache/fold_val_y.npy")

    train_ds.labels = np.array([label_map[lbl] for lbl in train_ds.labels])
    val_ds.labels = np.array([label_map[lbl] for lbl in val_ds.labels])

    train_labels = train_ds.labels
    class_counts = np.bincount(train_labels, minlength=7)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_weights = 1.0 / np.sqrt(class_counts[train_labels])
    num_train_samples = int(len(train_ds) * args.subset)
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(args.seed)
    sampler = WeightedRandomSampler(
        sample_weights,
        num_train_samples,
        replacement=True,
        generator=sampler_generator,
    )

    if args.subset < 1.0:
        val_indices = np.random.choice(len(val_ds), int(len(val_ds) * args.subset), replace=False)
        val_ds = Subset(val_ds, val_indices)

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

    model = LSTMModel()
    trainer = Trainer(model, device)
    history = trainer.fit(
        train_loader, val_loader, epochs=args.epochs, patience=args.patience
    )

    history_path = os.path.join(metrics_dir, "lstmmodel_training_history.csv")
    pd.DataFrame(history).to_csv(history_path, index_label="epoch")
    curve_path = os.path.join(figures_dir, "lstmmodel_training_curves.png")
    plot_training_curves(history, "LSTMModel", save_path=curve_path)
    print(f"  Training history saved → {history_path}")
    print(f"  Training curves saved → {curve_path}")

    ckpt_path = f"{models_dir}/lstmmodel.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"  Model saved → {ckpt_path}")

    print("\n" + "=" * 50)
    print("  Evaluation on holdout test set")
    print("=" * 50)

    test_X_path = "results/cache/test_X.npy"
    test_y_path = "results/cache/test_y.npy"

    if not (os.path.isfile(test_X_path) and os.path.isfile(test_y_path)):
        print("\nTest cache files not found — skipping test evaluation.")
        return

    test_ds = OilWellDataset(test_X_path, test_y_path)
    test_ds.labels = np.array([label_map[lbl] for lbl in test_ds.labels])
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=use_cuda, num_workers=0,
    )
    print(f"  Test samples: {len(test_ds)}")

    _, probs, y_true = trainer.predict(test_loader)
    test_preds = np.argmax(probs, axis=1)

    np.savez(
        os.path.join(metrics_dir, "lstmmodel_predictions.npz"),
        preds=test_preds,
        probs=probs,
        labels=y_true,
    )

    metrics = compute_metrics(y_true, test_preds, probs)
    print(f"\n  Test accuracy:    {metrics['accuracy']:.4f}")
    print(f"  Weighted F1:      {metrics['weighted_f1']:.4f}")
    print(f"  Macro F1:         {metrics['macro_f1']:.4f}")
    print(f"  Per-class F1:     {[round(f, 4) for f in metrics['per_class_f1']]}")
    print(f"{'=' * 50}")


if __name__ == '__main__':
    main()

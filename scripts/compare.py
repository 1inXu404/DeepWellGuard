#!/usr/bin/env python3
"""Compare model predictions: load .npz files, compute metrics, save CSV + figures.

Reads prediction files from results/metrics/ for CNN, LSTM, and CNN-LSTM-Attention
models, computes classification metrics via src.train.evaluate, and produces:
  - results/metrics/comparison.csv   (metrics table)
  - results/figures/confusion_*.png   (per-model confusion matrices)
  - results/figures/comparison_bar.png (F1 bar chart)
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from src.train.evaluate import compute_metrics, generate_confusion_matrix  # noqa: E402
from src.utils.config import MAPPED_CLASS_NAMES  # noqa: E402
from src.visualize.plots import plot_comparison_bar  # noqa: E402


def _resolve_key(data, *candidates):
    """Return the first key found in the .npz archive."""
    for key in candidates:
        if key in data:
            return data[key]
    available = list(data.keys())
    raise KeyError(
        f"None of {candidates} found in .npz. Available keys: {available}"
    )


import argparse  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compare model predictions.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CNN", "Uni-LSTM", "Bi-LSTM", "CNN-LSTM-Attention"],
        help="Names of models to compare. Prediction files must be in results/metrics/[name]_predictions.npz"
    )
    args = parser.parse_args()

    # Map the model names to their expected prediction file paths
    # Handle the specific name mapping used in training scripts
    file_name_mapping = {
        "CNN": "cnnmodel_predictions.npz",
        "Uni-LSTM": "unilstmmodel_predictions.npz",
        "Bi-LSTM": "bilstmmodel_predictions.npz",
        "CNN-LSTM-Attention": "cnnlstmattention_predictions.npz"
    }

    models = []
    metrics_root = Path("results/metrics")
    for m in args.models:
        target_filename = file_name_mapping.get(m, f"{m.lower().replace(' ', '_').replace('-', '_')}_predictions.npz")

        # Find all matching prediction files in timestamp subdirectories
        found_files = list(metrics_root.rglob(target_filename))

        if not found_files:
            # Also check the root metrics folder just in case
            root_file = metrics_root / target_filename
            if root_file.exists():
                found_files = [root_file]

        if found_files:
            # Sort by path which will sort by timestamp folder names chronologically
            # and pick the last one (latest)
            latest_file = sorted(found_files)[-1]
            models.append((m, str(latest_file)))
        else:
            print(f"WARN: Could not find prediction file for {m} (looked for {target_filename})")

    os.makedirs("results/metrics", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    results = []

    for name, path in models:
        if not os.path.exists(path):
            print(f"WARN: {path} not found, skipping {name}")
            continue

        data = np.load(path)
        y_true = _resolve_key(data, "labels", "y_true")
        y_pred = _resolve_key(data, "preds", "y_pred")

        metrics = compute_metrics(y_true, y_pred)
        metrics["model"] = name
        results.append(metrics)

        fig_name = f"results/figures/confusion_{name.replace(' ', '_').replace('-', '_')}.png"
        generate_confusion_matrix(
            y_true,
            y_pred,
            save_path=fig_name,
            class_names=MAPPED_CLASS_NAMES,
        )

        print(
            f"{name}: acc={metrics['accuracy']:.4f}, "
            f"w_f1={metrics['weighted_f1']:.4f}, "
            f"m_f1={metrics['macro_f1']:.4f}"
        )

    if not results:
        print("No prediction files found. Exiting.")
        return

    df = pd.DataFrame(results)
    df.to_csv("results/metrics/comparison.csv", index=False)
    print(f"Saved results/metrics/comparison.csv ({len(df)} rows)")

    plot_comparison_bar(df, "results/figures/comparison_bar.png")
    print("Saved results/figures/comparison_bar.png")

    print("Done.")


if __name__ == "__main__":
    main()

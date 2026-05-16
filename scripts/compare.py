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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.train.evaluate import compute_metrics, generate_confusion_matrix
from src.visualize.plots import plot_comparison_bar


def _resolve_key(data, *candidates):
    """Return the first key found in the .npz archive."""
    for key in candidates:
        if key in data:
            return data[key]
    available = list(data.keys())
    raise KeyError(
        f"None of {candidates} found in .npz. Available keys: {available}"
    )


import argparse

def main():
    parser = argparse.ArgumentParser(description="Compare model predictions.")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["CNN", "LSTM", "CNN-LSTM-Attention"],
        help="Names of models to compare. Prediction files must be in results/metrics/[name]_predictions.npz"
    )
    args = parser.parse_args()

    # Map the model names to their expected prediction file paths
    # Handle the specific name mapping used in training scripts
    file_name_mapping = {
        "CNN": "cnn",
        "LSTM": "lstm",
        "CNN-LSTM-Attention": "cnn_lstm_attn"
    }

    models = []
    for m in args.models:
        # Resolve the actual filename base
        base_name = file_name_mapping.get(m, m.lower().replace(" ", "_").replace("-", "_"))
        path = f"results/metrics/{base_name}_predictions.npz"
        models.append((m, path))

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
        generate_confusion_matrix(y_true, y_pred, save_path=fig_name)

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

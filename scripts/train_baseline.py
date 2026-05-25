#!/usr/bin/env python3
"""Train one pure baseline model with the shared baseline flow."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.baseline_runner import BASELINE_MODELS, add_baseline_args, train_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a pure baseline model.")
    parser.add_argument(
        "--model",
        choices=sorted(BASELINE_MODELS),
        default="bilstm",
        help="Baseline model to train.",
    )
    add_baseline_args(parser)
    args = parser.parse_args()
    train_baseline(args.model, args)


if __name__ == "__main__":
    main()

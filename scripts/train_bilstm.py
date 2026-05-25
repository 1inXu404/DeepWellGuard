#!/usr/bin/env python3
"""Train the pure Bi-LSTM baseline on 3W oil well data."""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.train.baseline_runner import add_baseline_args, train_baseline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Train BiLSTMModel on 3W oil well data.")
    add_baseline_args(parser)
    args = parser.parse_args()
    train_baseline("bilstm", args)


if __name__ == "__main__":
    main()

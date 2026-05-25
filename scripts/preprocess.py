#!/usr/bin/env python3
"""
Preprocessing cache script for 3W oil well dataset.
Performs: stratified holdout -> train/val split -> per-file preprocessing -> numpy cache.
"""

import sys
import os
import json
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import list_files_by_class, stratified_holdout_split, separate_holdout_files, split_from_existing_holdout  # noqa: E402
from src.data.preprocessor import preprocess_single_file, save_preprocessed  # noqa: E402
from src.utils.config import SEED, HOLDOUT_RATIO  # noqa: E402


def main():
    print("=" * 60)
    print("3W Oil Well Dataset - Preprocessing Cache")
    print("=" * 60)

    # Step 1: List files by class
    print("\n[1/4] Listing files by class...")
    files_by_class = list_files_by_class()
    total_files = sum(len(v) for v in files_by_class.values())
    print(f"  Found {total_files} files across {len(files_by_class)} classes: "
          f"{ {k: len(v) for k, v in files_by_class.items()} }")

    # Step 2: Reuse existing holdout if present; otherwise create one.
    print("\n[2/4] Preparing holdout split...")
    existing_split = split_from_existing_holdout(files_by_class)
    if existing_split is not None:
        train_val_files, test_files = existing_split
        print("  Reusing existing holdout files from data/holdout_test")
    else:
        train_val_files, test_files = stratified_holdout_split(
            files_by_class, holdout_ratio=HOLDOUT_RATIO, seed=SEED
        )
        separate_holdout_files(train_val_files, test_files)
        print("  Created new stratified holdout split")

    print(f"  Train+Val: {sum(len(v) for v in train_val_files.values())} files")
    print(f"  Test:  {sum(len(v) for v in test_files.values())} files")

    # Step 3: Train/Val split
    print("\n[3/4] Creating Train/Val split...")
    all_train_val_paths = []
    all_train_val_labels = []
    for class_label in sorted(train_val_files.keys()):
        for path in sorted(train_val_files[class_label]):
            all_train_val_paths.append(path)
            all_train_val_labels.append(class_label)

    # 80/20 split of the remaining training files
    train_paths, val_paths = train_test_split(
        all_train_val_paths, test_size=0.2, stratify=all_train_val_labels, random_state=SEED
    )

    # Helper function to process a list of paths
    def process_paths(paths, desc):
        X_list, y_list = [], []
        skipped = 0
        for path in tqdm(paths, desc=desc, leave=False):
            try:
                X, y = preprocess_single_file(path)
                if len(X) > 0:
                    X_list.append(X)
                    y_list.append(y)
                else:
                    skipped += 1
            except Exception as e:
                print(f"    WARN: Skipping {os.path.basename(path)}: {e}")
                skipped += 1

        if X_list:
            X_out = np.concatenate(X_list, axis=0)
            y_out = np.concatenate(y_list, axis=0)
        else:
            X_out, y_out = np.empty((0, 22, 120)), np.empty((0,))

        return X_out, y_out, skipped

    print("\n[4/4] Processing files and caching...")

    # Process train
    X_train, y_train, skipped_train = process_paths(train_paths, "  Train")
    save_preprocessed('train', X_train, y_train)
    train_classes = Counter(y_train.tolist())
    print(f"  Train: {len(y_train)} windows, skipped {skipped_train} files")

    # Process val
    X_val, y_val, skipped_val = process_paths(val_paths, "  Val")
    save_preprocessed('val', X_val, y_val)
    val_classes = Counter(y_val.tolist())
    print(f"  Val: {len(y_val)} windows, skipped {skipped_val} files")

    # Process test
    X_test, y_test, skipped_test = process_paths([p for cls in test_files.values() for p in cls], "  Test")
    cache_dir = os.path.join("results", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, "test_X.npy"), X_test)
    np.save(os.path.join(cache_dir, "test_y.npy"), y_test)
    test_classes = Counter(y_test.tolist())
    print(f"  Test: {len(y_test)} windows, skipped {skipped_test} files")

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print("\nCache directory: results/cache/")

    print(f"Train class distribution: {dict(train_classes)}")
    print(f"Val class distribution: {dict(val_classes)}")
    print(f"Test class distribution: {dict(test_classes)}")


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Preprocessing cache script for 3W oil well dataset.
Performs: stratified holdout → 5-fold CV → per-file preprocessing → numpy cache.
"""

import sys
import os
import time
from collections import Counter

import numpy as np
from tqdm import tqdm
from sklearn.model_selection import StratifiedKFold

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.loader import list_files_by_class, stratified_holdout_split, separate_holdout_files
from src.data.preprocessor import preprocess_single_file, save_preprocessed
from src.utils.config import SEED, HOLDOUT_RATIO


def main():
    print("=" * 60)
    print("3W Oil Well Dataset - Preprocessing Cache")
    print("=" * 60)

    # Step 1: List files by class
    print("\n[1/5] Listing files by class...")
    files_by_class = list_files_by_class()
    total_files = sum(len(v) for v in files_by_class.values())
    print(f"  Found {total_files} files across {len(files_by_class)} classes: "
          f"{ {k: len(v) for k, v in files_by_class.items()} }")

    # Step 2: Stratified holdout split
    print("\n[2/5] Stratified holdout split...")
    train_files, test_files = stratified_holdout_split(
        files_by_class, holdout_ratio=HOLDOUT_RATIO, seed=SEED
    )
    print(f"  Train: {sum(len(v) for v in train_files.values())} files")
    print(f"  Test:  {sum(len(v) for v in test_files.values())} files")

    # Step 3: Separate holdout files to disk
    print("\n[3/5] Separating holdout test files...")
    separate_holdout_files(train_files, test_files)

    # Step 4: 5-fold CV split on train files
    print("\n[4/5] Creating 5-fold CV splits...")
    # Flatten train files into arrays for StratifiedKFold
    all_train_paths = []
    all_train_labels = []
    for class_label in sorted(train_files.keys()):
        for path in sorted(train_files[class_label]):
            all_train_paths.append(path)
            all_train_labels.append(class_label)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    fold_stats = {}
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(all_train_paths, all_train_labels)):
        print(f"\n  Processing Fold {fold_idx}...")

        # Get train and val paths for this fold
        fold_train_paths = [all_train_paths[i] for i in train_idx]
        fold_val_paths = [all_train_paths[i] for i in val_idx]

        # Process train files
        X_train_list, y_train_list = [], []
        skipped_train = 0
        for path in tqdm(fold_train_paths, desc=f"  Fold {fold_idx} Train", leave=False):
            try:
                X, y = preprocess_single_file(path)
                if len(X) > 0:
                    X_train_list.append(X)
                    y_train_list.append(y)
                else:
                    skipped_train += 1
            except Exception as e:
                print(f"    WARN: Skipping {os.path.basename(path)}: {e}")
                skipped_train += 1

        if X_train_list:
            X_train = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            # Save train for this fold as fold_{fold_idx}_X/y.npy
            save_preprocessed(fold_idx, X_train, y_train)
            train_classes = Counter(y_train.tolist())
        else:
            X_train, y_train = np.empty((0, 22, 120)), np.empty((0,))
            train_classes = {}

        # Process val files - save as fold_{fold_idx}_val_X/y.npy
        X_val_list, y_val_list = [], []
        skipped_val = 0
        for path in tqdm(fold_val_paths, desc=f"  Fold {fold_idx} Val", leave=False):
            try:
                X, y = preprocess_single_file(path)
                if len(X) > 0:
                    X_val_list.append(X)
                    y_val_list.append(y)
                else:
                    skipped_val += 1
            except Exception as e:
                print(f"    WARN: Skipping {os.path.basename(path)}: {e}")
                skipped_val += 1

        if X_val_list:
            X_val = np.concatenate(X_val_list, axis=0)
            y_val = np.concatenate(y_val_list, axis=0)
            save_preprocessed(f'{fold_idx}_val', X_val, y_val)
            val_classes = Counter(y_val.tolist())
        else:
            X_val, y_val = np.empty((0, 22, 120)), np.empty((0,))
            val_classes = {}

        fold_stats[fold_idx] = {
            'train_samples': len(y_train),
            'val_samples': len(y_val),
            'train_classes': dict(train_classes),
            'val_classes': dict(val_classes),
            'skipped_train': skipped_train,
            'skipped_val': skipped_val
        }

        print(f"    Fold {fold_idx}: Train={len(y_train)} windows, Val={len(y_val)} windows")

    # Step 5: Process holdout test set
    print("\n[5/5] Processing holdout test set...")
    test_paths = []
    for class_label in sorted(test_files.keys()):
        for path in sorted(test_files[class_label]):
            test_paths.append(path)

    X_test_list, y_test_list = [], []
    skipped_test = 0
    for path in tqdm(test_paths, desc="  Test", leave=False):
        try:
            X, y = preprocess_single_file(path)
            if len(X) > 0:
                X_test_list.append(X)
                y_test_list.append(y)
            else:
                skipped_test += 1
        except Exception as e:
            print(f"    WARN: Skipping {os.path.basename(path)}: {e}")
            skipped_test += 1

    if X_test_list:
        X_test = np.concatenate(X_test_list, axis=0)
        y_test = np.concatenate(y_test_list, axis=0)
        # Save test cache WITHOUT 'fold_' prefix (requirement: test_X.npy / test_y.npy)
        cache_dir = os.path.join("results", "cache")
        os.makedirs(cache_dir, exist_ok=True)
        np.save(os.path.join(cache_dir, "test_X.npy"), X_test)
        np.save(os.path.join(cache_dir, "test_y.npy"), y_test)
        test_classes = Counter(y_test.tolist())
    else:
        test_classes = {}

    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nCache directory: results/cache/")

    # Check files
    expected_files = [f'fold_{i}_X.npy' for i in range(5)] + \
                     [f'fold_{i}_val_X.npy' for i in range(5)] + \
                     ['test_X.npy']
    existing_cache = os.listdir('results/cache')
    for fname in expected_files:
        exists = '✅' if fname in existing_cache else '❌'
        print(f"  {exists} {fname}")

    # Print per-fold stats
    print("\nPer-fold statistics:")
    print(f"{'Fold':<6} {'Train':<8} {'Val':<8} {'Train classes':<30} {'Skipped':<8}")
    print("-" * 60)
    for fold_id, stats in fold_stats.items():
        classes_str = str(stats['train_classes']) if stats['train_classes'] else '{}'
        print(f"{fold_id:<6} {stats['train_samples']:<8} {stats['val_samples']:<8} "
              f"{classes_str:<30} {stats['skipped_train'] + stats['skipped_val']:<8}")

    print(f"\nTest samples: {sum(len(v) for v in y_test_list) if y_test_list else 0}")
    print(f"Test class distribution: {dict(test_classes)}")


if __name__ == '__main__':
    main()

"""Data loading utilities for the 3W dataset.

Scans the 3w_dataset_2.0.0/ directory tree, groups .parquet files by class,
performs stratified holdout splitting, and physically separates test files
into a dedicated holdout directory.
"""

import json
import os
import random
import shutil
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from src.utils.config import EXCLUDED_CLASSES, HOLDOUT_RATIO, MAX_FILES_PER_CLASS, SEED

DATASET_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "3w_dataset_2.0.0")


def list_files_by_class(
    exclude_classes: Optional[List[int]] = None,
    dataset_root: str = DATASET_ROOT,
) -> Dict[int, List[str]]:
    """Scan dataset directory and list all .parquet files grouped by class.

    The dataset follows this layout::

        3w_dataset_2.0.0/
            0/WELL-*.parquet
            1/SIMULATED_*.parquet, WELL-*.parquet
            3/...
            ...

    Only subdirectories whose name is a single integer are considered;
    directories like ``folds/`` are silently skipped.

    Args:
        exclude_classes: Class labels to exclude. Defaults to ``EXCLUDED_CLASSES``.
        dataset_root: Path to the root of the 3W dataset.

    Returns:
        Dictionary mapping ``class_label`` → ``list of absolute .parquet paths``,
        sorted by class label.
    """
    if exclude_classes is None:
        exclude_classes = EXCLUDED_CLASSES
    exclude_set = set(exclude_classes)

    files_by_class: Dict[int, List[str]] = defaultdict(list)

    for entry in sorted(os.scandir(dataset_root), key=lambda e: e.name):
        if not entry.is_dir():
            continue
        # Only process folders named by a single integer
        try:
            class_label = int(entry.name)
        except ValueError:
            continue

        if class_label in exclude_set:
            continue

        parquet_files = [
            os.path.join(entry.path, f)
            for f in os.listdir(entry.path)
            if f.endswith(".parquet")
        ]
        if parquet_files:
            files_by_class[class_label].extend(sorted(parquet_files))

    return dict(sorted(files_by_class.items()))


def stratified_holdout_split(
    files_by_class: Dict[int, List[str]],
    holdout_ratio: float = HOLDOUT_RATIO,
    seed: int = SEED,
) -> Tuple[Dict[int, List[str]], Dict[int, List[str]]]:
    """Per-class stratified random train/test split.

    For each class, a fraction ``holdout_ratio`` of the files is randomly
    selected as the test (holdout) set. The remaining files form the training
    set. This guarantees that every retained class is represented in both
    splits with the same proportion.

    Args:
        files_by_class: Mapping ``class → file_paths``, e.g. from
            :func:`list_files_by_class`.
        holdout_ratio: Fraction of files to reserve for testing (per class).
        seed: Random seed for reproducibility.

    Returns:
        A 2-tuple ``(train_files_by_class, test_files_by_class)``.
    """
    rng = random.Random(seed)

    train_dict: Dict[int, List[str]] = {}
    test_dict: Dict[int, List[str]] = {}

    for class_label, file_paths in sorted(files_by_class.items()):
        # Shuffle deterministically before slicing
        ordered = sorted(file_paths)  # stable order before shuffle
        rng.shuffle(ordered)

        n_test = max(1, round(len(ordered) * holdout_ratio))
        test_paths = ordered[:n_test]
        train_paths = ordered[n_test:]

        train_dict[class_label] = sorted(train_paths)
        test_dict[class_label] = sorted(test_paths)

    return train_dict, test_dict


def limit_files_per_class(
    files_by_class: Dict[int, List[str]],
    max_files_per_class: int = MAX_FILES_PER_CLASS,
    seed: int = SEED,
) -> Dict[int, List[str]]:
    """Limit each class to at most ``max_files_per_class`` files.

    The selection is random but deterministic, controlled by ``seed``. Classes
    with fewer files are kept as-is.
    """
    rng = random.Random(seed)
    limited: Dict[int, List[str]] = {}

    for class_label, paths in sorted(files_by_class.items()):
        ordered = sorted(paths)
        if len(ordered) > max_files_per_class:
            selected = rng.sample(ordered, max_files_per_class)
        else:
            selected = ordered
        limited[class_label] = sorted(selected)

    return limited


def split_from_existing_holdout(
    files_by_class: Dict[int, List[str]],
    holdout_dir: str = "data/holdout_test",
) -> Optional[Tuple[Dict[int, List[str]], Dict[int, List[str]]]]:
    """Reuse an existing holdout directory as the fixed test split.

    If ``holdout_dir`` contains files under ``class_<N>/`` directories, those
    files are used as the test split. The train split is built from
    ``files_by_class`` after excluding files with the same basename in the same
    class. This prevents rerunning preprocessing from silently creating a new
    holdout split.

    Args:
        files_by_class: Mapping from original class label to source dataset
            file paths.
        holdout_dir: Existing holdout root, usually ``data/holdout_test``.

    Returns:
        ``(train_files_by_class, test_files_by_class)`` when reusable holdout
        files exist, otherwise ``None``.
    """
    holdout_dir = os.path.abspath(holdout_dir)
    if not os.path.isdir(holdout_dir):
        return None

    test_files: Dict[int, List[str]] = defaultdict(list)
    for entry in sorted(os.scandir(holdout_dir), key=lambda e: e.name):
        if not entry.is_dir() or not entry.name.startswith("class_"):
            continue
        try:
            class_label = int(entry.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        if class_label not in files_by_class:
            continue

        parquet_files = [
            os.path.join(entry.path, f)
            for f in os.listdir(entry.path)
            if f.endswith(".parquet")
        ]
        if parquet_files:
            test_files[class_label].extend(sorted(parquet_files))

    if not test_files:
        return None

    test_basenames = {
        class_label: {os.path.basename(path) for path in paths}
        for class_label, paths in test_files.items()
    }

    train_files: Dict[int, List[str]] = {}
    for class_label, paths in sorted(files_by_class.items()):
        excluded = test_basenames.get(class_label, set())
        train_files[class_label] = [
            path for path in sorted(paths)
            if os.path.basename(path) not in excluded
        ]
        test_files.setdefault(class_label, [])

    return dict(sorted(train_files.items())), dict(sorted(test_files.items()))


def separate_holdout_files(
    train_files: Dict[int, List[str]],
    test_files: Dict[int, List[str]],
    output_dir: str = "data/holdout_test",
    seed: int = SEED,
    holdout_ratio: float = HOLDOUT_RATIO,
    clear_existing: bool = True,
) -> None:
    """Physically copy test .parquet files into a dedicated holdout directory.

    Files are organised as ``output_dir/class_<N>/<original_filename>.parquet``.
    The original dataset is left untouched (copies, not moves).

    A manifest ``split_manifest.json`` is saved alongside the copied files
    recording the split parameters and the file mapping.

    Args:
        train_files: Training split (used only for manifest completeness).
        test_files: Test split whose files will be copied.
        output_dir: Destination root for the holdout tree.
        seed: Random seed recorded in the manifest.
        holdout_ratio: Holdout ratio recorded in the manifest.
        clear_existing: Remove old holdout files before writing the new split.
    """
    output_dir = os.path.abspath(output_dir)
    if clear_existing and os.path.isdir(output_dir):
        shutil.rmtree(output_dir)

    manifest: dict = {
        "seed": seed,
        "holdout_ratio": holdout_ratio,
        "test_files": {},
        "train_files": {},
    }

    for class_label, paths in test_files.items():
        class_dir = os.path.join(output_dir, f"class_{class_label}")
        os.makedirs(class_dir, exist_ok=True)

        filenames: List[str] = []
        for src_path in paths:
            dst_path = os.path.join(class_dir, os.path.basename(src_path))
            shutil.copy2(src_path, dst_path)
            filenames.append(os.path.basename(src_path))

        manifest["test_files"][str(class_label)] = filenames

    for class_label, paths in train_files.items():
        manifest["train_files"][str(class_label)] = [
            os.path.basename(p) for p in paths
        ]

    manifest_path = os.path.join(output_dir, "split_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Holdout test files copied to: {output_dir}")
    print(f"Manifest saved to: {manifest_path}")


def get_class_distribution(
    files_dict: Dict[int, List[str]],
) -> Dict[int, int]:
    """Return the number of files per class.

    Args:
        files_dict: Mapping ``class → file_paths``.

    Returns:
        ``{class_label: file_count}`` sorted by class label.
    """
    return {k: len(v) for k, v in sorted(files_dict.items())}

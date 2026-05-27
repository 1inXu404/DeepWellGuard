"""Tests for data pipeline: loader + preprocessor."""

import glob
from pathlib import Path

import numpy as np
import pytest

from src.data.loader import (
    limit_files_per_class,
    list_files_by_class,
    split_from_existing_holdout,
    stratified_holdout_split,
)
from src.data.preprocessor import preprocess_single_file
from src.utils.config import N_FEATURES, WINDOW_SIZE


class TestDataLoader:
    """Tests for loader functions (list_files_by_class, stratified_holdout_split)."""

    def test_lists_7_classes(self):
        """list_files_by_class should return exactly the 7 retained classes."""
        files = list_files_by_class()
        classes = sorted(files.keys())
        assert classes == [0, 1, 3, 4, 5, 6, 9]

    def test_holdout_ratio(self):
        """stratified_holdout_split with holdout_ratio=0.15 should yield ~15% test."""
        files = list_files_by_class()
        train, test = stratified_holdout_split(files, holdout_ratio=0.15, seed=42)
        total = sum(len(v) for v in files.values())
        test_total = sum(len(v) for v in test.values())
        assert 0.10 * total < test_total < 0.20 * total

    def test_existing_holdout_split_reuses_files(self, tmp_path):
        """Existing holdout files should define test split and be excluded from train."""
        dataset_file = tmp_path / "dataset" / "0" / "WELL-001.parquet"
        dataset_file.parent.mkdir(parents=True)
        dataset_file.write_text("placeholder")

        other_file = tmp_path / "dataset" / "0" / "WELL-002.parquet"
        other_file.write_text("placeholder")

        holdout_file = tmp_path / "holdout" / "class_0" / "WELL-001.parquet"
        holdout_file.parent.mkdir(parents=True)
        holdout_file.write_text("placeholder")

        train, test = split_from_existing_holdout(
            {0: [str(dataset_file), str(other_file)]},
            holdout_dir=str(tmp_path / "holdout"),
        )

        assert test[0] == [str(holdout_file)]
        assert train[0] == [str(other_file)]
        assert Path(train[0][0]).name != Path(test[0][0]).name

    def test_limit_files_per_class_caps_each_class(self):
        files = {0: [f"class0_{i}.parquet" for i in range(10)]}
        limited = limit_files_per_class(files, max_files_per_class=4, seed=42)
        assert len(limited[0]) == 4
        assert limited == limit_files_per_class(files, max_files_per_class=4, seed=42)

    def test_limit_then_holdout_split_uses_balanced_class_count(self):
        files = {0: [f"class0_{i}.parquet" for i in range(20)]}
        limited = limit_files_per_class(files, max_files_per_class=10, seed=42)
        train, test = stratified_holdout_split(
            limited,
            holdout_ratio=0.15,
            seed=42,
        )

        assert len(limited[0]) == 10
        assert len(test[0]) == 2
        assert len(train[0]) == 8


class TestPreprocessor:
    """Tests for preprocess_single_file."""

    def test_output_shape(self):
        """Preprocessed features must be 3-D with configured trailing dims."""
        files = sorted(glob.glob("3w_dataset_2.0.0/5/SIMULATED_*.parquet"))
        if not files:
            pytest.skip("No SIMULATED parquet files found for class 5")
        X, y = preprocess_single_file(files[0])
        assert X.ndim == 3
        assert X.shape[1:] == (N_FEATURES, WINDOW_SIZE)

    def test_no_nan(self):
        """No NaN values should survive preprocessing."""
        files = sorted(glob.glob("3w_dataset_2.0.0/5/SIMULATED_*.parquet"))
        if not files:
            pytest.skip("No SIMULATED parquet files found for class 5")
        X, y = preprocess_single_file(files[0])
        assert not np.isnan(X).any()

    def test_all_normal_class(self):
        """A file from class 0 should have every window label equal to 0."""
        files = sorted(glob.glob("3w_dataset_2.0.0/0/WELL-*.parquet"))
        if not files:
            pytest.skip("No WELL parquet files found for class 0")
        X, y = preprocess_single_file(files[0])
        assert set(y.tolist()) == {0}

"""Tests for data pipeline: loader + preprocessor."""

import glob

import numpy as np
import pytest

from src.data.loader import list_files_by_class, stratified_holdout_split
from src.data.preprocessor import preprocess_single_file


class TestDataLoader:
    """Tests for loader functions (list_files_by_class, stratified_holdout_split)."""

    def test_lists_7_classes(self):
        """list_files_by_class should return exactly the 7 retained classes."""
        files = list_files_by_class()
        classes = sorted(files.keys())
        assert classes == [0, 1, 3, 4, 5, 6, 9]

    def test_holdout_ratio(self):
        """stratified_holdout_split with holdout_ratio=0.2 should yield ~20% test
        files (within a 15-25% window)."""
        files = list_files_by_class()
        train, test = stratified_holdout_split(files, holdout_ratio=0.2, seed=42)
        total = sum(len(v) for v in files.values())
        test_total = sum(len(v) for v in test.values())
        assert 0.15 * total < test_total < 0.25 * total


class TestPreprocessor:
    """Tests for preprocess_single_file."""

    def test_output_shape(self):
        """Preprocessed features must be 3-D with trailing dims (22, 120)."""
        files = sorted(glob.glob("3w_dataset_2.0.0/5/SIMULATED_*.parquet"))
        if not files:
            pytest.skip("No SIMULATED parquet files found for class 5")
        X, y = preprocess_single_file(files[0])
        assert X.ndim == 3
        assert X.shape[1:] == (22, 120)

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

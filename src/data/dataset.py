"""PyTorch Dataset wrapper for the preprocessed 3W oil well data.

Reads preprocessed feature and label arrays from NumPy ``.npy`` cache files
produced by the preprocessing pipeline (Task 4).
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class OilWellDataset(Dataset):
    """PyTorch Dataset backed by preprocessed NumPy cache files.

    Expected input files (created by the preprocessing stage)::

        data/processed/train/
            X_train.npy    – shape (N_train, 22, 120), float32
            y_train.npy    – shape (N_train,),           int64
        data/processed/test/
            X_test.npy     – shape (N_test, 22, 120),  float32
            y_test.npy     – shape (N_test,),            int64

    Each call to ``__getitem__`` returns an ``(x, y)`` tuple of PyTorch
    tensors:

    * ``x``: ``torch.float32`` tensor of shape ``(n_features, window_size)``
    * ``y``: ``torch.long`` scalar tensor (class label in [0, 6))

    Parameters
    ----------
    features_path : str or Path
        Path to the ``.npy`` file containing the feature windows.
    labels_path : str or Path
        Path to the ``.npy`` file containing the integer labels.
    """

    def __init__(self, features_path: Path | str, labels_path: Path | str):
        features_path = Path(features_path)
        labels_path = Path(labels_path)

        if not features_path.exists():
            raise FileNotFoundError(f"Features file not found: {features_path}")
        if not labels_path.exists():
            raise FileNotFoundError(f"Labels file not found: {labels_path}")

        self.features: np.ndarray = np.load(str(features_path), mmap_mode='r')  # (N, 22, 120)
        self.labels: np.ndarray = np.load(str(labels_path), mmap_mode='r')  # (N,)

        if len(self.features) != len(self.labels):
            raise ValueError(
                f"Feature/label length mismatch: "
                f"{len(self.features)} vs {len(self.labels)}"
            )

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.features[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @property
    def n_features(self) -> int:
        """Number of sensor channels (22)."""
        return self.features.shape[1]

    @property
    def window_size(self) -> int:
        """Number of time steps per window (120)."""
        return self.features.shape[2]

    @property
    def n_classes(self) -> int:
        """Number of unique class labels."""
        return len(np.unique(self.labels))

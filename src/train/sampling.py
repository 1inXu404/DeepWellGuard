"""Sampling helpers shared by training entrypoints."""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Subset, WeightedRandomSampler

from src.utils.config import N_CLASSES


def labels_from_dataset(dataset) -> np.ndarray:
    """Return labels for an OilWellDataset or a Subset wrapping one."""
    if isinstance(dataset, Subset):
        labels = labels_from_dataset(dataset.dataset)
        return np.asarray(labels)[np.asarray(dataset.indices)]
    return np.asarray(dataset.labels)


def make_sqrt_balanced_sampler(
    dataset,
    num_samples: int | None = None,
    seed: int = 42,
    num_classes: int = N_CLASSES,
) -> WeightedRandomSampler:
    """Create the project's sqrt inverse-frequency sampler for a dataset."""
    labels = labels_from_dataset(dataset)
    class_counts = np.bincount(labels, minlength=num_classes)
    class_counts = np.where(class_counts == 0, 1, class_counts)
    sample_weights = 1.0 / np.sqrt(class_counts[labels])

    generator = torch.Generator()
    generator.manual_seed(seed)
    return WeightedRandomSampler(
        sample_weights,
        len(labels) if num_samples is None else num_samples,
        replacement=True,
        generator=generator,
    )

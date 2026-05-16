"""Data preprocessing pipeline for the 3W oil well dataset.

Loads raw Parquet files, applies cleanup + Z-score normalization + label
handling via ThreeWToolkit, windows the time series (window_size=120,
overlap=0.5), aggregates per-window labels via mode, and reshapes into
the (n_windows, 22, 120) tensor format required by the CNN/LSTM models.

Cache support: preprocessed arrays can be saved to and loaded from
``results/cache/`` as compressed ``.npy`` files.
"""

import os
import warnings
from typing import Tuple

import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode

from ThreeWToolkit.preprocessing import Windowing, WindowingConfig
from ThreeWToolkit.utils.data_utils import default_data_processing

from src.utils.config import OVERLAP, WINDOW_SIZE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MIN_FILE_LENGTH = 60  # skip files shorter than this (in time steps)
DEFAULT_CACHE_DIR = os.path.join("results", "cache")


# ---------------------------------------------------------------------------
# Core preprocessing
# ---------------------------------------------------------------------------
def preprocess_single_file(file_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single Parquet file and apply the full preprocessing pipeline.

    Pipeline steps
    ---------------
    1. Load raw data from Parquet via ``pandas.read_parquet``.
    2. Separate signal columns (all sensor columns except ``class`` and
       ``state``) from the ``class`` label column.
    3. Apply ``default_data_processing()`` for cleanup, Z-score
       normalisation, transient-label mapping, and NaN-filling.
    4. Apply ``Windowing`` (window_size=120, overlap=0.5,
       pad_last_window=True, boxcar window) to both signals and labels.
    5. For each label window take the **mode** (most frequent class) as the
       window-level label.
    6. Reshape the flat signal windows from ``(n_windows, 22*120)`` into
       ``(n_windows, 22, 120)``.

    Parameters
    ----------
    file_path : str
        Absolute or relative path to a ``.parquet`` file from the 3W dataset.

    Returns
    -------
    X : np.ndarray
        Float32 array of shape ``(n_windows, 22, 120)``.
    y : np.ndarray
        Int64 array of shape ``(n_windows,)`` — per-window class labels.

    Warns
    -----
    UserWarning
        If the file has fewer than *MIN_FILE_LENGTH* time steps.
        In that case empty arrays with the correct trailing dimensions
        are returned.

    Raises
    ------
    FileNotFoundError
        If *file_path* does not exist.
    AssertionError
        If internal shape or NaN invariants are violated.
    """
    # ── 1. Load ──────────────────────────────────────────────────────────
    raw = pd.read_parquet(file_path)

    # ── 2. Split signal / label ──────────────────────────────────────────
    signal_cols = [c for c in raw.columns if c not in ("class", "state")]
    signal_df = raw[signal_cols].copy()
    label_df = raw[["class"]].copy()

    # ── Early exit: file too short ───────────────────────────────────────
    n_timesteps = len(raw)
    if n_timesteps < MIN_FILE_LENGTH:
        warnings.warn(
            f"Skipping {file_path}: only {n_timesteps} time steps "
            f"(minimum {MIN_FILE_LENGTH} required)."
        )
        return (
            np.empty((0, 22, WINDOW_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    # ── 3. Clean, normalise, handle labels ──────────────────────────────
    processed = default_data_processing(
        data={"signal": signal_df, "label": label_df},
        fillna=True,
        target_column="class",
        fill_target_value=0,
    )
    signal_clean: pd.DataFrame = processed["signal"]   # (T, 22)
    label_clean: pd.DataFrame = processed["label"]     # (T, 1) — 'class'

    # ── 4. Windowing ─────────────────────────────────────────────────────
    windowing = Windowing(
        WindowingConfig(
            window="boxcar",
            window_size=WINDOW_SIZE,
            overlap=OVERLAP,
            pad_last_window=True,
            pad_value=0.0,
        )
    )

    windows_df = windowing(signal_clean)          # (n_windows, 22*120 + 1)
    label_windows_df = windowing(label_clean)     # (n_windows, 120 + 1)

    # ── 5. Label aggregation (mode) ─────────────────────────────────────
    label_cols = [c for c in label_windows_df.columns if c != "win"]
    label_array = label_windows_df[label_cols].values  # (n_windows, 120)
    # Guard against residual NaN (shouldn't happen after ffill/bfill, but
    # defensive)
    if np.isnan(label_array).any():
        label_array = np.nan_to_num(label_array, nan=0.0)

    mode_result = scipy_mode(label_array.astype(np.int64), axis=1, keepdims=False)
    y = mode_result.mode.ravel().astype(np.int64)

    # ── 6. Reshape signals ──────────────────────────────────────────────
    X_flat = windows_df.drop(columns=["win"]).values  # (n_windows, 22*120)
    n_vars = signal_clean.shape[1]  # should be 22
    X = X_flat.reshape(-1, n_vars, WINDOW_SIZE).astype(np.float32)

    # ── 7. Validation ────────────────────────────────────────────────────
    assert X.ndim == 3, f"Expected 3-D features, got {X.ndim}-D"
    assert X.shape[1:] == (22, WINDOW_SIZE), (
        f"Expected (n, 22, {WINDOW_SIZE}), got {X.shape}"
    )
    assert not np.isnan(X).any(), "NaN detected in feature windows"
    assert len(y) == len(X), f"Label/feature count mismatch: {len(y)} vs {len(X)}"

    return X, y


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------
def save_preprocessed(
    fold_id: str,
    X: np.ndarray,
    y: np.ndarray,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> None:
    """Save preprocessed feature and label arrays to NumPy cache files.

    Files are written as ``fold_{fold_id}_X.npy`` and
    ``fold_{fold_id}_y.npy`` inside *cache_dir*.

    Parameters
    ----------
    fold_id : str
        Identifier for the data fold (e.g. ``"train"``, ``"test"``,
        ``"holdout"``, or a fold number).
    X : np.ndarray
        Feature windows of shape ``(n_windows, 22, 120)``, float32.
    y : np.ndarray
        Per-window labels of shape ``(n_windows,)``, int64.
    cache_dir : str
        Directory in which to store the ``.npy`` files.
    """
    os.makedirs(cache_dir, exist_ok=True)
    np.save(os.path.join(cache_dir, f"fold_{fold_id}_X.npy"), X)
    np.save(os.path.join(cache_dir, f"fold_{fold_id}_y.npy"), y)


def load_preprocessed(
    fold_id: str,
    cache_dir: str = DEFAULT_CACHE_DIR,
) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed feature and label arrays from NumPy cache files.

    Parameters
    ----------
    fold_id : str
        Identifier for the data fold.
    cache_dir : str
        Directory containing the ``.npy`` cache files.

    Returns
    -------
    X : np.ndarray
        Feature windows of shape ``(n_windows, 22, 120)``, float32.
    y : np.ndarray
        Per-window labels of shape ``(n_windows,)``, int64.

    Raises
    ------
    FileNotFoundError
        If either ``fold_{fold_id}_X.npy`` or ``fold_{fold_id}_y.npy``
        does not exist.
    """
    x_path = os.path.join(cache_dir, f"fold_{fold_id}_X.npy")
    y_path = os.path.join(cache_dir, f"fold_{fold_id}_y.npy")

    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Cache file not found: {x_path}")
    if not os.path.exists(y_path):
        raise FileNotFoundError(f"Cache file not found: {y_path}")

    X: np.ndarray = np.load(x_path)
    y: np.ndarray = np.load(y_path)
    return X, y

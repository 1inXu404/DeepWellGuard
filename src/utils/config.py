import os
import random

import numpy as np
import torch

# Data parameters
WINDOW_SIZE = 120  # 120 time steps
OVERLAP = 0.5  # 50% overlap between windows
STRIDE = 60  # Effective stride = int(WINDOW_SIZE * (1 - OVERLAP))
DOWNSAMPLE_RATE = 2  # Downsample raw data by taking every 2nd row

# Data info
N_FEATURES = 22  # After removing UNUSED_TAGS from 27 sensors
N_CLASSES = 7  # Classes 0,1,3,4,5,6,9 (excluded 2,7,8)
EXCLUDED_CLASSES = [2, 7, 8]
RETAINED_CLASSES = [0, 1, 3, 4, 5, 6, 9]
HOLDOUT_RATIO = 0.2  # Per-class stratified holdout ratio

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.0003
MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 15

# CNN specific
CNN_CHANNELS = [64, 128, 256]
CNN_KERNEL = 3

# LSTM specific
LSTM_HIDDEN = 128
LSTM_LAYERS = 2
LSTM_BIDIRECTIONAL = True

# Attention specific
ATTN_HEADS = 4
ATTN_DROPOUT = 0.1

# Reproducibility
SEED = 42


def set_global_seed(seed: int = SEED, deterministic: bool = True) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Random seed used across the project.
        deterministic: If True, prefer deterministic CUDA/cuDNN behavior.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.use_deterministic_algorithms(False)


set_global_seed(SEED)

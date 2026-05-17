import random

import numpy as np
import torch

# Data parameters
WINDOW_SIZE = 120  # 120 time steps
OVERLAP = 0.5  # 50% overlap between windows
STRIDE = 60  # Effective stride = int(WINDOW_SIZE * (1 - OVERLAP))
DOWNSAMPLE_RATE = 5  # Downsample raw data by taking every 5th row

# Data info
N_FEATURES = 22  # After removing UNUSED_TAGS from 27 sensors
N_CLASSES = 7  # Classes 0,1,3,4,5,6,9 (excluded 2,7,8)
EXCLUDED_CLASSES = [2, 7, 8]
RETAINED_CLASSES = [0, 1, 3, 4, 5, 6, 9]
HOLDOUT_RATIO = 0.2  # Per-class stratified holdout ratio

# Training hyperparameters
BATCH_SIZE = 32
LEARNING_RATE = 0.001
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
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

"""Unified training loop with early stopping, class weighting, gradient clipping.

Provides the Trainer class that handles the full training lifecycle
for any PyTorch model accepting (batch, channels, time) input and
returning (batch, num_classes) logits.
"""

import copy
from contextlib import nullcontext
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# PyTorch 2.4+ uses torch.amp; older uses torch.cuda.amp
try:
    from torch.amp import GradScaler, autocast
    def _get_autocast():
        return autocast('cuda')
except ImportError:
    from torch.cuda.amp import GradScaler, autocast
    def _get_autocast():
        return autocast()

from src.utils.config import EARLY_STOPPING_PATIENCE, LEARNING_RATE, N_CLASSES


class Trainer:
    """Unified trainer with early stopping, class weighting, and gradient clipping.

    Works with any model whose ``forward(x)`` accepts ``(batch, C, T)``
    inputs and returns ``(batch, num_classes)`` logits.

    Attributes:
        model: The wrapped model, moved to ``self.device`` on construction.
        device: Torch device used for all computations.
        best_model_state: Deep copy of the best model weights (by val loss).
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: Optional[dict] = None,
    ):
        """Initialise the trainer.

        Args:
            model: A PyTorch model.
            device: Torch device (cpu / cuda / mps).
            config: Optional dict of hyperparameter overrides
                (``lr``, ``patience``, ``grad_clip``, ``num_classes``).
        """
        self.model = model.to(device)
        self.device = device
        self.best_model_state = None

        # Allow optional overrides via config dict
        cfg = config or {}
        self.learning_rate = cfg.get("lr", LEARNING_RATE)
        self.patience = cfg.get("patience", EARLY_STOPPING_PATIENCE)
        self.grad_clip = cfg.get("grad_clip", 1.0)
        self.num_classes = cfg.get("num_classes", N_CLASSES)

        # Mixed-precision (AMP) setup — GPU only
        self.use_amp = device.type == "cuda"
        if self.use_amp:
            try:
                self.scaler = GradScaler('cuda')
            except TypeError:
                self.scaler = GradScaler()  # PyTorch < 2.4
        else:
            self.scaler = None

    # ------------------------------------------------------------------
    # Training / Validation
    # ------------------------------------------------------------------

    def train_epoch(
        self,
        dataloader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
    ) -> float:
        """Run one training epoch.

        Returns:
            Average training loss across all batches.
        """
        self.model.train()
        total_loss = 0.0

        for x, y in dataloader:
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            optimizer.zero_grad()

            if self.use_amp:
                with _get_autocast():
                    logits = self.model(x)
                    loss = criterion(logits, y)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                logits = self.model(x)
                loss = criterion(logits, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=self.grad_clip
                )
                optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)

    def validate(
        self,
        dataloader,
        criterion: nn.Module,
    ) -> Tuple[float, float]:
        """Run validation and return (loss, accuracy).

        No gradient computation — uses ``torch.no_grad()``.

        Returns:
            Tuple of ``(avg_val_loss, accuracy)`` where accuracy is in [0, 1].
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        ctx = _get_autocast() if self.use_amp else nullcontext()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device, non_blocking=True)
                y = y.to(self.device, non_blocking=True)
                with ctx:
                    logits = self.model(x)
                    loss = criterion(logits, y)
                total_loss += loss.item()

                preds = logits.argmax(dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)

        return total_loss / len(dataloader), correct / total

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader,
        val_loader,
        epochs: int = 100,
        patience: Optional[int] = None,
    ) -> Dict[str, list]:
        """Full training loop with early stopping and class weighting.

        * Automatically computes class weights from training labels
          (inverse frequency).
        * Uses Adam optimiser with the configured learning rate.
        * Restores the best model weights (by validation loss) on
          early stopping or after the final epoch.

        Args:
            train_loader: Training DataLoader.
            val_loader: Validation DataLoader.
            epochs: Maximum number of epochs (default 100).
            patience: Early-stopping patience; falls back to
                ``self.patience`` when ``None``.

        Returns:
            History dict with keys ``'train_loss'``, ``'val_loss'``,
            ``'val_acc'`` — each a list of per-epoch values.
        """
        patience = patience if patience is not None else self.patience

        # We are using WeightedRandomSampler, so no class weights in loss
        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )
        
        # Only apply CosineAnnealingLR to the improved model (CNNLSTMAttention)
        is_attn_model = self.model.__class__.__name__ == "CNNLSTMAttention"
        scheduler = None
        if is_attn_model:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=epochs, eta_min=1e-5
            )

        history: Dict[str, list] = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
        }

        best_val_acc = -1.0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader, optimizer, criterion)
            val_loss, val_acc = self.validate(val_loader, criterion)
            
            # Step the learning rate scheduler if it exists
            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = self.learning_rate

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            
            # Per-epoch progress
            best_mark = ' *' if val_acc > best_val_acc else ''
            print(f"  Epoch {epoch:3d}/{epochs} | "
                  f"LR={current_lr:.6f} | "
                  f"train_loss={train_loss:.4f} | "
                  f"val_loss={val_loss:.4f} | "
                  f"val_acc={val_acc:.4f}{best_mark}")

            # Early-stopping logic based on Accuracy instead of Loss
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch}")
                    break

        # Restore best weights
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)

        return history

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        dataloader,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run inference on a dataloader.

        Returns:
            Tuple of ``(preds, probs, y_true)``.
            ``preds`` — ``(N,)`` int array of predicted class indices.
            ``probs`` — ``(N, num_classes)`` float array of softmax probabilities.
            ``y_true`` — ``(N,)`` int array of true labels.
        """
        self.model.eval()
        all_preds: list = []
        all_probs: list = []
        all_labels: list = []

        ctx = _get_autocast() if self.use_amp else nullcontext()
        with torch.no_grad():
            for x, y in dataloader:
                x = x.to(self.device, non_blocking=True)
                with ctx:
                    logits = self.model(x)
                probs = F.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                all_preds.append(preds.cpu().numpy())
                all_probs.append(probs.cpu().numpy())
                all_labels.append(y.numpy())

        return np.concatenate(all_preds), np.concatenate(all_probs), np.concatenate(all_labels)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_class_weights(self, dataloader) -> torch.Tensor:
        """Compute class weights inversely proportional to class frequencies.

        Classes that never appear in the training set receive a weight
        computed from a floor count of 1 (avoids division by zero).
        Weights are normalised so their mean equals 1.

        Returns:
            Float tensor of shape ``(num_classes,)``.
        """
        all_labels: list = []
        for _, y in dataloader:
            all_labels.append(y.numpy())
        labels = np.concatenate(all_labels)

        # Handle non-contiguous labels (e.g. [0,1,3,4,5,6,9] → [0..6])
        class_counts = np.zeros(self.num_classes, dtype=np.int64)
        unique, counts = np.unique(labels, return_counts=True)
        if unique.max() >= self.num_classes:
            label_map = {old: new for new, old in enumerate(sorted(unique))}
            labels = np.array([label_map[lbl] for lbl in labels])
            unique, counts = np.unique(labels, return_counts=True)
        for lbl, cnt in zip(unique, counts):
            if lbl < self.num_classes:
                class_counts[lbl] = cnt
        # Guard against missing classes (count == 0)
        class_counts = np.where(class_counts == 0, 1, class_counts)

        weights = 1.0 / class_counts.astype(np.float32)
        weights = weights / weights.sum() * len(weights)  # normalise
        return torch.tensor(weights, dtype=torch.float32)

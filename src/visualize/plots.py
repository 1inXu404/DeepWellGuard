"""Visualization utilities: training curves, confusion matrices, model comparison.

All plot functions accept an optional `save_path`; when provided the figure is
saved to disk and closed instead of being shown interactively.
"""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MultipleLocator, MaxNLocator  # noqa: E402

from src.utils.config import MAPPED_CLASS_NAMES  # noqa: E402


def _mapped_class_names(n_classes):
    return [
        MAPPED_CLASS_NAMES[i] if i < len(MAPPED_CLASS_NAMES) else f"class{i}"
        for i in range(n_classes)
    ]


def _set_dense_ticks(axis):
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_locator(MultipleLocator(0.1))
    axis.yaxis.set_minor_locator(MultipleLocator(0.05))
    axis.grid(True, axis="y", which="major", linestyle="--", linewidth=0.7, alpha=0.35)
    axis.grid(True, axis="y", which="minor", linestyle=":", linewidth=0.5, alpha=0.22)


def plot_training_curves(history_dict, model_name, save_path=None):
    """Plot training & validation loss + validation accuracy over epochs.

    Args:
        history_dict: dict with keys 'train_loss', 'val_loss', 'val_acc'.
                      Each value is a 1-D array-like of length n_epochs.
        model_name: str, used in the plot title.
        save_path: optional str path to save the figure as PNG.

    Creates a dual-axis plot: left y-axis for loss, right y-axis for accuracy.
    """
    fig, ax1 = plt.subplots(figsize=(11, 5.8))

    train_loss = np.asarray(history_dict["train_loss"], dtype=float)
    val_loss = np.asarray(history_dict["val_loss"], dtype=float)
    val_acc = np.asarray(history_dict["val_acc"], dtype=float)
    n_epochs = min(len(train_loss), len(val_loss), len(val_acc))
    if n_epochs == 0:
        raise ValueError("Training history must contain at least one epoch.")

    epochs = np.arange(1, n_epochs + 1)
    train_loss = train_loss[:n_epochs]
    val_loss = val_loss[:n_epochs]
    val_acc = val_acc[:n_epochs]

    for values, color, linestyle, label in [
        (train_loss, "#1f77b4", "-", "Train Loss"),
        (val_loss, "#4c78a8", "--", "Val Loss"),
    ]:
        ax1.plot(
            epochs,
            values,
            marker="o",
            markersize=3,
            linestyle=linestyle,
            linewidth=2.0,
            color=color,
            alpha=0.9,
            label=label,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_xlim(0.5, 1.5) if n_epochs == 1 else ax1.set_xlim(1, n_epochs)
    ax1.xaxis.set_major_locator(MaxNLocator(nbins=min(20, n_epochs), integer=True))
    ax1.xaxis.set_minor_locator(MaxNLocator(nbins=n_epochs, integer=True))
    _set_dense_ticks(ax1)

    ax2 = ax1.twinx()
    ax2.plot(
        epochs,
        val_acc,
        marker="o",
        markersize=3,
        linestyle="-",
        linewidth=2.0,
        color="#d62728",
        alpha=0.9,
        label="Val Acc",
    )
    ax2.set_ylabel("Accuracy", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    _set_dense_ticks(ax2)

    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=3,
        borderaxespad=0.4,
    )

    plt.title(f"{model_name} Training Curves")
    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(y_true, y_pred, save_path=None, class_names=None):
    """Plot a normalized confusion matrix heatmap.

    Args:
        y_true: Ground-truth labels, array-like of shape (n_samples,).
        y_pred: Predicted labels, array-like of shape (n_samples,).
        save_path: optional str path to save the figure as PNG.
        class_names: optional list of class name strings for axis labels.
                     Defaults to [0, 1, ..., n_classes-1].

    Returns:
        numpy.ndarray: The normalized confusion matrix.
    """
    from sklearn.metrics import confusion_matrix

    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    if class_names is None:
        class_names = _mapped_class_names(cm.shape[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return cm


def plot_comparison_bar(metrics_df, save_path=None):
    """Bar chart comparing weighted F1 and macro F1 across models.

    Args:
        metrics_df: pandas DataFrame with columns 'model', 'weighted_f1', 'macro_f1'.
        save_path: optional str path to save the figure as PNG.
    """
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(metrics_df))
    width = 0.35

    ax.bar(x - width / 2, metrics_df["weighted_f1"], width, label="Weighted F1")
    ax.bar(x + width / 2, metrics_df["macro_f1"], width, label="Macro F1")

    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["model"])
    ax.set_ylabel("F1 Score")
    ax.set_title("Model Comparison")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

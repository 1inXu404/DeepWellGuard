"""Evaluation metrics module for classification models.

Provides functions for computing accuracy, F1 scores, confusion matrices,
classification reports, and saving results to CSV.
"""

import pandas as pd
from src.utils.config import MAPPED_CLASS_NAMES
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize


def _mapped_class_names(n_classes):
    return [
        MAPPED_CLASS_NAMES[i] if i < len(MAPPED_CLASS_NAMES) else f"class{i}"
        for i in range(n_classes)
    ]


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute classification metrics.

    Args:
        y_true: Ground-truth labels, array-like of shape (n_samples,).
        y_pred: Predicted labels, array-like of shape (n_samples,).
        y_proba: Predicted probabilities (optional, reserved for future use).

    Returns:
        dict with keys:
            accuracy (float)
            weighted_f1 (float)
            macro_f1 (float)
            per_class_precision (list[float])
            per_class_recall (list[float])
            per_class_f1 (list[float])
    """
    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["weighted_f1"] = float(
        f1_score(y_true, y_pred, average="weighted", zero_division=0)
    )
    metrics["macro_f1"] = float(
        f1_score(y_true, y_pred, average="macro", zero_division=0)
    )
    metrics["per_class_precision"] = precision_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    metrics["per_class_recall"] = recall_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()
    metrics["per_class_f1"] = f1_score(
        y_true, y_pred, average=None, zero_division=0
    ).tolist()

    return metrics


def generate_confusion_matrix(y_true, y_pred, save_path=None, class_names=None):
    """Generate a normalized confusion matrix heatmap.

    Normalizes by 'true' class so each row sums to 1.

    Args:
        y_true: Ground-truth labels, array-like of shape (n_samples,).
        y_pred: Predicted labels, array-like of shape (n_samples,).
        save_path: If provided, saves the figure as a PNG to this path.
        class_names: List of class name strings for axis labels.
                     Defaults to [0, 1, ..., n_classes-1].

    Returns:
        numpy.ndarray: The normalized confusion matrix.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred, normalize="true")

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = _mapped_class_names(n_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return cm


def generate_confusion_matrix_counts(y_true, y_pred, save_path=None, class_names=None):
    """Generate a confusion matrix heatmap with absolute counts (not normalized).

    Args:
        y_true: Ground-truth labels, array-like of shape (n_samples,).
        y_pred: Predicted labels, array-like of shape (n_samples,).
        save_path: If provided, saves the figure as a PNG to this path.
        class_names: List of class name strings for axis labels.
                     Defaults to [0, 1, ..., n_classes-1].

    Returns:
        numpy.ndarray: The confusion matrix (counts).
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)

    n_classes = cm.shape[0]
    if class_names is None:
        class_names = _mapped_class_names(n_classes)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",  # Display integer format
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return cm


def generate_classification_report(y_true, y_pred):
    """Return sklearn's classification report as a string.

    Args:
        y_true: Ground-truth labels, array-like of shape (n_samples,).
        y_pred: Predicted labels, array-like of shape (n_samples,).

    Returns:
        str: Formatted classification report (precision, recall, f1 per class).
    """
    return classification_report(y_true, y_pred, zero_division=0)


def save_metrics_csv(metrics_dict, save_path):
    """Save a metrics dictionary to a CSV file.

    Args:
        metrics_dict: dict of metric_name -> value.
                      Values can be scalar or list-like.
        save_path: Path to save the CSV file.
    """
    df = pd.DataFrame([metrics_dict])
    df.to_csv(save_path, index=False)


def generate_roc_curve(y_true, y_proba, save_path=None, class_names=None):
    """Generate and plot multi-class ROC-AUC curves.

    Args:
        y_true: Ground-truth labels, array-like of shape (n_samples,).
        y_proba: Predicted probabilities, array-like of shape (n_samples, n_classes).
        save_path: If provided, saves the figure as a PNG to this path.
        class_names: List of class name strings for the legend.
    """
    import matplotlib.pyplot as plt
    from itertools import cycle

    n_classes = y_proba.shape[1]

    # Binarize the output
    classes = list(range(n_classes))
    y_true_bin = label_binarize(y_true, classes=classes)

    if n_classes == 2:
        y_true_bin = y_true_bin.reshape(-1, 1)

    if class_names is None:
        class_names = _mapped_class_names(n_classes)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_proba[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot all ROC curves
    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(
            fpr[i],
            tpr[i],
            color=color,
            lw=2,
            label=f"ROC curve of class {class_names[i]} (area = {roc_auc[i]:.4f})",
        )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-class Receiver Operating Characteristic (ROC)")
    plt.legend(loc="lower right")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def metrics_all(y_pred, y_true):
    """Calculate binary classification metrics from logits and integer labels."""
    logits = torch.as_tensor(y_pred).detach().float().cpu()
    labels = torch.as_tensor(y_true).detach().long().cpu().numpy()
    probabilities = torch.softmax(logits, dim=1)[:, 1].numpy()
    predictions = logits.argmax(dim=1).numpy()
    has_both_classes = np.unique(labels).size == 2

    return {
        "acc": float(accuracy_score(labels, predictions)),
        "prc": float(average_precision_score(labels, probabilities)),
        "roc": float(roc_auc_score(labels, probabilities)) if has_both_classes else float("nan"),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
    }


def binary_curves(y_true, y_prob, threshold=0.5):
    """Return binary ROC/PR curve coordinates and a fixed-label confusion matrix."""
    labels = np.asarray(y_true, dtype=np.int64)
    probabilities = np.asarray(y_prob, dtype=np.float64)
    if np.unique(labels).size != 2:
        raise ValueError("ROC and PR plots require both classes in the evaluation set.")

    fpr, tpr, _ = roc_curve(labels, probabilities)
    precision, recall, _ = precision_recall_curve(labels, probabilities)
    predictions = (probabilities > threshold).astype(np.int64)
    matrix = confusion_matrix(labels, predictions, labels=[0, 1])
    return {
        "fpr": fpr,
        "tpr": tpr,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": matrix,
        "roc_auc": float(roc_auc_score(labels, probabilities)),
    }
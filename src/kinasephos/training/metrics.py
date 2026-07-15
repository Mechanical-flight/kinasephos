import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
    roc_auc_score,
)


def binary_metrics(targets, probabilities, threshold: float = 0.5) -> dict[str, object]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)
    matrix = confusion_matrix(targets, predictions, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "accuracy": accuracy_score(targets, predictions),
        "precision": precision_score(targets, predictions, zero_division=0),
        "recall": recall_score(targets, predictions, zero_division=0),
        "f1": f1_score(targets, predictions, zero_division=0),
        "roc_auc": roc_auc_score(targets, probabilities) if len(set(targets)) > 1 else None,
        "pr_auc": average_precision_score(targets, probabilities) if targets.sum() else None,
        "mcc": matthews_corrcoef(targets, predictions),
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "confusion_matrix": matrix.tolist(),
    }


def family_metrics(targets, probabilities) -> dict[str, object]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    valid = (targets >= 0) & (targets < 4)
    targets = targets[valid]
    probabilities = probabilities[valid]
    if not len(targets):
        return {"status": "unavailable", "valid_samples": 0}
    top1 = probabilities.argmax(axis=1)
    top3 = np.argsort(probabilities, axis=1)[:, -3:]
    precision, recall, f1, support = precision_recall_fscore_support(
        targets, top1, labels=range(4), zero_division=0
    )
    names = ("CMGC", "AGC", "TK", "CAMK")
    return {
        "status": "available",
        "valid_samples": len(targets),
        "top1_accuracy": accuracy_score(targets, top1),
        "top3_accuracy": float(
            np.mean([target in row for target, row in zip(targets, top3, strict=True)])
        ),
        "macro_f1": f1_score(targets, top1, average="macro", zero_division=0),
        "weighted_f1": f1_score(targets, top1, average="weighted", zero_division=0),
        "per_class": {
            name: {
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
                "support": int(support[index]),
            }
            for index, name in enumerate(names)
        },
        "confusion_matrix": confusion_matrix(targets, top1, labels=range(4)).tolist(),
    }

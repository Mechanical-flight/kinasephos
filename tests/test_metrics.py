import numpy as np

from kinasephos.training.metrics import family_metrics


def test_family_metrics_include_all_per_class_rows():
    targets = np.array([0, 1, 2, 3])
    probabilities = np.eye(4)
    metrics = family_metrics(targets, probabilities)
    assert metrics["top1_accuracy"] == 1.0
    assert set(metrics["per_class"]) == {"CMGC", "AGC", "TK", "CAMK"}
    assert metrics["per_class"]["TK"] == {
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "support": 1,
    }

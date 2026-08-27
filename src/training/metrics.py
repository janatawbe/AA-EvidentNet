"""Training/validation metric accumulation.

Reuses scikit-learn (already a project dependency) for macro-F1, balanced
accuracy, and accuracy — no metric math is reimplemented here. This is
intentionally a small subset (loss, accuracy, macro-F1 for training;
+balanced accuracy for validation) — the full research evaluation suite
(calibration, selective prediction, hard-pair analysis, ...) is explicitly
out of scope for Task 6 and lands in later tasks.
"""

from typing import Any, Dict, List, Sequence

import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def compute_classification_metrics(
    y_true: Sequence[int], y_pred: Sequence[int], include_balanced_accuracy: bool = False
) -> Dict[str, float]:
    """Pure metric computation over a full epoch's predictions/labels."""
    if not y_true:
        result = {"accuracy": 0.0, "macro_f1": 0.0}
    else:
        result = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        }
    if include_balanced_accuracy:
        result["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred)) if y_true else 0.0
    return result


class MetricAccumulator:
    """Accumulates loss and predictions/labels across an epoch's batches,
    then computes epoch-level metrics once at the end (sklearn metrics like
    macro-F1 are not meaningfully averaged batch-by-batch)."""

    def __init__(self) -> None:
        self.total_loss = 0.0
        self.total_samples = 0
        self.all_preds: List[int] = []
        self.all_labels: List[int] = []

    def update(self, loss_value: float, batch_size: int, preds: torch.Tensor, labels: torch.Tensor) -> None:
        self.total_loss += loss_value * batch_size
        self.total_samples += batch_size
        self.all_preds.extend(preds.detach().cpu().tolist())
        self.all_labels.extend(labels.detach().cpu().tolist())

    def compute(self, include_balanced_accuracy: bool = False) -> Dict[str, Any]:
        avg_loss = self.total_loss / self.total_samples if self.total_samples else 0.0
        metrics = compute_classification_metrics(self.all_labels, self.all_preds, include_balanced_accuracy)
        return {"loss": avg_loss, **metrics}

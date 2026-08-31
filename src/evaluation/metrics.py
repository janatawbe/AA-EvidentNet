"""Multiclass evaluation metrics for the final held-out test evaluation
(Task 8 -- src/evaluation/final_test.py).

Deliberately separate from src/training/metrics.py, which is scoped ONLY
to per-epoch training/validation-loop bookkeeping (loss, accuracy,
macro-F1, +balanced accuracy for validation) and whose own docstring
explicitly says the full evaluation suite is out of scope for it. This
module implements exactly that full suite's CORE metrics: overall
accuracy/balanced accuracy/macro precision/recall/F1/ROC-AUC/PR-AUC,
per-class precision/recall/specificity/F1/ROC-AUC/PR-AUC, and a full
confusion matrix -- reusing scikit-learn throughout (no metric math is
reimplemented), consistent with the project's existing convention.

Every function takes a FIXED `num_classes` and always evaluates over
labels `0..num_classes-1` explicitly (via sklearn's `labels=` parameter),
never sklearn's default "only observed labels" behavior -- so a class
with zero samples in a given evaluation (e.g. a tiny synthetic test
fixture, or in principle any future test split) never silently reshapes
or reorders a confusion matrix or metric array. Column/row index `k`
always means the same class in every array this module returns; the
caller (src/evaluation/final_test.py) is responsible for mapping that
index back to a canonical class name via the project's single ordering
(src.data.dataset.build_class_to_idx).

Undefined metrics (e.g. ROC-AUC for a class with zero positive or zero
negative examples in this evaluation) are reported as `None` with an
explicit reason string, never silently coerced to 0.0 or omitted. Macro
precision/recall/F1 follow this project's existing, already-established
`zero_division=0` sklearn convention (same as src/training/metrics.py);
macro ROC-AUC/PR-AUC are instead averaged only over classes where the
underlying binary metric is actually defined, with the count of
classes that contributed reported alongside the average.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


class MetricsInputError(Exception):
    """Raised for malformed metric inputs (shape mismatches, out-of-range
    labels, wrong num_classes) -- never silently truncated, padded, or
    coerced."""


def _validate_labels(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> Tuple[np.ndarray, np.ndarray]:
    if num_classes <= 0:
        raise MetricsInputError(f"num_classes must be > 0, got {num_classes}")
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if y_true_arr.shape != y_pred_arr.shape:
        raise MetricsInputError(f"y_true shape {y_true_arr.shape} != y_pred shape {y_pred_arr.shape}")
    if y_true_arr.ndim != 1:
        raise MetricsInputError(f"y_true/y_pred must be 1-D, got shape {y_true_arr.shape}")
    if y_true_arr.size > 0:
        for name, arr in (("y_true", y_true_arr), ("y_pred", y_pred_arr)):
            if int(arr.min()) < 0 or int(arr.max()) >= num_classes:
                raise MetricsInputError(f"{name} contains a label outside [0, {num_classes - 1}]")
    return y_true_arr, y_pred_arr


def _validate_probabilities(probabilities: Sequence[Sequence[float]], n_samples: int, num_classes: int) -> np.ndarray:
    probabilities_arr = np.asarray(probabilities, dtype=float)
    expected_shape = (n_samples, num_classes)
    if probabilities_arr.shape != expected_shape:
        raise MetricsInputError(f"probabilities must have shape {expected_shape}, got {probabilities_arr.shape}")
    return probabilities_arr


def compute_confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """K x K confusion matrix (rows=true, cols=predicted), fixed label
    ordering 0..num_classes-1 regardless of which labels actually appear."""
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred, num_classes)
    return confusion_matrix(y_true_arr, y_pred_arr, labels=list(range(num_classes)))


def _binary_roc_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    positives = int(y_true_bin.sum())
    negatives = int(y_true_bin.size - positives)
    if positives == 0:
        return None, "no positive samples for this class in this evaluation"
    if negatives == 0:
        return None, "no negative samples for this class in this evaluation (every sample belongs to it)"
    return float(roc_auc_score(y_true_bin, scores)), None


def _binary_pr_auc(y_true_bin: np.ndarray, scores: np.ndarray) -> Tuple[Optional[float], Optional[str]]:
    positives = int(y_true_bin.sum())
    if positives == 0:
        return None, "no positive samples for this class in this evaluation"
    return float(average_precision_score(y_true_bin, scores)), None


def compute_per_class_roc_auc(
    y_true: Sequence[int], probabilities: Sequence[Sequence[float]], num_classes: int
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    """One-vs-rest ROC-AUC per class. Returns (values, undefined_reasons);
    values[k] is None (with a reason in undefined_reasons[k]) exactly when
    class k has zero positive or zero negative samples in y_true."""
    y_true_arr = np.asarray(y_true)
    probabilities_arr = _validate_probabilities(probabilities, len(y_true_arr), num_classes)
    values: List[Optional[float]] = []
    reasons: List[Optional[str]] = []
    for k in range(num_classes):
        y_true_bin = (y_true_arr == k).astype(int)
        value, reason = _binary_roc_auc(y_true_bin, probabilities_arr[:, k])
        values.append(value)
        reasons.append(reason)
    return values, reasons


def compute_per_class_pr_auc(
    y_true: Sequence[int], probabilities: Sequence[Sequence[float]], num_classes: int
) -> Tuple[List[Optional[float]], List[Optional[str]]]:
    """One-vs-rest PR-AUC (average precision) per class. Returns (values,
    undefined_reasons); values[k] is None exactly when class k has zero
    positive samples in y_true (PR-AUC is not meaningful with no positives,
    regardless of what a given sklearn version's average_precision_score
    happens to return in that case)."""
    y_true_arr = np.asarray(y_true)
    probabilities_arr = _validate_probabilities(probabilities, len(y_true_arr), num_classes)
    values: List[Optional[float]] = []
    reasons: List[Optional[str]] = []
    for k in range(num_classes):
        y_true_bin = (y_true_arr == k).astype(int)
        value, reason = _binary_pr_auc(y_true_bin, probabilities_arr[:, k])
        values.append(value)
        reasons.append(reason)
    return values, reasons


def compute_overall_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    num_classes: int,
) -> Dict[str, Any]:
    """Accuracy, balanced accuracy, macro precision/recall/F1 (sklearn's
    zero_division=0 convention), and macro ROC-AUC/PR-AUC (averaged only
    over classes where defined -- see module docstring). All keys always
    present; values are None (never a fabricated number) whenever the
    corresponding metric is undefined for this evaluation (e.g. zero
    samples, or zero classes with a defined binary ROC-AUC/PR-AUC)."""
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred, num_classes)
    probabilities_arr = _validate_probabilities(probabilities, len(y_true_arr), num_classes)
    labels = list(range(num_classes))

    if y_true_arr.size == 0:
        overall: Dict[str, Any] = {
            "num_samples": 0,
            "accuracy": None,
            "balanced_accuracy": None,
            "macro_precision": None,
            "macro_recall": None,
            "macro_f1": None,
        }
    else:
        overall = {
            "num_samples": int(y_true_arr.size),
            "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_arr, y_pred_arr)),
            "macro_precision": float(precision_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
            "macro_recall": float(recall_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
            "macro_f1": float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)),
        }

    roc_auc_values, roc_auc_reasons = compute_per_class_roc_auc(y_true_arr, probabilities_arr, num_classes)
    pr_auc_values, pr_auc_reasons = compute_per_class_pr_auc(y_true_arr, probabilities_arr, num_classes)
    defined_roc = [v for v in roc_auc_values if v is not None]
    defined_pr = [v for v in pr_auc_values if v is not None]

    overall["macro_roc_auc"] = float(np.mean(defined_roc)) if defined_roc else None
    overall["macro_roc_auc_num_classes_defined"] = len(defined_roc)
    overall["macro_pr_auc"] = float(np.mean(defined_pr)) if defined_pr else None
    overall["macro_pr_auc_num_classes_defined"] = len(defined_pr)
    overall["num_classes"] = num_classes

    return overall


def compute_per_class_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    num_classes: int,
) -> List[Dict[str, Any]]:
    """One row per class (index 0..num_classes-1): precision, recall
    (=sensitivity), specificity, F1, ROC-AUC, PR-AUC, plus `support`
    (actual count of this class) and `predicted_count` (count predicted as
    this class) and explicit `*_defined` booleans so a caller never
    mistakes a 0.0 placeholder for a genuine zero."""
    y_true_arr, y_pred_arr = _validate_labels(y_true, y_pred, num_classes)
    probabilities_arr = _validate_probabilities(probabilities, len(y_true_arr), num_classes)
    labels = list(range(num_classes))

    cm = compute_confusion_matrix(y_true_arr, y_pred_arr, num_classes)
    total = int(cm.sum())

    if y_true_arr.size > 0:
        precisions = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
        recalls = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
        f1s = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    else:
        precisions = recalls = f1s = [0.0] * num_classes

    roc_auc_values, roc_auc_reasons = compute_per_class_roc_auc(y_true_arr, probabilities_arr, num_classes)
    pr_auc_values, pr_auc_reasons = compute_per_class_pr_auc(y_true_arr, probabilities_arr, num_classes)

    rows: List[Dict[str, Any]] = []
    for k in range(num_classes):
        support = int(cm[k, :].sum())
        predicted_count = int(cm[:, k].sum())
        tp = int(cm[k, k])
        fp = predicted_count - tp
        fn = support - tp
        tn = total - tp - fp - fn
        specificity_defined = (tn + fp) > 0
        rows.append(
            {
                "class_index": k,
                "support": support,
                "predicted_count": predicted_count,
                "precision": float(precisions[k]),
                "precision_defined": predicted_count > 0,
                "recall": float(recalls[k]),
                "recall_defined": support > 0,
                "specificity": float(tn / (tn + fp)) if specificity_defined else None,
                "specificity_defined": specificity_defined,
                "f1": float(f1s[k]),
                "roc_auc": roc_auc_values[k],
                "roc_auc_undefined_reason": roc_auc_reasons[k],
                "pr_auc": pr_auc_values[k],
                "pr_auc_undefined_reason": pr_auc_reasons[k],
            }
        )
    return rows


@dataclass(frozen=True)
class EvaluationResult:
    confusion_matrix: np.ndarray
    overall: Dict[str, Any]
    per_class: List[Dict[str, Any]]


def evaluate_predictions(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probabilities: Sequence[Sequence[float]],
    num_classes: int,
) -> EvaluationResult:
    """One-call convenience wrapper computing everything final_test.py
    needs: the confusion matrix, overall metrics, and per-class metrics."""
    return EvaluationResult(
        confusion_matrix=compute_confusion_matrix(y_true, y_pred, num_classes),
        overall=compute_overall_metrics(y_true, y_pred, probabilities, num_classes),
        per_class=compute_per_class_metrics(y_true, y_pred, probabilities, num_classes),
    )

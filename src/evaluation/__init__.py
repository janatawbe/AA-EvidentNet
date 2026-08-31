"""Evaluation metrics and the final held-out test pipeline.

`final_test.py` (Task 8) implements the frozen-checkpoint final test
evaluation (`python run_pipeline.py final_test`). `metrics.py` (Task 8)
implements the core multiclass metrics it uses. Calibration, selective
prediction, hard-pair analysis, Grad-CAM, and robustness are not yet
implemented (later tasks).
"""

from .final_test import (
    ALL_MODEL_NAMES,
    FinalTestError,
    FinalTestSummary,
    run_final_test,
)
from .metrics import (
    EvaluationResult,
    MetricsInputError,
    compute_confusion_matrix,
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_per_class_pr_auc,
    compute_per_class_roc_auc,
    evaluate_predictions,
)

__all__ = [
    "ALL_MODEL_NAMES",
    "FinalTestError",
    "FinalTestSummary",
    "run_final_test",
    "EvaluationResult",
    "MetricsInputError",
    "compute_confusion_matrix",
    "compute_overall_metrics",
    "compute_per_class_metrics",
    "compute_per_class_pr_auc",
    "compute_per_class_roc_auc",
    "evaluate_predictions",
]

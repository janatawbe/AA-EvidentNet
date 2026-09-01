"""Evaluation metrics, the final held-out test pipeline, and robustness
evaluation.

`final_test.py` (Task 8) implements the frozen-checkpoint final test
evaluation (`python run_pipeline.py final_test`). `metrics.py` (Task 8)
implements the core multiclass metrics it uses. `robustness.py`
implements robustness evaluation under fixed, predefined image
degradations (`python run_pipeline.py robustness`) - a separate,
additional test-time analysis that never reads, writes, or overwrites
anything final_test.py already produced. Calibration, selective
prediction, hard-pair analysis, and Grad-CAM are not yet implemented
(later tasks).
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
from .robustness import (
    CLEAN_REFERENCE_LABEL,
    DEFAULT_DEGRADATION_SEVERITIES,
    RobustnessError,
    RobustnessSummary,
    apply_degradation,
    run_robustness_evaluation,
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
    "CLEAN_REFERENCE_LABEL",
    "DEFAULT_DEGRADATION_SEVERITIES",
    "RobustnessError",
    "RobustnessSummary",
    "apply_degradation",
    "run_robustness_evaluation",
]

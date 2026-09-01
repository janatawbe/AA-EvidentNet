"""Evaluation metrics, the final held-out test pipeline, robustness
evaluation, and the feature-distance OOD/EDL uncertainty combination.

`final_test.py` (Task 8) implements the frozen-checkpoint final test
evaluation (`python run_pipeline.py final_test`). `metrics.py` (Task 8)
implements the core multiclass metrics it uses. `robustness.py`
implements robustness evaluation under fixed, predefined image
degradations (`python run_pipeline.py robustness`) - a separate,
additional test-time analysis that never reads, writes, or overwrites
anything final_test.py already produced. `ood_uncertainty.py` combines a
feature-distance (cosine, to the nearest train-original class prototype)
OOD score with AA-EvidentNet's own EDL uncertainty
(`python run_pipeline.py ood_uncertainty`) - AA-EvidentNet only, calibrated
entirely from train_original.csv/val_original.csv, never from
test_original.csv, and never overwriting final_test.py's or robustness.py's
outputs. Calibration, selective prediction (beyond the coverage sweep
ood_uncertainty.py already reads), hard-pair analysis, and Grad-CAM are not
yet implemented (later tasks).
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
from .ood_uncertainty import (
    DEFAULT_WEIGHT_GRID,
    OODCalibration,
    OODUncertaintyError,
    OODUncertaintySummary,
    calibrate_ood_uncertainty,
    compute_class_prototypes,
    nearest_prototype_cosine_distance,
    run_ood_uncertainty_evaluation,
    select_combine_weight,
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
    "DEFAULT_WEIGHT_GRID",
    "OODCalibration",
    "OODUncertaintyError",
    "OODUncertaintySummary",
    "calibrate_ood_uncertainty",
    "compute_class_prototypes",
    "nearest_prototype_cosine_distance",
    "run_ood_uncertainty_evaluation",
    "select_combine_weight",
]

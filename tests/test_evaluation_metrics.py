"""Tests for src.evaluation.metrics (Task 8: final-test core metrics).

Expected values for the primary cases are derived by hand (not just
cross-checked against sklearn's own output), so these tests catch a
genuine correctness regression, not merely a change in behavior."""

import numpy as np
import pytest

from src.evaluation.metrics import (
    MetricsInputError,
    compute_confusion_matrix,
    compute_overall_metrics,
    compute_per_class_metrics,
    compute_per_class_pr_auc,
    compute_per_class_roc_auc,
    evaluate_predictions,
)

# --- a hand-verified, perfectly symmetric 3-class example ---
# y_true=[0,0,1,1,2,2], y_pred=[0,1,1,2,2,0]
# confusion matrix (rows=true, cols=pred):
#   [[1,1,0],
#    [0,1,1],
#    [1,0,1]]
# Every class: support=2, predicted_count=2, TP=1, FP=1, FN=1, TN=3
#   -> precision=recall=f1=0.5, specificity=3/4=0.75 for every class.
# accuracy = 3/6 = 0.5; balanced_accuracy = mean(recall) = 0.5.
SYMMETRIC_Y_TRUE = [0, 0, 1, 1, 2, 2]
SYMMETRIC_Y_PRED = [0, 1, 1, 2, 2, 0]
SYMMETRIC_NUM_CLASSES = 3
# Probabilities are irrelevant to these particular assertions but must be
# shaped correctly; put a modest amount of signal on the predicted class.
SYMMETRIC_PROBS = [
    [0.6, 0.3, 0.1],
    [0.2, 0.6, 0.2],
    [0.2, 0.6, 0.2],
    [0.1, 0.2, 0.7],
    [0.1, 0.2, 0.7],
    [0.7, 0.1, 0.2],
]


def test_confusion_matrix_matches_hand_computation():
    cm = compute_confusion_matrix(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_NUM_CLASSES)
    expected = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
    assert np.array_equal(cm, expected)


def test_confusion_matrix_fixed_ordering_includes_absent_classes():
    # num_classes=4 but class 3 never appears in true or pred - the matrix
    # must still be 4x4 with an all-zero row/column for class 3, not
    # silently shrink to 3x3 (sklearn's default "observed labels only"
    # behavior, which this module explicitly overrides).
    cm = compute_confusion_matrix([0, 1, 2], [0, 1, 2], num_classes=4)
    assert cm.shape == (4, 4)
    assert np.array_equal(cm[3, :], [0, 0, 0, 0])
    assert np.array_equal(cm[:, 3], [0, 0, 0, 0])


def test_accuracy_matches_hand_computation():
    overall = compute_overall_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert overall["accuracy"] == pytest.approx(0.5)


def test_balanced_accuracy_matches_hand_computation():
    overall = compute_overall_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert overall["balanced_accuracy"] == pytest.approx(0.5)


def test_macro_precision_matches_hand_computation():
    overall = compute_overall_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert overall["macro_precision"] == pytest.approx(0.5)


def test_macro_recall_matches_hand_computation():
    overall = compute_overall_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert overall["macro_recall"] == pytest.approx(0.5)


def test_macro_f1_matches_hand_computation():
    overall = compute_overall_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert overall["macro_f1"] == pytest.approx(0.5)


def test_per_class_precision_recall_f1_match_hand_computation():
    rows = compute_per_class_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert len(rows) == 3
    for row in rows:
        assert row["precision"] == pytest.approx(0.5)
        assert row["recall"] == pytest.approx(0.5)
        assert row["f1"] == pytest.approx(0.5)
        assert row["support"] == 2
        assert row["predicted_count"] == 2
        assert row["precision_defined"] is True
        assert row["recall_defined"] is True


def test_per_class_specificity_matches_hand_computation():
    rows = compute_per_class_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    for row in rows:
        assert row["specificity"] == pytest.approx(0.75)
        assert row["specificity_defined"] is True


def test_per_class_metrics_class_index_matches_position():
    rows = compute_per_class_metrics(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert [row["class_index"] for row in rows] == [0, 1, 2]


# --- ROC-AUC / PR-AUC: perfect separation ---

PERFECT_SEP_Y_TRUE = [0, 0, 1, 1]
PERFECT_SEP_PROBS = [
    [0.9, 0.1],
    [0.8, 0.2],
    [0.2, 0.8],
    [0.1, 0.9],
]


def test_roc_auc_perfect_separation_equals_one():
    values, reasons = compute_per_class_roc_auc(PERFECT_SEP_Y_TRUE, PERFECT_SEP_PROBS, num_classes=2)
    assert values == pytest.approx([1.0, 1.0])
    assert reasons == [None, None]


def test_pr_auc_perfect_separation_equals_one():
    values, reasons = compute_per_class_pr_auc(PERFECT_SEP_Y_TRUE, PERFECT_SEP_PROBS, num_classes=2)
    assert values == pytest.approx([1.0, 1.0])
    assert reasons == [None, None]


def test_macro_roc_auc_and_pr_auc_reflect_perfect_separation():
    overall = compute_overall_metrics(PERFECT_SEP_Y_TRUE, [0, 0, 1, 1], PERFECT_SEP_PROBS, num_classes=2)
    assert overall["macro_roc_auc"] == pytest.approx(1.0)
    assert overall["macro_roc_auc_num_classes_defined"] == 2
    assert overall["macro_pr_auc"] == pytest.approx(1.0)
    assert overall["macro_pr_auc_num_classes_defined"] == 2


# --- ROC-AUC / PR-AUC: undefined handling ---


def test_roc_auc_undefined_when_class_has_no_positive_or_no_negative_samples():
    # Every sample is class 0: class 0 has zero negatives, class 1 has zero positives.
    y_true = [0, 0, 0]
    probs = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    values, reasons = compute_per_class_roc_auc(y_true, probs, num_classes=2)
    assert values == [None, None]
    assert reasons[0] is not None and "negative" in reasons[0]
    assert reasons[1] is not None and "positive" in reasons[1]


def test_pr_auc_undefined_only_when_class_has_no_positive_samples():
    # PR-AUC only requires >=1 positive - class 0 (all positive) IS
    # defined even though its ROC-AUC is not; class 1 (zero positive) is
    # undefined. This asymmetry vs. ROC-AUC is intentional.
    y_true = [0, 0, 0]
    probs = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    values, reasons = compute_per_class_pr_auc(y_true, probs, num_classes=2)
    assert values[0] is not None
    assert values[1] is None
    assert reasons[1] is not None and "positive" in reasons[1]


def test_macro_roc_auc_none_when_zero_classes_defined():
    y_true = [0, 0, 0]
    probs = [[0.9, 0.1], [0.8, 0.2], [0.7, 0.3]]
    overall = compute_overall_metrics(y_true, [0, 0, 0], probs, num_classes=2)
    assert overall["macro_roc_auc"] is None
    assert overall["macro_roc_auc_num_classes_defined"] == 0
    # PR-AUC for class 0 is still defined, so macro PR-AUC is not None.
    assert overall["macro_pr_auc"] is not None
    assert overall["macro_pr_auc_num_classes_defined"] == 1


def test_specificity_undefined_when_no_actual_negatives():
    # A 1-class-only batch (num_classes=1): every sample is the only
    # class, so there are no negatives at all for it - specificity is
    # mathematically undefined (0/0), not silently 0 or 1.
    rows = compute_per_class_metrics([0, 0, 0], [0, 0, 0], [[1.0], [1.0], [1.0]], num_classes=1)
    assert rows[0]["specificity_defined"] is False
    assert rows[0]["specificity"] is None


# --- empty input handling ---


def test_overall_metrics_empty_input_reports_none_not_error():
    overall = compute_overall_metrics([], [], np.zeros((0, 3)), num_classes=3)
    assert overall["num_samples"] == 0
    assert overall["accuracy"] is None
    assert overall["balanced_accuracy"] is None
    assert overall["macro_precision"] is None
    assert overall["macro_recall"] is None
    assert overall["macro_f1"] is None


def test_per_class_metrics_empty_input_reports_zero_support_not_error():
    rows = compute_per_class_metrics([], [], np.zeros((0, 3)), num_classes=3)
    assert len(rows) == 3
    for row in rows:
        assert row["support"] == 0
        assert row["recall_defined"] is False


# --- input validation ---


def test_shape_mismatch_raises_clearly():
    with pytest.raises(MetricsInputError, match="shape"):
        compute_confusion_matrix([0, 1], [0], num_classes=2)


def test_label_out_of_range_raises_clearly():
    with pytest.raises(MetricsInputError, match=r"\[0, 1\]"):
        compute_confusion_matrix([0, 2], [0, 1], num_classes=2)


def test_negative_label_raises_clearly():
    with pytest.raises(MetricsInputError, match=r"\[0, 1\]"):
        compute_confusion_matrix([0, -1], [0, 1], num_classes=2)


def test_non_positive_num_classes_raises_clearly():
    with pytest.raises(MetricsInputError, match="num_classes"):
        compute_confusion_matrix([0], [0], num_classes=0)


def test_wrong_probabilities_shape_raises_clearly():
    with pytest.raises(MetricsInputError, match="probabilities"):
        compute_overall_metrics([0, 1], [0, 1], [[0.5, 0.5]], num_classes=2)  # only 1 row for 2 samples


def test_non_1d_labels_raise_clearly():
    with pytest.raises(MetricsInputError, match="1-D"):
        compute_confusion_matrix([[0, 1]], [[0, 1]], num_classes=2)


# --- evaluate_predictions: single-call integration ---


def test_evaluate_predictions_bundles_all_three_results_consistently():
    result = evaluate_predictions(SYMMETRIC_Y_TRUE, SYMMETRIC_Y_PRED, SYMMETRIC_PROBS, SYMMETRIC_NUM_CLASSES)
    assert result.confusion_matrix.shape == (3, 3)
    assert result.overall["accuracy"] == pytest.approx(0.5)
    assert len(result.per_class) == 3
    assert result.overall["num_samples"] == 6

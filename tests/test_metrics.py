"""Tests for src.training.metrics."""

import torch

from src.training.metrics import MetricAccumulator, compute_classification_metrics


def test_compute_classification_metrics_perfect_predictions():
    y_true = [0, 1, 2, 0, 1, 2]
    y_pred = [0, 1, 2, 0, 1, 2]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0


def test_compute_classification_metrics_all_wrong():
    y_true = [0, 0, 0]
    y_pred = [1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_compute_classification_metrics_empty():
    metrics = compute_classification_metrics([], [])
    assert metrics["accuracy"] == 0.0
    assert metrics["macro_f1"] == 0.0


def test_compute_classification_metrics_includes_balanced_accuracy_when_requested():
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    metrics = compute_classification_metrics(y_true, y_pred, include_balanced_accuracy=True)
    assert "balanced_accuracy" in metrics
    assert 0.0 <= metrics["balanced_accuracy"] <= 1.0


def test_compute_classification_metrics_excludes_balanced_accuracy_by_default():
    metrics = compute_classification_metrics([0, 1], [0, 1])
    assert "balanced_accuracy" not in metrics


def test_metric_accumulator_computes_correct_loss_average():
    accumulator = MetricAccumulator()
    accumulator.update(loss_value=2.0, batch_size=2, preds=torch.tensor([0, 1]), labels=torch.tensor([0, 1]))
    accumulator.update(loss_value=4.0, batch_size=2, preds=torch.tensor([0, 0]), labels=torch.tensor([0, 1]))
    result = accumulator.compute()
    # weighted average: (2.0*2 + 4.0*2) / 4 = 3.0
    assert result["loss"] == 3.0
    assert result["accuracy"] == 0.75  # 3 correct out of 4


def test_metric_accumulator_empty_returns_zero_loss():
    accumulator = MetricAccumulator()
    result = accumulator.compute()
    assert result["loss"] == 0.0
    assert result["accuracy"] == 0.0


def test_metric_accumulator_balanced_accuracy_included_when_requested():
    accumulator = MetricAccumulator()
    accumulator.update(1.0, 4, torch.tensor([0, 1, 1, 1]), torch.tensor([0, 0, 1, 1]))
    result = accumulator.compute(include_balanced_accuracy=True)
    assert "balanced_accuracy" in result

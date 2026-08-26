"""Tests for src.models.model_check: the offline CPU smoke-test utility.
Always pretrained=False here — no internet access required."""

import csv

from src.models.factory import MODEL_NAMES
from src.models.model_check import MODEL_PARAMETERS_COLUMNS, check_model

MINI_CONFIG = {
    "num_classes": 10,
    "image_size": 224,
    "baselines": {
        "resnet50": {"architecture": "resnet50", "pretrained": False, "num_classes": 10},
        "efficientnetb0": {"architecture": "efficientnet_b0", "pretrained": False, "num_classes": 10},
        "maxvit": {"architecture": "maxvit_tiny_tf_224", "pretrained": False, "num_classes": 10},
    },
}


def test_check_model_reports_correct_shapes_and_no_nans():
    result = check_model("resnet50", MINI_CONFIG, pretrained_override=False, batch_size=2)
    assert result.output_shape == (2, 10)
    assert result.feature_shape[0] == 2
    assert result.feature_shape[1] == result.feature_dimension
    assert result.has_nan_or_inf is False
    assert result.parameters_total > 0


def test_check_model_respects_pretrained_override_false():
    result = check_model("efficientnetb0", MINI_CONFIG, pretrained_override=False)
    assert result.architecture == "efficientnet_b0"


def test_run_model_check_writes_csv_from_config_dict(tmp_path, monkeypatch):
    import src.models.model_check as mc

    monkeypatch.setattr(mc, "load_config", lambda path: MINI_CONFIG)
    output_csv = tmp_path / "model_parameters.csv"

    results = mc.run_model_check(config_path="unused.yaml", check_pretrained=False, output_csv=output_csv)

    assert len(results) == 3
    assert output_csv.exists()

    with open(output_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert {r["model"] for r in rows} == set(MODEL_NAMES)
    for row in rows:
        assert set(row.keys()) == set(MODEL_PARAMETERS_COLUMNS)
        assert int(row["parameters_total"]) > 0
        assert int(row["feature_dimension"]) > 0


def test_run_model_check_all_baselines_pass_sanity(monkeypatch):
    import src.models.model_check as mc

    monkeypatch.setattr(mc, "load_config", lambda path: MINI_CONFIG)
    results = mc.run_model_check(config_path="unused.yaml", check_pretrained=False, output_csv=None)
    assert len(results) == len(MODEL_NAMES)
    for r in results:
        assert not r.has_nan_or_inf

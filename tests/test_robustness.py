"""Tests for src.evaluation.robustness (robustness evaluation of already-
frozen, already-finally-tested checkpoints - a later, additional test-time
analysis, entirely separate from Task 8's final held-out test evaluation).

CRITICAL: none of these tests ever run real inference on the real 438-image
data/manifests/test_original.csv or touch data/raw/ - every fixture builds
its own tiny, self-contained synthetic dataset/checkpoint under tmp_path,
exactly like tests/test_final_test.py already does. Integration tests use a
small, tmp_path-scoped evaluation.yaml (2-3 degradation conditions instead
of the real 17) purely to keep runtime fast; a dedicated test below checks
that the real configs/evaluation.yaml / DEFAULT_DEGRADATION_SEVERITIES still
carry the exact, unmodified fixed severities the project requires.
"""

import csv as csv_module
import json

import pytest
import torch
import yaml

from src.data.dataset import DatasetManifestError
from src.data.records import write_csv
from src.evaluation.robustness import (
    CLEAN_REFERENCE_LABEL,
    DEFAULT_DEGRADATION_SEVERITIES,
    RobustnessError,
    apply_degradation,
    run_robustness_evaluation,
)
from src.models.factory import create_model
from src.training.checkpointing import CheckpointIncompatibleError, build_checkpoint, save_checkpoint
from src.utils.config import load_config
from src.utils.hashing import hash_file
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]  # already alphabetical - matches build_class_to_idx's ordering
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]

# A tiny degradation table used by most integration tests, purely to keep
# them fast (3 conditions + clean_reference = 4 forward passes instead of
# the real 18). The exact, fixed, real severities are checked separately by
# test_real_config_and_defaults_use_exact_required_severities below.
FAST_DEGRADATIONS = {"brightness": [0.70], "gaussian_noise": [0.05]}


def _write_manifest(manifest_path, rows):
    write_csv(rows, MANIFEST_COLUMNS, manifest_path)


def _make_test_manifest_rows(raw_dir, canonical_classes, n_per_class=1, split="test", is_original="true"):
    rows = []
    for class_name in canonical_classes:
        for i in range(n_per_class):
            filename = f"{class_name}_{i}.jpg"
            rel_path = f"{class_name}/{filename}"
            make_image(raw_dir / rel_path)
            sample_id = f"sample_{class_name}_{i}"
            rows.append(
                {
                    "path": rel_path,
                    "class": class_name,
                    "split": split,
                    "original_id": sample_id,
                    "parent_original_id": sample_id,
                    "is_original": is_original,
                    "augmentation_type": "original" if is_original == "true" else "combined",
                }
            )
    return rows


def _write_min_evaluation_config(tmp_path, robustness_dir=None, degradations=None, batch_size=8, num_workers=0, config_name="evaluation.yaml"):
    config = {
        "robustness": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "degradations": degradations if degradations is not None else FAST_DEGRADATIONS,
        },
        "output_paths": {
            "robustness_dir": str(robustness_dir if robustness_dir is not None else tmp_path / "robustness_out"),
        },
    }
    config_path = tmp_path / config_name
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def _setup(tmp_path, model_name="resnet50", num_classes_override=None, n_per_class=1, manifest_rows=None):
    canonical_classes = CANONICAL_CLASSES[: num_classes_override or len(CANONICAL_CLASSES)]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows if manifest_rows is not None else _make_test_manifest_rows(raw_dir, canonical_classes, n_per_class)
    _write_manifest(manifests_dir / "test_original.csv", rows)

    include_aa = model_name == "aa_evidentnet"
    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=include_aa)
    evaluation_cfg = _write_min_evaluation_config(tmp_path)

    return dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir


def _build_checkpoint(tmp_path, model_name, models_cfg, num_classes, run_id="run"):
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model(model_name, models_config)
    checkpoint = build_checkpoint(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        epoch=7,
        best_metric=0.85,
        monitor_metric="val_macro_f1",
        training_config={},
        seed=42,
        model_name=model_name,
        architecture=model.architecture,
        num_classes=num_classes,
        dataset_manifest_hash="deadbeef",
        git_commit="abc123",
    )
    checkpoint_dir = tmp_path / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


# --- exact, fixed, required severities (no model, no inference) ---


def test_real_config_and_defaults_use_exact_required_severities():
    required = {
        "brightness": [0.70, 0.85, 1.15, 1.30],
        "contrast": [0.70, 0.85, 1.15, 1.30],
        "gaussian_noise": [0.02, 0.05, 0.10],
        "gaussian_blur": [0.5, 1.0, 2.0],
        "reduced_resolution": [168, 112, 56],
    }
    assert DEFAULT_DEGRADATION_SEVERITIES == required

    real_config = load_config("configs/evaluation.yaml")
    assert real_config["robustness"]["degradations"] == required


# --- apply_degradation: shape/range preservation for every required severity ---


@pytest.mark.parametrize("severity", DEFAULT_DEGRADATION_SEVERITIES["brightness"])
def test_brightness_preserves_shape_and_range(severity):
    image = torch.rand(3, 32, 32)
    out = apply_degradation(image, "brightness", severity)
    assert out.shape == image.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize("severity", DEFAULT_DEGRADATION_SEVERITIES["contrast"])
def test_contrast_preserves_shape_and_range(severity):
    image = torch.rand(3, 32, 32)
    out = apply_degradation(image, "contrast", severity)
    assert out.shape == image.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize("severity", DEFAULT_DEGRADATION_SEVERITIES["gaussian_noise"])
def test_gaussian_noise_preserves_shape_and_range(severity):
    image = torch.rand(3, 32, 32)
    out = apply_degradation(image, "gaussian_noise", severity, base_seed=42, sample_id="sample_1")
    assert out.shape == image.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize("severity", DEFAULT_DEGRADATION_SEVERITIES["gaussian_blur"])
def test_gaussian_blur_preserves_shape_and_range(severity):
    image = torch.rand(3, 32, 32)
    out = apply_degradation(image, "gaussian_blur", severity)
    assert out.shape == image.shape
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


@pytest.mark.parametrize("severity", DEFAULT_DEGRADATION_SEVERITIES["reduced_resolution"])
def test_reduced_resolution_preserves_shape_and_range_and_restores_original_size(severity):
    image = torch.rand(3, 224, 224)
    out = apply_degradation(image, "reduced_resolution", severity)
    assert out.shape == image.shape  # restored back to 224x224
    assert torch.all(out >= 0.0) and torch.all(out <= 1.0)


def test_all_default_degradation_severities_are_individually_accepted():
    image = torch.rand(3, 224, 224)
    for degradation, severities in DEFAULT_DEGRADATION_SEVERITIES.items():
        for severity in severities:
            out = apply_degradation(image, degradation, severity, base_seed=1, sample_id="s")
            assert out.shape == image.shape


# --- apply_degradation: rejection of invalid inputs ---


def test_apply_degradation_rejects_unknown_degradation_name():
    image = torch.rand(3, 32, 32)
    with pytest.raises(RobustnessError, match="Unknown degradation"):
        apply_degradation(image, "not_a_real_degradation", 1.0)


def test_apply_degradation_rejects_severity_not_in_fixed_table():
    image = torch.rand(3, 32, 32)
    with pytest.raises(RobustnessError, match="not one of the fixed"):
        apply_degradation(image, "brightness", 0.99)  # not one of 0.70/0.85/1.15/1.30


def test_apply_degradation_gaussian_noise_requires_seed_and_sample_id():
    image = torch.rand(3, 32, 32)
    with pytest.raises(RobustnessError, match="base_seed and sample_id"):
        apply_degradation(image, "gaussian_noise", 0.05)


def test_apply_degradation_rejects_non_chw_tensor():
    batched_image = torch.rand(2, 3, 32, 32)  # [B, C, H, W], not a single [C, H, W] image
    with pytest.raises(RobustnessError, match=r"\[C, H, W\]"):
        apply_degradation(batched_image, "brightness", 0.70)


# --- deterministic Gaussian noise ---


def test_gaussian_noise_is_deterministic_for_same_seed_sample_and_severity():
    image = torch.rand(3, 32, 32)
    out1 = apply_degradation(image, "gaussian_noise", 0.05, base_seed=42, sample_id="sample_1")
    out2 = apply_degradation(image, "gaussian_noise", 0.05, base_seed=42, sample_id="sample_1")
    assert torch.equal(out1, out2)


def test_gaussian_noise_differs_across_sample_ids():
    image = torch.rand(3, 32, 32)
    out1 = apply_degradation(image, "gaussian_noise", 0.05, base_seed=42, sample_id="sample_1")
    out2 = apply_degradation(image, "gaussian_noise", 0.05, base_seed=42, sample_id="sample_2")
    assert not torch.equal(out1, out2)


def test_gaussian_noise_differs_across_severities():
    image = torch.rand(3, 32, 32)
    out1 = apply_degradation(image, "gaussian_noise", 0.02, base_seed=42, sample_id="sample_1")
    out2 = apply_degradation(image, "gaussian_noise", 0.10, base_seed=42, sample_id="sample_1")
    assert not torch.equal(out1, out2)


def test_gaussian_noise_differs_across_base_seeds():
    image = torch.rand(3, 32, 32)
    out1 = apply_degradation(image, "gaussian_noise", 0.05, base_seed=1, sample_id="sample_1")
    out2 = apply_degradation(image, "gaussian_noise", 0.05, base_seed=2, sample_id="sample_1")
    assert not torch.equal(out1, out2)


# --- original source images are never modified ---


def test_apply_degradation_does_not_modify_input_tensor_in_place():
    image = torch.rand(3, 32, 32)
    original = image.clone()
    for degradation, severities in DEFAULT_DEGRADATION_SEVERITIES.items():
        apply_degradation(image, degradation, severities[0], base_seed=1, sample_id="s")
    assert torch.equal(image, original)


def test_source_raw_image_files_not_modified_by_full_evaluation(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    raw_dir = tmp_path / "raw"
    hashes_before = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}

    run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    hashes_after = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}
    assert hashes_after == hashes_before


def test_test_manifest_not_modified_by_full_evaluation(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    manifest_path = manifests_dir / "test_original.csv"
    hash_before = hash_file(manifest_path)

    run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    assert hash_file(manifest_path) == hash_before


# --- test-manifest safety enforcement ---


def test_rejects_manifest_with_non_test_split(tmp_path):
    raw_dir = tmp_path / "raw"
    bad_rows = _make_test_manifest_rows(raw_dir, CANONICAL_CLASSES[:2], n_per_class=1, split="train")
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(
        tmp_path, "resnet50", num_classes_override=2, manifest_rows=bad_rows
    )
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, 2)

    with pytest.raises(DatasetManifestError, match="split"):
        run_robustness_evaluation(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_rejects_manifest_with_augmented_sample(tmp_path):
    raw_dir = tmp_path / "raw"
    bad_rows = _make_test_manifest_rows(raw_dir, CANONICAL_CLASSES[:2], n_per_class=1, is_original="false")
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(
        tmp_path, "resnet50", num_classes_override=2, manifest_rows=bad_rows
    )
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, 2)

    with pytest.raises(DatasetManifestError):
        run_robustness_evaluation(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_missing_test_manifest_raises_robustness_error(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in CANONICAL_CLASSES[:2]}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)  # no test_original.csv written
    models_cfg = write_min_models_config(tmp_path, num_classes=2)
    evaluation_cfg = _write_min_evaluation_config(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, 2)

    with pytest.raises(RobustnessError, match="test_original.csv"):
        run_robustness_evaluation(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_unknown_model_name_raises_robustness_error(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    with pytest.raises(RobustnessError, match="Unknown model"):
        run_robustness_evaluation(
            model_name="not_a_real_model", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_incompatible_checkpoint_model_name_rejected(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    with pytest.raises(CheckpointIncompatibleError):
        run_robustness_evaluation(
            model_name="efficientnetb0", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


# --- eval-mode / no-grad / no-training guarantees ---


def test_evaluation_calls_model_eval(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    calls = []
    original_eval = torch.nn.Module.eval

    def spy_eval(self):
        calls.append(self)
        return original_eval(self)

    monkeypatch.setattr(torch.nn.Module, "eval", spy_eval)
    run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )
    assert len(calls) >= 1


def test_evaluation_never_constructs_an_optimizer(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed during robustness evaluation")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )


def test_evaluation_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur during robustness evaluation")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )


def test_checkpoint_file_is_never_modified(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    assert hash_file(checkpoint_path) == hash_before


# --- outputs: location, separation from final_test, schema ---


def test_outputs_written_under_results_robustness_not_raw_predictions(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    robustness_dir = tmp_path / "robustness_out"

    summary = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    from pathlib import Path

    metrics_path = Path(summary.metrics_path)
    metadata_path = Path(summary.metadata_path)
    assert metrics_path.is_file()
    assert metadata_path.is_file()
    assert robustness_dir in metrics_path.parents
    assert robustness_dir in metadata_path.parents
    assert "raw_predictions" not in str(metrics_path)
    assert "raw_predictions" not in str(metadata_path)


def test_metrics_csv_schema_one_row_per_condition(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=2)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    with open(summary.metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))

    expected_columns = {"model", "degradation", "severity", "n", "accuracy", "balanced_accuracy", "macro_f1", "mean_uncertainty"}
    assert expected_columns.issubset(rows[0].keys())

    # FAST_DEGRADATIONS has 2 conditions + 1 clean_reference = 3 rows.
    assert len(rows) == 3
    labels = {(r["degradation"], r["severity"]) for r in rows}
    assert (CLEAN_REFERENCE_LABEL, "") in labels
    assert ("brightness", "0.7") in labels
    assert ("gaussian_noise", "0.05") in labels

    n_expected = str(len(canonical_classes) * 2)
    for row in rows:
        assert row["model"] == "resnet50"
        assert row["n"] == n_expected
        assert row["mean_uncertainty"] == ""  # baseline model: no evidential uncertainty


def test_aa_evidentnet_mean_uncertainty_is_populated(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "aa_evidentnet", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "aa_evidentnet", models_cfg, len(canonical_classes))

    summary = run_robustness_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    with open(summary.metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == 3
    for row in rows:
        assert row["mean_uncertainty"] != ""
        uncertainty = float(row["mean_uncertainty"])
        assert 0.0 <= uncertainty <= 1.0


def test_metadata_includes_required_reproducibility_fields(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    expected_checkpoint_hash = hash_file(checkpoint_path)
    expected_manifest_hash = hash_file(manifests_dir / "test_original.csv")

    summary = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        seed=42, device_override="cpu", num_workers_override=0,
    )

    with open(summary.metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["model_name"] == "resnet50"
    assert metadata["checkpoint_path"] == str(checkpoint_path)
    assert metadata["checkpoint_sha256"] == expected_checkpoint_hash == summary.checkpoint_hash
    assert metadata["test_manifest_path"] == str(manifests_dir / "test_original.csv")
    assert metadata["test_manifest_sha256"] == expected_manifest_hash
    assert metadata["class_names"] == canonical_classes
    assert metadata["num_samples"] == summary.num_samples
    assert metadata["seed"] == 42
    assert metadata["device"] == "cpu"
    assert metadata["degradations"] == FAST_DEGRADATIONS
    assert "git_commit" in metadata
    assert "timestamp_utc" in metadata
    assert metadata["metrics_path"] == summary.metrics_path


def test_clean_reference_row_present_when_requested_and_absent_when_not(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    with_clean = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0, include_clean_reference=True,
    )
    without_clean = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0, include_clean_reference=False,
    )

    assert any(r["degradation"] == CLEAN_REFERENCE_LABEL for r in with_clean.rows)
    assert not any(r["degradation"] == CLEAN_REFERENCE_LABEL for r in without_clean.rows)


def test_two_invocations_get_distinct_non_overwriting_output_directories(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary_a = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )
    summary_b = run_robustness_evaluation(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    from pathlib import Path

    assert summary_a.robustness_run_id != summary_b.robustness_run_id
    assert summary_a.metrics_path != summary_b.metrics_path
    assert Path(summary_a.metrics_path).is_file()
    assert Path(summary_b.metrics_path).is_file()

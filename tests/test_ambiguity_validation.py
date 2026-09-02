"""Tests for src.evaluation.ambiguity_validation.run_ambiguity_validation -
the 7 validation-only analyses of sample-level ambiguity against
val_original.csv, using the SAME frozen reference checkpoint / prototypes
/ class matrix / margin normalization already built by
src.training.ambiguity_setup.build_learned_class_ambiguity.

CRITICAL: no test here touches the real dataset, data/raw/, or any real
manifest. Several tests explicitly construct a manifests_dir with NO
test_original.csv present at all (and no import in this module can even
reference one), proving zero dependency on the test set anywhere in this
development path.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.records import write_csv
from src.evaluation.ambiguity_validation import AmbiguityValidationError, run_ambiguity_validation
from src.models.factory import create_model
from src.training.ambiguity_setup import build_learned_class_ambiguity
from src.training.checkpointing import build_checkpoint, save_checkpoint
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]


def _make_rows(raw_dir, canonical_classes, split, n_per_class, prefix):
    rows = []
    for class_name in canonical_classes:
        for i in range(n_per_class):
            filename = f"{prefix}_{class_name}_{i}.jpg"
            rel_path = f"{class_name}/{filename}"
            make_image(raw_dir / rel_path)
            sample_id = f"sample_{prefix}_{class_name}_{i}"
            rows.append(
                {
                    "path": rel_path,
                    "class": class_name,
                    "split": split,
                    "original_id": sample_id,
                    "parent_original_id": sample_id,
                    "is_original": "true",
                    "augmentation_type": "original",
                }
            )
    return rows


def _setup(tmp_path, n_train=3, n_val=3, num_classes=4, write_val=True):
    canonical_classes = CANONICAL_CLASSES[:num_classes]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=num_classes, include_aa_evidentnet=True)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_csv(_make_rows(raw_dir, canonical_classes, "train", n_train, "train"), MANIFEST_COLUMNS, manifests_dir / "train_original.csv")
    if write_val:
        write_csv(_make_rows(raw_dir, canonical_classes, "val", n_val, "val"), MANIFEST_COLUMNS, manifests_dir / "val_original.csv")

    return dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir


def _build_checkpoint(tmp_path, models_cfg, num_classes, run_id="reference_run"):
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    checkpoint = build_checkpoint(
        model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), scheduler=None,
        epoch=5, best_metric=0.8, monitor_metric="val_macro_f1", training_config={}, seed=42,
        model_name="aa_evidentnet", architecture=model.architecture, num_classes=num_classes,
        dataset_manifest_hash="deadbeef", git_commit="abc123",
    )
    checkpoint_dir = tmp_path / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


def _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path):
    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    return build_learned_class_ambiguity(
        reference_checkpoint_path=checkpoint_path,
        reference_model_name="aa_evidentnet",
        models_config=models_config,
        dataset_config=dataset_config,
        canonical_classes=canonical_classes,
        raw_dir=raw_dir,
        processed_train_dir=tmp_path / "processed" / "train",
        train_manifest_path=manifests_dir / "train_original.csv",
        device=torch.device("cpu"),
        batch_size=8,
        num_workers=0,
    )


def _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir, **overrides):
    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    kwargs = dict(
        artifact=artifact,
        val_manifest_path=manifests_dir / "val_original.csv",
        dataset_config=dataset_config,
        models_config=models_config,
        raw_dir=raw_dir,
        processed_train_dir=tmp_path / "processed" / "train",
        device=torch.device("cpu"),
        output_dir=tmp_path / "ambiguity_out",
        batch_size=8,
        num_workers=0,
    )
    kwargs.update(overrides)
    return run_ambiguity_validation(**kwargs)


# --- end-to-end schema / outputs ---


def test_run_end_to_end_produces_expected_outputs(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=3)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert summary.num_val_samples == len(canonical_classes) * 3
    assert Path(summary.matrix_path).is_file()
    assert Path(summary.metrics_path).is_file()
    assert Path(summary.metadata_path).is_file()
    assert "raw_predictions" not in str(summary.matrix_path)
    assert "robustness" not in str(summary.matrix_path)
    assert "ood_uncertainty" not in str(summary.matrix_path)

    with open(summary.metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    assert metrics["run_id"] == summary.run_id
    assert set(metrics["quadrant_error_rates"].keys()) == {
        "low_ambiguity_low_uncertainty", "low_ambiguity_high_uncertainty",
        "high_ambiguity_low_uncertainty", "high_ambiguity_high_uncertainty",
    }
    assert sum(metrics["quadrant_counts"].values()) == summary.num_val_samples


def test_matrix_csv_has_class_name_rows_and_columns(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=2, n_val=2)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    import csv as csv_module

    with open(summary.matrix_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == len(canonical_classes)
    assert set(rows[0].keys()) == {"class_name"} | set(canonical_classes)
    for name in canonical_classes:
        row = next(r for r in rows if r["class_name"] == name)
        assert float(row[name]) == 0.0  # diagonal


# --- individual analyses ---


def test_error_detection_metrics_are_none_or_bounded(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    if summary.error_detection_auroc is not None:
        assert 0.0 <= summary.error_detection_auroc <= 1.0
    if summary.error_detection_auprc is not None:
        assert 0.0 <= summary.error_detection_auprc <= 1.0


def test_competing_class_hit_rate_is_bounded_or_none_when_zero_errors(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=2)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)
    if summary.num_errors == 0:
        assert summary.competing_class_hit_rate_among_errors is None
    else:
        assert 0.0 <= summary.competing_class_hit_rate_among_errors <= 1.0


def test_ambiguity_mean_median_correct_incorrect_are_well_formed(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    for value in (
        summary.ambiguity_mean_correct, summary.ambiguity_median_correct,
        summary.ambiguity_mean_incorrect, summary.ambiguity_median_incorrect,
    ):
        assert value is None or 0.0 <= value <= 1.0


def test_hard_pair_class_ambiguity_none_when_fixture_classes_never_in_hard_pairs(tmp_path):
    # CANONICAL_CLASSES ("Alpha".."Gamma") never match the real hard-pair
    # class names, so the hard-pair group is legitimately empty here.
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=3)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert summary.ambiguity_mean_hard_pair_classes is None
    assert summary.ambiguity_median_hard_pair_classes is None
    assert summary.ambiguity_mean_other_classes is not None


def test_hard_pair_class_ambiguity_computed_when_real_class_names_used(tmp_path):
    real_classes = ["Healthy", "Glaucoma", "Myopia", "Pterygium"]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in real_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=len(real_classes), include_aa_evidentnet=True)
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    write_csv(_make_rows(raw_dir, real_classes, "train", 3, "train"), MANIFEST_COLUMNS, manifests_dir / "train_original.csv")
    write_csv(_make_rows(raw_dir, real_classes, "val", 3, "val"), MANIFEST_COLUMNS, manifests_dir / "val_original.csv")

    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(real_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, real_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    # "Healthy" and "Glaucoma" are a real fixed hard pair -> the hard-pair
    # group must be non-empty.
    assert summary.ambiguity_mean_hard_pair_classes is not None


def test_spearman_vs_edl_uncertainty_is_bounded_or_none(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    if summary.spearman_ambiguity_vs_edl_uncertainty is not None:
        assert -1.0 <= summary.spearman_ambiguity_vs_edl_uncertainty <= 1.0


def test_quadrant_counts_sum_to_total_val_samples(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=5)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert sum(summary.quadrant_counts.values()) == summary.num_val_samples
    for rate in summary.quadrant_error_rates.values():
        assert rate is None or 0.0 <= rate <= 1.0


def test_class_matrix_never_built_from_confusion_matrix(tmp_path):
    # The learned matrix must be identical regardless of how many val
    # samples exist (or even whether run_ambiguity_validation is called at
    # all) - it is a pure function of train_original.csv's prototypes,
    # never of the validation confusion matrix computed inside this
    # analysis.
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=2)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    matrix_before = artifact.matrix_numpy.copy()

    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert np.array_equal(artifact.matrix_numpy, matrix_before)


def test_top5_pairs_reference_real_class_names(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=3)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert len(summary.top5_learned_ambiguity_pairs) <= 6  # C(4,2) = 6 possible pairs for 4 classes
    for pair in summary.top5_learned_ambiguity_pairs:
        assert pair["class_a"] in canonical_classes
        assert pair["class_b"] in canonical_classes


# --- leakage safeguards: no dependency on test_original.csv anywhere ---


def test_run_ambiguity_validation_has_no_test_manifest_parameter():
    # A structural (not textual) leakage safeguard: the function's only
    # manifest-path parameter is val_manifest_path - there is no
    # test_manifest_path (or similarly named) parameter for a caller to
    # even pass a test manifest through, unlike final_test.py/robustness.py
    # which legitimately accept one. (A plain substring search over the
    # module source is NOT used here since the module's own docstring
    # correctly explains, in prose, that it never touches
    # test_original.csv - that mention would be a false positive.)
    import inspect

    signature = inspect.signature(run_ambiguity_validation)
    param_names = set(signature.parameters.keys())
    assert "val_manifest_path" in param_names
    assert not any("test" in name.lower() for name in param_names)


def test_succeeds_with_no_test_manifest_present_at_all(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=3)
    assert not (manifests_dir / "test_original.csv").exists()
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)
    assert not (manifests_dir / "test_original.csv").exists()  # still never created
    assert summary.num_val_samples == len(canonical_classes) * 3


def test_raises_when_val_manifest_missing(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, write_val=False)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    with pytest.raises(AmbiguityValidationError, match="val_original.csv"):
        _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)


def test_val_manifest_and_raw_files_never_modified(tmp_path):
    from src.utils.hashing import hash_file

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=3)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    val_hash_before = hash_file(manifests_dir / "val_original.csv")
    raw_hashes_before = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}

    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert hash_file(manifests_dir / "val_original.csv") == val_hash_before
    raw_hashes_after = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}
    assert raw_hashes_after == raw_hashes_before


# --- eval-mode / no-grad guarantees on the reloaded reference model ---


def test_validation_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=2)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur during ambiguity validation")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)


def test_validation_never_constructs_an_optimizer(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=2)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed during ambiguity validation")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)


def test_two_invocations_get_distinct_non_overwriting_output_directories(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=3, n_val=2)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary_a = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)
    summary_b = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert summary_a.run_id != summary_b.run_id
    assert summary_a.metrics_path != summary_b.metrics_path
    assert Path(summary_a.metrics_path).is_file()
    assert Path(summary_b.metrics_path).is_file()

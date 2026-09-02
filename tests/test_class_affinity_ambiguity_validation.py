"""Tests for src.evaluation.class_affinity_ambiguity_validation (Phase 3) -
the validation-only analyses of class-affinity ambiguity, and the
three-way prototype/neighborhood/class-affinity comparison artifact.

CRITICAL: no test here touches the real dataset, data/raw/, or any real
manifest. Several tests explicitly construct a manifests_dir with NO
test_original.csv present at all, proving zero dependency on the test set.
"""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.records import write_csv
from src.evaluation.class_affinity_ambiguity_validation import (
    ClassAffinityAmbiguityValidationError,
    build_three_phase_comparison,
    run_class_affinity_ambiguity_validation,
)
from src.models.factory import create_model
from src.training.checkpointing import build_checkpoint, save_checkpoint
from src.training.class_affinity_ambiguity_setup import build_class_affinity_ambiguity
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]

# A fixed, distinct base color per class - see the identical, more
# detailed comment in tests/test_class_affinity_ambiguity_setup.py for
# why this replaces make_image()'s default per-path hash-derived color.
_CLASS_COLOR_PALETTE = [(220, 20, 20), (20, 180, 20), (20, 20, 220), (200, 200, 20)]


def _make_rows(raw_dir, canonical_classes, split, n_per_class, prefix):
    rows = []
    for class_idx, class_name in enumerate(canonical_classes):
        base_color = _CLASS_COLOR_PALETTE[class_idx % len(_CLASS_COLOR_PALETTE)]
        for i in range(n_per_class):
            filename = f"{prefix}_{class_name}_{i}.jpg"
            rel_path = f"{class_name}/{filename}"
            jitter = i % 20
            color = tuple(min(255, c + jitter) for c in base_color)
            make_image(raw_dir / rel_path, color=color)
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


def _setup(tmp_path, n_train=8, n_val=4, num_classes=4, write_val=True):
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
    # Fixed seed - see tests/test_class_affinity_ambiguity_setup.py's
    # identical comment: the train-derived scales this phase depends on
    # are sensitive to an untrained model's incidental embedding
    # structure, so the seed is pinned for reproducible pass/fail status.
    torch.manual_seed(42)
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


def _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path, m=5):
    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    return build_class_affinity_ambiguity(
        reference_checkpoint_path=checkpoint_path,
        reference_model_name="aa_evidentnet",
        models_config=models_config,
        dataset_config=dataset_config,
        canonical_classes=canonical_classes,
        raw_dir=raw_dir,
        processed_train_dir=tmp_path / "processed" / "train",
        train_manifest_path=manifests_dir / "train_original.csv",
        device=torch.device("cpu"),
        m=m,
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
    return run_class_affinity_ambiguity_validation(**kwargs)


# --- end-to-end schema / outputs ---


def test_run_end_to_end_produces_expected_outputs(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert summary.num_val_samples == len(canonical_classes) * 4
    assert summary.m == artifact.m
    assert summary.temperature == pytest.approx(0.1)
    assert summary.margin_scale == pytest.approx(artifact.margin_scale)
    assert Path(summary.matrix_path).is_file()
    assert Path(summary.metrics_path).is_file()
    assert Path(summary.metadata_path).is_file()
    assert Path(summary.matrix_path).name == "class_affinity_ambiguity_matrix.csv"
    assert Path(summary.metrics_path).name == "class_affinity_validation_metrics.json"
    assert Path(summary.metadata_path).name == "class_affinity_metadata.json"

    with open(summary.metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["test_data_used"] is False
    assert metadata["m"] == 5
    assert metadata["temperature"] == pytest.approx(0.1)
    assert "margin_scale" in metadata
    assert "boundary_gap_scale" in metadata
    assert "primary_score_formula" in metadata
    assert "label_aware_formula_analysis_only" in metadata
    assert metadata["training_confirmation"]


def test_error_detection_metrics_bounded_or_none(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=6)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    if summary.error_detection_auroc is not None:
        assert 0.0 <= summary.error_detection_auroc <= 1.0
    if summary.error_detection_auprc is not None:
        assert 0.0 <= summary.error_detection_auprc <= 1.0
    if summary.spearman_ambiguity_vs_edl_uncertainty is not None:
        assert -1.0 <= summary.spearman_ambiguity_vs_edl_uncertainty <= 1.0


def test_ambiguity_values_bounded(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=6)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    for value in (
        summary.ambiguity_mean_correct, summary.ambiguity_median_correct,
        summary.ambiguity_mean_incorrect, summary.ambiguity_median_incorrect,
    ):
        assert value is None or 0.0 <= value <= 1.0


def test_quadrant_counts_sum_to_total_val_samples(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=5)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert sum(summary.quadrant_counts.values()) == summary.num_val_samples


def test_hard_pair_ranks_reported_for_all_three_named_pairs(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert len(summary.hard_pair_ranks) == 3
    names = {(r["class_a"], r["class_b"]) for r in summary.hard_pair_ranks}
    assert ("Healthy", "Glaucoma") in names
    assert ("Disc Edema", "Glaucoma") in names
    assert ("Diabetic Retinopathy", "Central Serous Chorioretinopathy") in names
    # None of these class names exist in this fixture -> rank/value must be None, not a crash.
    for entry in summary.hard_pair_ranks:
        assert entry["rank"] is None
        assert entry["value"] is None


def test_matrix_never_mutated_by_validation(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    matrix_before = artifact.matrix_numpy.copy()

    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert np.array_equal(artifact.matrix_numpy, matrix_before)


# --- leakage safeguards ---


def test_succeeds_with_no_test_manifest_present_at_all(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    assert not (manifests_dir / "test_original.csv").exists()
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)
    assert not (manifests_dir / "test_original.csv").exists()
    assert summary.num_val_samples == len(canonical_classes) * 4


def test_raises_when_val_manifest_missing(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, write_val=False)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    with pytest.raises(ClassAffinityAmbiguityValidationError, match="val_original.csv"):
        _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)


def test_run_class_affinity_ambiguity_validation_has_no_test_manifest_parameter():
    import inspect

    signature = inspect.signature(run_class_affinity_ambiguity_validation)
    param_names = set(signature.parameters.keys())
    assert "val_manifest_path" in param_names
    assert not any("test" in name.lower() for name in param_names)


def test_val_manifest_and_raw_files_never_modified(tmp_path):
    from src.utils.hashing import hash_file

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    val_hash_before = hash_file(manifests_dir / "val_original.csv")
    raw_hashes_before = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}

    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert hash_file(manifests_dir / "val_original.csv") == val_hash_before
    assert {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))} == raw_hashes_before


def test_validation_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur during class-affinity ambiguity validation")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)


def test_validation_never_constructs_an_optimizer(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed during class-affinity ambiguity validation")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)


def test_two_invocations_get_distinct_non_overwriting_output_directories(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    summary_a = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)
    summary_b = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    assert summary_a.run_id != summary_b.run_id
    assert Path(summary_a.metrics_path).is_file()
    assert Path(summary_b.metrics_path).is_file()


# --- three-phase comparison artifact ---


def _write_fake_phase_artifacts(tmp_path, canonical_classes, prefix, seed):
    metrics_path = tmp_path / f"{prefix}_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "error_detection_auroc": 0.55,
                "error_detection_auprc": 0.2,
                "class_matrix_vs_confusion_spearman": 0.3,
                "competing_class_hit_rate_among_errors": 0.01,
            },
            f,
        )
    matrix_path = tmp_path / f"{prefix}_matrix.csv"
    rng = np.random.default_rng(seed)
    raw = rng.random((len(canonical_classes), len(canonical_classes)))
    raw = (raw + raw.T) / 2
    np.fill_diagonal(raw, 0.0)
    rows = []
    for i, name in enumerate(canonical_classes):
        row = {"class_name": name}
        for j, other in enumerate(canonical_classes):
            row[other] = float(raw[i, j])
        rows.append(row)
    write_csv(rows, ["class_name"] + list(canonical_classes), matrix_path)
    return metrics_path, matrix_path


def test_three_phase_comparison_contains_all_phases_and_named_pairs(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_train=8, n_val=4)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _build_artifact(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    summary = _run_validation(tmp_path, dataset_cfg, models_cfg, artifact, manifests_dir, raw_dir)

    p1_metrics, p1_matrix = _write_fake_phase_artifacts(tmp_path, canonical_classes, "p1", seed=1)
    p2_metrics, p2_matrix = _write_fake_phase_artifacts(tmp_path, canonical_classes, "p2", seed=2)
    output_path = tmp_path / "ambiguity_out" / "comparison.json"

    comparison = build_three_phase_comparison(
        phase1_metrics_path=p1_metrics, phase1_matrix_path=p1_matrix,
        phase2_metrics_path=p2_metrics, phase2_matrix_path=p2_matrix,
        phase3_summary=summary, phase3_matrix=artifact.matrix_numpy,
        canonical_classes=canonical_classes, output_path=output_path,
        named_pairs=[[canonical_classes[0], canonical_classes[1]]],
    )

    assert output_path.is_file()
    assert comparison["phase1_prototype"]["error_detection_auroc"] == pytest.approx(0.55)
    assert comparison["phase3_class_affinity"]["error_detection_auroc"] == summary.error_detection_auroc
    assert len(comparison["named_pair_comparison"]) == 1
    entry = comparison["named_pair_comparison"][0]
    assert entry["phase1_prototype_rank"] is not None
    assert entry["phase2_neighborhood_rank"] is not None
    assert entry["phase3_class_affinity_rank"] is not None
    assert "note" in comparison
    assert "better" not in comparison["note"] or "does not" in comparison["note"].lower() or "not claim" in comparison["note"].lower()

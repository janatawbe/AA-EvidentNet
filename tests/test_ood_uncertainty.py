"""Tests for src.evaluation.ood_uncertainty: a feature-distance (cosine)
OOD detector combined with AA-EvidentNet's own EDL uncertainty, calibrated
entirely from train_original.csv/val_original.csv and evaluated (frozen)
on test_original.csv plus every robustness.py condition.

CRITICAL: none of these tests ever run real inference on the real 438-
image data/manifests/test_original.csv, the real train/val manifests, or
data/raw/ - every fixture builds its own tiny, self-contained synthetic
dataset/checkpoint under tmp_path, exactly like tests/test_robustness.py
and tests/test_final_test.py already do. Integration tests use a small,
tmp_path-scoped evaluation.yaml (a reduced degradation table and weight
grid) purely to keep runtime fast.
"""

import csv as csv_module
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from src.data.dataset import DatasetManifestError
from src.data.records import write_csv
from src.evaluation.ood_uncertainty import (
    DEFAULT_WEIGHT_GRID,
    NormalizationParams,
    OODUncertaintyError,
    _apply_minmax,
    _corruption_strength,
    _error_detection_auprc,
    _error_detection_auroc,
    _fit_minmax,
    _spearman,
    calibrate_ood_uncertainty,
    compute_class_prototypes,
    nearest_prototype_cosine_distance,
    run_ood_uncertainty_evaluation,
    select_combine_weight,
)
from src.evaluation.robustness import CLEAN_REFERENCE_LABEL
from src.models.factory import create_model
from src.training.checkpointing import CheckpointIncompatibleError, build_checkpoint, save_checkpoint
from src.utils.hashing import hash_file
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]  # already alphabetical - matches build_class_to_idx's ordering
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]

# Small, fast fixture degradation table + weight grid for most integration
# tests. The real, fixed, required severities are the same ones
# tests/test_robustness.py already verifies against
# DEFAULT_DEGRADATION_SEVERITIES/configs/evaluation.yaml; reused unmodified
# here (this module never redefines its own severities).
FAST_DEGRADATIONS = {"brightness": [0.70], "gaussian_noise": [0.05]}
FAST_WEIGHT_GRID = [0.0, 0.5, 1.0, 2.0]


def _write_manifest(manifest_path, rows):
    write_csv(rows, MANIFEST_COLUMNS, manifest_path)


def _make_rows(raw_dir, canonical_classes, split, n_per_class, prefix, is_original="true"):
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
                    "is_original": is_original,
                    "augmentation_type": "original" if is_original == "true" else "combined",
                }
            )
    return rows


def _write_min_evaluation_config(
    tmp_path,
    ood_dir=None,
    degradations=None,
    weight_grid=None,
    coverage_levels=None,
    batch_size=8,
    num_workers=0,
    config_name="evaluation.yaml",
):
    config = {
        "ood_uncertainty": {
            "batch_size": batch_size,
            "num_workers": num_workers,
            "degradations": degradations if degradations is not None else FAST_DEGRADATIONS,
            "weight_grid": weight_grid if weight_grid is not None else FAST_WEIGHT_GRID,
        },
        "selective_prediction": {"coverage_levels": coverage_levels if coverage_levels is not None else [1.0, 0.5]},
        "output_paths": {"ood_uncertainty_dir": str(ood_dir if ood_dir is not None else tmp_path / "ood_out")},
    }
    config_path = tmp_path / config_name
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def _setup(
    tmp_path,
    num_classes_override=None,
    n_train_per_class=3,
    n_val_per_class=2,
    n_test_per_class=1,
    train_rows=None,
    val_rows=None,
    test_rows=None,
    write_train=True,
    write_val=True,
    write_test=True,
):
    canonical_classes = CANONICAL_CLASSES[: num_classes_override or len(CANONICAL_CLASSES)]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    if write_train:
        rows = train_rows if train_rows is not None else _make_rows(raw_dir, canonical_classes, "train", n_train_per_class, "train")
        _write_manifest(manifests_dir / "train_original.csv", rows)
    if write_val:
        rows = val_rows if val_rows is not None else _make_rows(raw_dir, canonical_classes, "val", n_val_per_class, "val")
        _write_manifest(manifests_dir / "val_original.csv", rows)
    if write_test:
        rows = test_rows if test_rows is not None else _make_rows(raw_dir, canonical_classes, "test", n_test_per_class, "test")
        _write_manifest(manifests_dir / "test_original.csv", rows)

    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=True)
    evaluation_cfg = _write_min_evaluation_config(tmp_path)

    return dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir


def _build_checkpoint(tmp_path, models_cfg, num_classes, run_id="run"):
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    checkpoint = build_checkpoint(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        epoch=7,
        best_metric=0.85,
        monitor_metric="val_macro_f1",
        training_config={},
        seed=42,
        model_name="aa_evidentnet",
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


def _load_model(models_cfg, num_classes):
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    return create_model("aa_evidentnet", models_config)


# --- cosine distance ---


def test_cosine_distance_identical_vectors_is_zero():
    embeddings = np.array([[1.0, 2.0, 3.0]])
    prototypes = np.array([[1.0, 2.0, 3.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_orthogonal_vectors_is_one():
    embeddings = np.array([[1.0, 0.0]])
    prototypes = np.array([[0.0, 1.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(1.0, abs=1e-6)


def test_cosine_distance_opposite_vectors_is_two():
    embeddings = np.array([[1.0, 0.0]])
    prototypes = np.array([[-1.0, 0.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(2.0, abs=1e-6)


def test_cosine_distance_scale_invariant():
    embeddings = np.array([[2.0, 0.0]])
    prototypes = np.array([[10.0, 0.0]])  # same direction, different magnitude
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(0.0, abs=1e-6)


def test_cosine_distance_picks_nearest_of_multiple_prototypes():
    embeddings = np.array([[1.0, 0.0]])
    prototypes = np.array([[0.0, 1.0], [1.0, 0.0001], [-1.0, 0.0]])
    distances = nearest_prototype_cosine_distance(embeddings, prototypes)
    assert distances[0] == pytest.approx(0.0, abs=1e-3)


def test_cosine_distance_rejects_mismatched_dims():
    embeddings = np.zeros((2, 3))
    prototypes = np.zeros((2, 4))
    with pytest.raises(OODUncertaintyError, match="embedding dim"):
        nearest_prototype_cosine_distance(embeddings, prototypes)


def test_cosine_distance_rejects_non_2d_input():
    with pytest.raises(OODUncertaintyError, match="2-D"):
        nearest_prototype_cosine_distance(np.zeros(3), np.zeros((1, 3)))


# --- normalization ---


def test_fit_and_apply_minmax_basic():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    params = _fit_minmax(values)
    assert params.min == 1.0 and params.max == 5.0
    normalized = _apply_minmax(values, params)
    assert normalized[0] == pytest.approx(0.0)
    assert normalized[-1] == pytest.approx(1.0)
    assert normalized[2] == pytest.approx(0.5)


def test_apply_minmax_floors_below_range_at_zero():
    params = NormalizationParams(min=1.0, max=5.0)
    normalized = _apply_minmax(np.array([-10.0, 0.0]), params)
    assert np.all(normalized == 0.0)


def test_apply_minmax_does_not_cap_above_one():
    params = NormalizationParams(min=0.0, max=1.0)
    normalized = _apply_minmax(np.array([5.0]), params)
    assert normalized[0] == pytest.approx(5.0)


def test_apply_minmax_degenerate_zero_span_returns_zeros():
    params = NormalizationParams(min=3.0, max=3.0)
    normalized = _apply_minmax(np.array([3.0, 3.0, 7.0]), params)
    assert np.all(normalized == 0.0)


# --- error-detection AUROC/AUPRC ---


def test_error_detection_auroc_perfect_separation():
    is_error = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])  # higher score -> more likely an error
    assert _error_detection_auroc(is_error, score) == pytest.approx(1.0)


def test_error_detection_auroc_undefined_when_no_errors():
    is_error = np.array([0, 0, 0])
    score = np.array([0.1, 0.2, 0.3])
    assert _error_detection_auroc(is_error, score) is None


def test_error_detection_auroc_undefined_when_all_errors():
    is_error = np.array([1, 1, 1])
    score = np.array([0.1, 0.2, 0.3])
    assert _error_detection_auroc(is_error, score) is None


def test_error_detection_auprc_undefined_when_no_errors():
    is_error = np.array([0, 0, 0])
    score = np.array([0.1, 0.2, 0.3])
    assert _error_detection_auprc(is_error, score) is None


def test_error_detection_auprc_defined_when_all_errors():
    # AUPRC only requires >=1 positive, unlike AUROC which also needs a negative.
    is_error = np.array([1, 1, 1])
    score = np.array([0.1, 0.2, 0.3])
    assert _error_detection_auprc(is_error, score) == pytest.approx(1.0)


# --- combine-weight selection ---


def test_select_combine_weight_picks_best_auroc():
    is_error = np.array([0, 0, 1, 1])
    normalized_edl = np.array([0.1, 0.1, 0.1, 0.1])  # uninformative on its own (constant)
    normalized_ood = np.array([0.1, 0.2, 0.8, 0.9])  # perfectly separates errors on its own
    weight, results = select_combine_weight(is_error, normalized_edl, normalized_ood, weight_grid=[0.0, 1.0, 5.0])
    # normalized_ood alone (weight=1.0) already reaches AUROC=1.0 (perfect
    # separation, since normalized_edl is constant and shifts nothing); a
    # larger weight cannot improve on a perfect score, so the tie-break
    # rule picks the smallest weight that already achieves it, not weight=0
    # (which drops the informative signal entirely and is not tied at 1.0).
    assert weight == 1.0
    assert results[0]["val_error_detection_auroc"] != pytest.approx(1.0)  # weight=0.0: uninformative-only, not perfect
    assert results[1]["val_error_detection_auroc"] == pytest.approx(1.0)  # weight=1.0: perfect
    assert results[2]["val_error_detection_auroc"] == pytest.approx(1.0)  # weight=5.0: also perfect, but not smallest
    assert len(results) == 3
    assert {r["weight"] for r in results} == {0.0, 1.0, 5.0}


def test_select_combine_weight_ties_break_toward_smallest():
    is_error = np.array([0, 1])
    normalized_edl = np.array([0.0, 1.0])  # already perfectly separates errors
    normalized_ood = np.array([0.0, 0.0])  # adds nothing -> every weight ties at AUROC=1.0
    weight, results = select_combine_weight(is_error, normalized_edl, normalized_ood, weight_grid=[0.0, 1.0, 2.0])
    assert weight == 0.0
    assert all(r["val_error_detection_auroc"] == pytest.approx(1.0) for r in results)


def test_select_combine_weight_falls_back_to_smallest_when_all_undefined():
    is_error = np.array([0, 0, 0])  # no errors anywhere -> AUROC undefined for every candidate
    normalized_edl = np.array([0.1, 0.2, 0.3])
    normalized_ood = np.array([0.1, 0.2, 0.3])
    weight, results = select_combine_weight(is_error, normalized_edl, normalized_ood, weight_grid=[0.0, 3.0])
    assert weight == 0.0
    assert all(r["val_error_detection_auroc"] is None for r in results)


def test_default_weight_grid_is_fixed():
    assert DEFAULT_WEIGHT_GRID == [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]


# --- corruption-strength convention ---


def test_corruption_strength_brightness_contrast_symmetric_around_one():
    assert _corruption_strength("brightness", 0.70) == pytest.approx(0.30)
    assert _corruption_strength("brightness", 1.30) == pytest.approx(0.30)
    assert _corruption_strength("contrast", 0.85) == pytest.approx(0.15)


def test_corruption_strength_noise_and_blur_are_identity():
    assert _corruption_strength("gaussian_noise", 0.05) == pytest.approx(0.05)
    assert _corruption_strength("gaussian_blur", 2.0) == pytest.approx(2.0)


def test_corruption_strength_reduced_resolution_inverts_target_size():
    assert _corruption_strength("reduced_resolution", 168, image_size=224) == pytest.approx(56.0)
    assert _corruption_strength("reduced_resolution", 56, image_size=224) > _corruption_strength(
        "reduced_resolution", 168, image_size=224
    )


def test_corruption_strength_rejects_unknown_degradation():
    with pytest.raises(OODUncertaintyError, match="no corruption-strength convention"):
        _corruption_strength("not_a_real_degradation", 1.0)


# --- Spearman correlation ---


def test_spearman_perfect_positive_correlation():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)


def test_spearman_perfect_negative_correlation():
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == pytest.approx(-1.0)


def test_spearman_undefined_with_fewer_than_two_points():
    assert _spearman([1], [1]) is None


def test_spearman_undefined_with_zero_variance():
    assert _spearman([1, 1, 1], [1, 2, 3]) is None


# --- compute_class_prototypes ---


def test_compute_class_prototypes_matches_hand_computed_mean(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, n_train_per_class=3
    )
    model = _load_model(models_cfg, len(canonical_classes))
    model.eval()

    from src.data.dataset import RetinalDataset
    from src.data.dataloaders import build_eval_dataloader
    from src.data.transforms import build_transforms_from_config
    import yaml as yaml_module

    dataset_config = yaml_module.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)
    train_dataset = RetinalDataset.from_manifest(
        manifests_dir / "train_original.csv", canonical_classes, raw_dir, tmp_path / "processed" / "train",
        transform=eval_transform, expected_split="train", require_all_original=True,
    )
    loader = build_eval_dataloader(train_dataset, batch_size=4, num_workers=0)

    prototypes, counts = compute_class_prototypes(model, loader, torch.device("cpu"), len(canonical_classes))
    assert prototypes.shape == (len(canonical_classes), model.embedding_dim)
    assert counts == [3] * len(canonical_classes)

    # Hand-recompute by running the model again over the same (deterministic,
    # non-augmented) samples and averaging manually.
    with torch.inference_mode():
        all_embeddings = []
        all_labels = []
        for batch in build_eval_dataloader(train_dataset, batch_size=4, num_workers=0):
            output = model(batch["image"], return_features=True)
            all_embeddings.append(output.embedding.numpy())
            all_labels.extend(batch["label"].numpy().tolist())
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    all_labels = np.array(all_labels)
    for k in range(len(canonical_classes)):
        expected = all_embeddings[all_labels == k].mean(axis=0)
        assert prototypes[k] == pytest.approx(expected, abs=1e-5)


def test_compute_class_prototypes_raises_for_missing_class(tmp_path):
    canonical_classes = CANONICAL_CLASSES[:3]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=True)

    # Deliberately omit all rows for the last class.
    rows = _make_rows(raw_dir, canonical_classes[:2], "train", 2, "train")
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    _write_manifest(manifests_dir / "train_original.csv", rows)

    model = _load_model(models_cfg, len(canonical_classes))
    model.eval()

    from src.data.dataset import RetinalDataset
    from src.data.dataloaders import build_eval_dataloader
    from src.data.transforms import build_transforms_from_config
    import yaml as yaml_module

    dataset_config = yaml_module.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)
    train_dataset = RetinalDataset.from_manifest(
        manifests_dir / "train_original.csv", canonical_classes, raw_dir, tmp_path / "processed" / "train",
        transform=eval_transform, expected_split="train", require_all_original=True,
    )
    loader = build_eval_dataloader(train_dataset, batch_size=4, num_workers=0)

    with pytest.raises(OODUncertaintyError, match=r"zero samples for class index"):
        compute_class_prototypes(model, loader, torch.device("cpu"), len(canonical_classes))


# --- calibration: train/val only, never test ---


def test_calibration_never_requires_test_manifest(tmp_path):
    """calibrate_ood_uncertainty must succeed using only train_original.csv
    and val_original.csv, with no test_original.csv present anywhere -
    proving calibration has zero dependency on the test manifest."""
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, write_test=False
    )
    assert not (manifests_dir / "test_original.csv").exists()

    model = _load_model(models_cfg, len(canonical_classes))
    model.eval()

    from src.data.transforms import build_transforms_from_config
    import yaml as yaml_module

    dataset_config = yaml_module.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)

    calibration = calibrate_ood_uncertainty(
        model=model, device=torch.device("cpu"), canonical_classes=canonical_classes, raw_dir=raw_dir,
        processed_train_dir=tmp_path / "processed" / "train", manifests_dir=manifests_dir,
        eval_transform=eval_transform, batch_size=8, num_workers=0, weight_grid=FAST_WEIGHT_GRID,
    )
    assert calibration.prototypes.shape[0] == len(canonical_classes)
    assert calibration.weight in FAST_WEIGHT_GRID


def test_calibration_raises_when_train_manifest_missing(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, write_train=False, write_test=False
    )
    model = _load_model(models_cfg, len(canonical_classes))
    model.eval()

    from src.data.transforms import build_transforms_from_config
    import yaml as yaml_module

    dataset_config = yaml_module.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)

    with pytest.raises(OODUncertaintyError, match="train_original.csv"):
        calibrate_ood_uncertainty(
            model=model, device=torch.device("cpu"), canonical_classes=canonical_classes, raw_dir=raw_dir,
            processed_train_dir=tmp_path / "processed" / "train", manifests_dir=manifests_dir,
            eval_transform=eval_transform, batch_size=8, num_workers=0,
        )


def test_calibration_raises_when_val_manifest_missing(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, write_val=False, write_test=False
    )
    model = _load_model(models_cfg, len(canonical_classes))
    model.eval()

    from src.data.transforms import build_transforms_from_config
    import yaml as yaml_module

    dataset_config = yaml_module.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    _, eval_transform = build_transforms_from_config(dataset_config)

    with pytest.raises(OODUncertaintyError, match="val_original.csv"):
        calibrate_ood_uncertainty(
            model=model, device=torch.device("cpu"), canonical_classes=canonical_classes, raw_dir=raw_dir,
            processed_train_dir=tmp_path / "processed" / "train", manifests_dir=manifests_dir,
            eval_transform=eval_transform, batch_size=8, num_workers=0,
        )


# --- run_ood_uncertainty_evaluation: end to end ---


def test_run_end_to_end_schema_and_outputs(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    summary = run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    metrics_path = Path(summary.metrics_path)
    risk_coverage_path = Path(summary.risk_coverage_path)
    figure_path = Path(summary.figure_path)
    metadata_path = Path(summary.metadata_path)
    assert metrics_path.is_file()
    assert risk_coverage_path.is_file()
    assert figure_path.is_file()
    assert metadata_path.is_file()

    ood_dir = Path(evaluation_cfg.parent) / "ood_out"
    assert ood_dir in metrics_path.parents
    assert "raw_predictions" not in str(metrics_path)
    assert "robustness" not in str(metrics_path)
    assert "tables" not in str(metrics_path)

    with open(metrics_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    # FAST_DEGRADATIONS has 2 conditions + 1 clean_reference = 3 rows.
    assert len(rows) == 3
    expected_columns = {
        "model", "degradation", "severity", "n", "accuracy",
        "mean_edl_uncertainty", "mean_ood_score", "mean_combined_score",
        "raw_mean_edl_uncertainty", "raw_mean_ood_distance",
        "edl_error_auroc", "edl_error_auprc",
        "ood_error_auroc", "ood_error_auprc",
        "combined_error_auroc", "combined_error_auprc",
    }
    assert expected_columns.issubset(rows[0].keys())
    for row in rows:
        assert row["model"] == "aa_evidentnet"

    with open(risk_coverage_path, newline="", encoding="utf-8") as f:
        rc_rows = list(csv_module.DictReader(f))
    assert {r["score_type"] for r in rc_rows} == {"edl", "ood", "combined"}
    assert {r["coverage"] for r in rc_rows} == {"1.0", "0.5"}

    with open(metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["model_name"] == "aa_evidentnet"
    assert metadata["prototype_distance_metric"] == "cosine"
    assert metadata["combine_weight"] == summary.weight
    assert metadata["combine_weight"] in FAST_WEIGHT_GRID
    assert len(metadata["combine_weight_grid_results"]) == len(FAST_WEIGHT_GRID)
    assert metadata["class_names"] == canonical_classes
    assert metadata["train_num_samples"] == len(canonical_classes) * 3
    assert metadata["val_num_samples"] == len(canonical_classes) * 2
    assert metadata["num_test_samples"] == len(canonical_classes) * 1
    assert "git_commit" in metadata
    assert "timestamp_utc" in metadata
    assert len(metadata["severity_correlations"]) > 0


def test_rejects_non_aa_evidentnet_model(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(OODUncertaintyError, match="only defined for"):
        run_ood_uncertainty_evaluation(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_missing_test_manifest_raises(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, write_test=False)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(OODUncertaintyError, match="test_original.csv"):
        run_ood_uncertainty_evaluation(
            model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_rejects_test_manifest_with_non_test_split(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, num_classes_override=2,
    )
    bad_test_rows = _make_rows(raw_dir, canonical_classes, "train", 1, "badtest")  # split="train", not "test"
    _write_manifest(manifests_dir / "test_original.csv", bad_test_rows)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(DatasetManifestError, match="split"):
        run_ood_uncertainty_evaluation(
            model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_rejects_test_manifest_with_augmented_sample(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, num_classes_override=2,
    )
    bad_test_rows = _make_rows(raw_dir, canonical_classes, "test", 1, "badtest", is_original="false")
    _write_manifest(manifests_dir / "test_original.csv", bad_test_rows)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(DatasetManifestError):
        run_ood_uncertainty_evaluation(
            model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


def test_incompatible_checkpoint_rejected(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, num_classes=99)  # deliberately wrong

    with pytest.raises(CheckpointIncompatibleError):
        run_ood_uncertainty_evaluation(
            model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
            device_override="cpu", num_workers_override=0,
        )


# --- eval-mode / no-grad / no-training guarantees ---


def test_evaluation_calls_model_eval(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    calls = []
    original_eval = torch.nn.Module.eval

    def spy_eval(self):
        calls.append(self)
        return original_eval(self)

    monkeypatch.setattr(torch.nn.Module, "eval", spy_eval)
    run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )
    assert len(calls) >= 1


def test_evaluation_never_constructs_an_optimizer(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed during ood_uncertainty evaluation")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )


def test_evaluation_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur during ood_uncertainty evaluation")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )


def test_checkpoint_file_never_modified(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    assert hash_file(checkpoint_path) == hash_before


def test_raw_image_files_and_manifests_never_modified(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    hashes_before = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}
    manifest_hashes_before = {
        name: hash_file(manifests_dir / name) for name in ("train_original.csv", "val_original.csv", "test_original.csv")
    }

    run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    hashes_after = {p: hash_file(p) for p in sorted(raw_dir.rglob("*.jpg"))}
    manifest_hashes_after = {
        name: hash_file(manifests_dir / name) for name in ("train_original.csv", "val_original.csv", "test_original.csv")
    }
    assert hashes_after == hashes_before
    assert manifest_hashes_after == manifest_hashes_before


def test_two_invocations_get_distinct_non_overwriting_output_directories(tmp_path):
    dataset_cfg, models_cfg, evaluation_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path)
    checkpoint_path = _build_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    summary_a = run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )
    summary_b = run_ood_uncertainty_evaluation(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, evaluation_config_path=evaluation_cfg,
        device_override="cpu", num_workers_override=0,
    )

    assert summary_a.run_id != summary_b.run_id
    assert summary_a.metrics_path != summary_b.metrics_path
    assert Path(summary_a.metrics_path).is_file()
    assert Path(summary_b.metrics_path).is_file()

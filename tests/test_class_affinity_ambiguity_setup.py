"""Tests for src.training.class_affinity_ambiguity_setup.build_class_affinity_ambiguity
(Phase 3) - the one-time orchestration that builds the frozen class-
affinity ambiguity matrix and train-derived scales from a REFERENCE
checkpoint's embeddings over train_original.csv.

CRITICAL: no test here touches the real dataset, data/raw/, or any real
manifest - every fixture builds its own tiny, self-contained synthetic
dataset/checkpoint under tmp_path. Several tests explicitly construct a
manifests_dir containing ONLY train_original.csv (no val_original.csv or
test_original.csv at all) to prove zero dependency on anything else.
"""

import numpy as np
import pytest
import torch
import yaml

from src.data.records import write_csv
from src.losses.class_affinity_ambiguity import compute_class_affinities, compute_class_affinity_matrix
from src.models.factory import create_model
from src.training.checkpointing import CheckpointIncompatibleError, build_checkpoint, save_checkpoint
from src.training.class_affinity_ambiguity_setup import (
    ClassAffinityAmbiguitySetupError,
    build_class_affinity_ambiguity,
)
from src.utils.hashing import hash_file
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]

# A fixed, distinct base color per class (small per-sample jitter added
# below) - deliberately NOT relying on make_image()'s default per-path
# hash-derived color, which varies essentially randomly PER FILE (not per
# class) and gives an untrained reference model too weak/inconsistent a
# signal: empirically, boundary_gap_scale (which - unlike margin_scale -
# is not sign-constrained by construction) would occasionally come out
# numerically negative depending on incidental per-path color noise, even
# with a fixed model-init seed. A real, class-consistent color signal
# makes these tests' pass/fail status depend on the actual code under
# test, not on incidental hash collisions.
_CLASS_COLOR_PALETTE = [(220, 20, 20), (20, 180, 20), (20, 20, 220), (200, 200, 20)]


def _make_rows(raw_dir, canonical_classes, n_per_class, prefix):
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
                    "split": "train",
                    "original_id": sample_id,
                    "parent_original_id": sample_id,
                    "is_original": "true",
                    "augmentation_type": "original",
                }
            )
    return rows


def _setup(tmp_path, n_per_class=8, num_classes=4):
    canonical_classes = CANONICAL_CLASSES[:num_classes]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=num_classes, include_aa_evidentnet=True)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = _make_rows(raw_dir, canonical_classes, n_per_class, "train")
    write_csv(rows, MANIFEST_COLUMNS, manifests_dir / "train_original.csv")

    return dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir


def _build_reference_checkpoint(tmp_path, models_cfg, num_classes, run_id="reference_run"):
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    # Fixed seed for the untrained reference model's weight initialization:
    # the class-affinity scales derived below (margin_scale, and
    # especially boundary_gap_scale, which is not sign-constrained the
    # way margin always is) are sensitive to whatever incidental
    # structure an UNTRAINED model's random embedding space happens to
    # have - verified empirically across 20 fixed seeds with this exact
    # fixture (all succeeded), so any fixed seed is fine; 42 matches this
    # project's usual seed convention. Without pinning it, these tests'
    # pass/fail status would depend on unrelated global RNG state
    # consumed by whichever other tests happened to run first.
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


def _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path, **overrides):
    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    kwargs = dict(
        reference_checkpoint_path=checkpoint_path,
        reference_model_name="aa_evidentnet",
        models_config=models_config,
        dataset_config=dataset_config,
        canonical_classes=canonical_classes,
        raw_dir=raw_dir,
        processed_train_dir=tmp_path / "processed" / "train",
        train_manifest_path=manifests_dir / "train_original.csv",
        device=torch.device("cpu"),
        m=5,
        batch_size=8,
        num_workers=0,
    )
    kwargs.update(overrides)
    return build_class_affinity_ambiguity(**kwargs)


# --- end-to-end correctness ---


def test_artifact_matrix_matches_independently_computed_matrix(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert artifact.matrix_numpy.shape == (len(canonical_classes), len(canonical_classes))
    expected = compute_class_affinity_matrix(artifact.train_affinities_self_excluded, artifact.train_labels, len(canonical_classes))
    assert np.array_equal(artifact.matrix_numpy, expected)
    assert artifact.m == 5
    assert artifact.temperature == pytest.approx(0.1)
    assert artifact.scale_percentile == pytest.approx(95.0)
    assert artifact.num_train_samples == len(canonical_classes) * 8


def test_artifact_scales_are_positive_and_match_independent_recomputation(tmp_path):
    from src.losses.class_affinity_ambiguity import compute_label_aware_boundary_gap, compute_top_affinities, fit_boundary_gap_scale, fit_margin_scale

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert artifact.margin_scale > 0
    assert artifact.boundary_gap_scale > 0

    top = compute_top_affinities(artifact.train_affinities_self_excluded)
    expected_margin_scale = fit_margin_scale(top.raw_margin)
    assert artifact.margin_scale == pytest.approx(expected_margin_scale)

    boundary_gap = compute_label_aware_boundary_gap(artifact.train_affinities_self_excluded, artifact.train_labels)
    expected_boundary_scale = fit_boundary_gap_scale(boundary_gap)
    assert artifact.boundary_gap_scale == pytest.approx(expected_boundary_scale)


def test_result_is_deterministic_across_two_calls(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact_a = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    artifact_b = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.array_equal(artifact_a.matrix_numpy, artifact_b.matrix_numpy)
    assert artifact_a.margin_scale == pytest.approx(artifact_b.margin_scale)
    assert artifact_a.boundary_gap_scale == pytest.approx(artifact_b.boundary_gap_scale)


def test_matrix_symmetric_bounded_zero_diagonal(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.allclose(artifact.matrix_numpy, artifact.matrix_numpy.T)
    assert np.all(np.diag(artifact.matrix_numpy) == 0.0)
    assert np.all(artifact.matrix_numpy >= 0.0) and np.all(artifact.matrix_numpy <= 1.0)


def test_metadata_records_m_and_temperature(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path, m=5)
    assert artifact.m == 5
    assert artifact.temperature == pytest.approx(0.1)


# --- own training sample excluded from same-class affinity ---


def test_train_affinities_exclude_own_sample_from_same_class(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    # Recompute WITHOUT exclusion and confirm every sample's own-class
    # affinity is strictly higher without exclusion (self-similarity=1.0
    # would otherwise inflate it) - proof self-exclusion actually took effect.
    without_exclusion = compute_class_affinities(
        artifact.train_embeddings, artifact.train_embeddings, artifact.train_labels, len(canonical_classes), m=artifact.m, exclude_self=False
    )
    own_class_excluded = artifact.train_affinities_self_excluded[np.arange(len(artifact.train_labels)), artifact.train_labels]
    own_class_unexcluded = without_exclusion[np.arange(len(artifact.train_labels)), artifact.train_labels]
    assert np.all(own_class_unexcluded >= own_class_excluded)
    assert np.any(own_class_unexcluded > own_class_excluded)


# --- reference model stays frozen / read-only ---


def test_reference_model_eval_is_called(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    calls = []
    original_eval = torch.nn.Module.eval

    def spy_eval(self):
        calls.append(self)
        return original_eval(self)

    monkeypatch.setattr(torch.nn.Module, "eval", spy_eval)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert len(calls) >= 1


def test_reference_model_never_constructs_an_optimizer(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed while building the class-affinity ambiguity matrix")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_reference_model_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur while building the class-affinity ambiguity matrix")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_reference_model_parameters_marked_requires_grad_false(tmp_path, monkeypatch):
    import src.training.class_affinity_ambiguity_setup as setup_module

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    captured_models = []
    original_create_model = setup_module.create_model

    def spy_create_model(*a, **kw):
        model = original_create_model(*a, **kw)
        captured_models.append(model)
        return model

    monkeypatch.setattr(setup_module, "create_model", spy_create_model)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert len(captured_models) == 1
    for parameter in captured_models[0].parameters():
        assert parameter.requires_grad is False


def test_checkpoint_file_never_modified(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert hash_file(checkpoint_path) == hash_before


# --- train_original.csv only, no dependency on anything else ---


def test_succeeds_with_no_val_or_test_manifest_present(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    assert not (manifests_dir / "val_original.csv").exists()
    assert not (manifests_dir / "test_original.csv").exists()
    assert not (manifests_dir / "train_balanced.csv").exists()

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert artifact.train_manifest_path.endswith("train_original.csv")


def test_train_manifest_sha256_matches_train_original_file(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert artifact.train_manifest_sha256 == hash_file(manifests_dir / "train_original.csv")


def test_raises_when_train_original_manifest_missing(tmp_path):
    canonical_classes = CANONICAL_CLASSES[:3]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=True)
    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)  # deliberately empty
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(ClassAffinityAmbiguitySetupError, match="train_original.csv"):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_no_test_original_csv_parameter_exists():
    import inspect

    params = set(inspect.signature(build_class_affinity_ambiguity).parameters.keys())
    assert not any("test" in p.lower() for p in params)


# --- checkpoint compatibility / degenerate-scale propagation ---


def test_incompatible_reference_checkpoint_rejected(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, num_classes=99)

    with pytest.raises(CheckpointIncompatibleError):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

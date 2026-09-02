"""Tests for
src.training.class_affinity_ambiguity_cs_supcon_setup.build_class_affinity_ambiguity_for_cs_supcon
(feature/learned-ambiguity, Phase 3-experimental) - the thin wrapper that
builds Phase 3's frozen class-affinity matrix (via the EXISTING, unmodified
src.training.class_affinity_ambiguity_setup.build_class_affinity_ambiguity)
and wraps it as a non-trainable buffer ready for
CSSupConLoss.set_learned_ambiguity_matrix(...).

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
from src.losses.cs_supcon import CSSupConLoss
from src.models.factory import create_model
from src.training.checkpointing import build_checkpoint, save_checkpoint
from src.training.class_affinity_ambiguity_cs_supcon_setup import (
    MATRIX_CONSTRUCTION_METHOD,
    ClassAffinityAmbiguityCSSupConArtifact,
    build_class_affinity_ambiguity_for_cs_supcon,
)
from src.training.class_affinity_ambiguity_setup import (
    ClassAffinityAmbiguitySetupError,
    build_class_affinity_ambiguity,
)
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]

# Same rationale as tests/test_class_affinity_ambiguity_setup.py's
# identical fixture: a fixed, class-consistent color (not
# make_image()'s default per-path hash color) so an untrained reference
# model's train-derived scales are not sensitive to incidental per-file
# color noise.
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


def _setup(tmp_path, n_per_class=8, num_classes=4, write_test_manifest=False):
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
    if write_test_manifest:
        # Deliberately only used by the one test that asserts this file's
        # mere presence changes nothing - never read by the function under
        # test itself.
        write_csv(rows, MANIFEST_COLUMNS, manifests_dir / "test_original.csv")

    return dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir


def _build_reference_checkpoint(tmp_path, models_cfg, num_classes, run_id="reference_run"):
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
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
        batch_size=8,
        num_workers=0,
    )
    kwargs.update(overrides)
    return build_class_affinity_ambiguity_for_cs_supcon(**kwargs)


# --- 1. reusability: matrix reused verbatim from the existing, unmodified Phase 3 setup ---


def test_matrix_matches_the_unmodified_phase3_setup_function_exactly(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    dataset_config = yaml.safe_load(dataset_cfg.read_text(encoding="utf-8"))
    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))

    direct_artifact = build_class_affinity_ambiguity(
        reference_checkpoint_path=checkpoint_path, reference_model_name="aa_evidentnet",
        models_config=models_config, dataset_config=dataset_config, canonical_classes=canonical_classes,
        raw_dir=raw_dir, processed_train_dir=tmp_path / "processed" / "train",
        train_manifest_path=manifests_dir / "train_original.csv", device=torch.device("cpu"),
        batch_size=8, num_workers=0,
    )
    wrapped_artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.array_equal(wrapped_artifact.matrix_numpy, direct_artifact.matrix_numpy)
    assert wrapped_artifact.m == direct_artifact.m == 5
    assert wrapped_artifact.reference_checkpoint_sha256 == direct_artifact.reference_checkpoint_sha256


def test_can_be_installed_directly_into_cs_supcon_loss(tmp_path):
    """Test requirement #1: the Phase 3 learned matrix can be supplied to
    CS-SupCon training."""
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    loss_fn = CSSupConLoss(ambiguity_source="learned_class_affinity", ambiguity_scale=1.0)
    loss_fn.set_learned_ambiguity_matrix(artifact.matrix_buffer)

    embeddings = torch.randn(8, 16, requires_grad=True)
    labels = torch.randint(0, len(canonical_classes), (8,))
    loss = loss_fn(embeddings, labels, num_classes=len(canonical_classes))
    assert torch.isfinite(loss)
    loss.backward()
    assert embeddings.grad is not None


# --- 2/3/4. matrix properties ---


def test_matrix_is_symmetric(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert np.allclose(artifact.matrix_numpy, artifact.matrix_numpy.T)
    assert torch.allclose(artifact.matrix_buffer, artifact.matrix_buffer.T)


def test_matrix_diagonal_is_zero(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert np.all(np.diag(artifact.matrix_numpy) == 0.0)
    assert torch.all(torch.diag(artifact.matrix_buffer) == 0.0)


def test_matrix_values_are_in_zero_one_range(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert np.all(artifact.matrix_numpy >= 0.0)
    assert np.all(artifact.matrix_numpy <= 1.0)


# --- 5. frozen / non-trainable ---


def test_matrix_buffer_is_frozen_non_trainable(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert artifact.matrix_buffer.requires_grad is False


def test_matrix_buffer_is_not_registered_as_a_parameter_once_installed(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    loss_fn = CSSupConLoss(ambiguity_source="learned_class_affinity", ambiguity_scale=1.0)
    loss_fn.set_learned_ambiguity_matrix(artifact.matrix_buffer)

    assert len(list(loss_fn.parameters())) == 0  # never trainable, never touched by an optimizer
    assert "learned_ambiguity_matrix" in dict(loss_fn.named_buffers())


# --- 11/12. train_original.csv only, no test manifest ---


def test_matrix_construction_uses_train_original_only_even_when_test_original_exists(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(
        tmp_path, n_per_class=8, write_test_manifest=True
    )
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact_with_test_present = _call(
        tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path
    )

    (manifests_dir / "test_original.csv").unlink()
    artifact_without_test_present = _call(
        tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path
    )

    assert np.array_equal(artifact_with_test_present.matrix_numpy, artifact_without_test_present.matrix_numpy)


def test_no_test_manifest_parameter_exists():
    import inspect

    signature = inspect.signature(build_class_affinity_ambiguity_for_cs_supcon)
    param_names = set(signature.parameters.keys())
    assert "train_manifest_path" in param_names
    assert not any("test" in name.lower() for name in param_names)


def test_raises_when_train_original_manifest_missing(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    (manifests_dir / "train_original.csv").unlink()

    with pytest.raises(ClassAffinityAmbiguitySetupError, match="train_original.csv"):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


# --- 14. constructed once, not per-batch/epoch (determinism across calls
# stands in for "no hidden internal recomputation loop") ---


def test_construction_is_deterministic_across_repeated_calls(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact_a = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    artifact_b = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.array_equal(artifact_a.matrix_numpy, artifact_b.matrix_numpy)


def test_function_takes_no_epoch_or_step_argument():
    """A structural guard against ever accidentally calling this per-batch
    or per-epoch: it has no epoch/step-shaped parameter at all."""
    import inspect

    signature = inspect.signature(build_class_affinity_ambiguity_for_cs_supcon)
    param_names = {name.lower() for name in signature.parameters}
    assert not any("epoch" in name or "step" in name for name in param_names)


# --- 15. reference checkpoint parameters remain frozen ---


def test_reference_checkpoint_parameters_remain_frozen_after_construction(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    spy_models = []
    original_create_model = create_model

    import src.training.class_affinity_ambiguity_setup as setup_module

    def spy_create_model(*args, **kwargs):
        model = original_create_model(*args, **kwargs)
        spy_models.append(model)
        return model

    import unittest.mock as mock

    with mock.patch.object(setup_module, "create_model", side_effect=spy_create_model):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert len(spy_models) == 1
    reference_model = spy_models[0]
    assert reference_model.training is False  # eval mode
    assert all(not p.requires_grad for p in reference_model.parameters())


def test_reference_checkpoint_file_itself_is_never_modified(tmp_path):
    from src.utils.hashing import hash_file

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert hash_file(checkpoint_path) == hash_before


# --- 17. metadata / provenance fields present on the artifact ---


def test_artifact_records_full_provenance(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=8)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert isinstance(artifact, ClassAffinityAmbiguityCSSupConArtifact)
    assert artifact.reference_checkpoint_path == str(checkpoint_path)
    assert len(artifact.reference_checkpoint_sha256) == 64  # sha256 hex digest
    assert artifact.reference_model_name == "aa_evidentnet"
    assert artifact.train_manifest_path.endswith("train_original.csv")
    assert len(artifact.train_manifest_sha256) == 64
    assert artifact.canonical_classes == canonical_classes
    assert artifact.num_train_samples == len(canonical_classes) * 8
    assert artifact.m == 5
    assert artifact.matrix_construction_method == MATRIX_CONSTRUCTION_METHOD
    assert "train_original.csv" in artifact.matrix_construction_method

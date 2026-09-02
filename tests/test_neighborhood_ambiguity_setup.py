"""Tests for src.training.neighborhood_ambiguity_setup.build_neighborhood_class_ambiguity
(Phase 2) - the one-time orchestration that builds the frozen cross-class
neighborhood ambiguity matrix from a REFERENCE checkpoint's embeddings
over train_original.csv.

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
from src.losses.neighborhood_ambiguity import compute_neighborhood_class_matrix
from src.models.factory import create_model
from src.training.checkpointing import CheckpointIncompatibleError, build_checkpoint, save_checkpoint
from src.training.neighborhood_ambiguity_setup import NeighborhoodAmbiguitySetupError, build_neighborhood_class_ambiguity
from src.utils.hashing import hash_file
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]


def _make_rows(raw_dir, canonical_classes, n_per_class, prefix):
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
                    "split": "train",
                    "original_id": sample_id,
                    "parent_original_id": sample_id,
                    "is_original": "true",
                    "augmentation_type": "original",
                }
            )
    return rows


def _setup(tmp_path, n_per_class=6, num_classes=4):
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
        k=5,
        batch_size=8,
        num_workers=0,
    )
    kwargs.update(overrides)
    return build_neighborhood_class_ambiguity(**kwargs)


# --- end-to-end correctness ---


def test_artifact_matrix_matches_independently_computed_matrix(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert artifact.matrix_numpy.shape == (len(canonical_classes), len(canonical_classes))
    expected = compute_neighborhood_class_matrix(artifact.neighbors, artifact.train_labels, len(canonical_classes))
    assert np.array_equal(artifact.matrix_numpy, expected)
    assert artifact.k == 5
    assert artifact.num_train_samples == len(canonical_classes) * 6


def test_artifact_train_embeddings_shape_matches_model_embedding_dim(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert artifact.train_embeddings.shape[0] == len(canonical_classes) * 6
    assert artifact.train_labels.shape[0] == len(canonical_classes) * 6


def test_matrix_symmetric_bounded_zero_diagonal(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.allclose(artifact.matrix_numpy, artifact.matrix_numpy.T)
    assert np.all(np.diag(artifact.matrix_numpy) == 0.0)
    assert np.all(artifact.matrix_numpy >= 0.0) and np.all(artifact.matrix_numpy <= 1.0)


def test_result_is_deterministic_across_two_calls(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact_a = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    artifact_b = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.array_equal(artifact_a.matrix_numpy, artifact_b.matrix_numpy)
    assert np.array_equal(artifact_a.train_embeddings, artifact_b.train_embeddings)


# --- reference model stays frozen / read-only ---


def test_reference_model_eval_is_called(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
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
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed while building the neighborhood ambiguity matrix")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_reference_model_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur while building the neighborhood ambiguity matrix")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_reference_model_parameters_marked_requires_grad_false(tmp_path, monkeypatch):
    import src.training.neighborhood_ambiguity_setup as setup_module

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
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
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert hash_file(checkpoint_path) == hash_before


# --- train_original.csv only, no dependency on anything else ---


def test_succeeds_with_no_val_or_test_manifest_present(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    assert not (manifests_dir / "val_original.csv").exists()
    assert not (manifests_dir / "test_original.csv").exists()
    assert not (manifests_dir / "train_balanced.csv").exists()

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert artifact.train_manifest_path.endswith("train_original.csv")


def test_train_manifest_sha256_matches_train_original_file(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
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

    with pytest.raises(NeighborhoodAmbiguitySetupError, match="train_original.csv"):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


# --- checkpoint compatibility / k validation ---


def test_incompatible_reference_checkpoint_rejected(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=6)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, num_classes=99)

    with pytest.raises(CheckpointIncompatibleError):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_k_too_large_for_dataset_raises_setup_error(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    # n_per_class=2, 4 classes -> only 6 cross-class candidates per sample;
    # k=20 must fail loudly, never silently truncate.
    with pytest.raises(NeighborhoodAmbiguitySetupError, match="eligible"):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path, k=20)

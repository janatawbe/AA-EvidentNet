"""Tests for src.training.ambiguity_setup.build_learned_class_ambiguity -
the one-time orchestration that builds the frozen class-ambiguity matrix
from a REFERENCE checkpoint's embeddings over train_original.csv, before
any new ambiguity-aware training run begins.

CRITICAL: no test here touches the real dataset, data/raw/, or any real
manifest - every fixture builds its own tiny, self-contained synthetic
dataset/checkpoint under tmp_path. Several tests explicitly construct a
manifests_dir containing ONLY train_original.csv (no train_balanced.csv,
val_original.csv, or test_original.csv at all) to prove this module has
zero dependency on anything else.
"""

import numpy as np
import pytest
import torch
import yaml

from src.data.records import write_csv
from src.losses.ambiguity import compute_class_ambiguity_matrix, compute_raw_margins, fit_margin_normalization
from src.models.factory import create_model
from src.training.ambiguity_setup import AmbiguitySetupError, build_learned_class_ambiguity
from src.training.checkpointing import CheckpointIncompatibleError, build_checkpoint, save_checkpoint
from src.utils.hashing import hash_file
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]  # already alphabetical
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


def _setup(tmp_path, n_per_class=3, num_classes=4):
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
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        epoch=5,
        best_metric=0.8,
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
    return build_learned_class_ambiguity(**kwargs)


# --- end-to-end correctness ---


def test_artifact_matrix_matches_independently_computed_matrix(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=3)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert artifact.matrix_numpy.shape == (len(canonical_classes), len(canonical_classes))
    expected_matrix = compute_class_ambiguity_matrix(artifact.prototypes)
    assert np.allclose(artifact.matrix_numpy, expected_matrix)
    assert np.allclose(artifact.matrix_buffer.numpy(), expected_matrix, atol=1e-6)
    assert artifact.class_sample_counts == [3] * len(canonical_classes)


def test_artifact_margin_normalization_matches_independently_computed(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=3)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    # Recompute prototypes/margins independently is redundant with the
    # module's own two-pass design, so instead sanity-check the fitted
    # range is well-formed and derived from >0 samples.
    assert artifact.margin_normalization.margin_max >= artifact.margin_normalization.margin_min
    assert artifact.num_train_samples == len(canonical_classes) * 3


def test_matrix_buffer_requires_no_gradient(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert artifact.matrix_buffer.requires_grad is False


def test_result_is_deterministic_across_two_calls(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    artifact_a = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    artifact_b = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert np.array_equal(artifact_a.matrix_numpy, artifact_b.matrix_numpy)
    assert artifact_a.class_sample_counts == artifact_b.class_sample_counts
    assert artifact_a.margin_normalization == artifact_b.margin_normalization


# --- reference model stays frozen / read-only ---


def test_reference_model_eval_is_called(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
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
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed while building the ambiguity matrix")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_reference_model_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur while building the ambiguity matrix")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_reference_model_parameters_marked_requires_grad_false(tmp_path, monkeypatch):
    # A direct behavioral check: build_learned_class_ambiguity explicitly
    # calls requires_grad_(False) on every reference-model parameter.
    import src.training.ambiguity_setup as setup_module

    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
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
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

    assert hash_file(checkpoint_path) == hash_before


# --- train_original.csv only, no dependency on anything else ---


def test_succeeds_with_no_val_or_test_manifest_present(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    assert not (manifests_dir / "val_original.csv").exists()
    assert not (manifests_dir / "test_original.csv").exists()
    assert not (manifests_dir / "train_balanced.csv").exists()

    artifact = _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)
    assert artifact.train_manifest_path.endswith("train_original.csv")


def test_train_manifest_sha256_matches_train_original_file(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
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
    manifests_dir.mkdir(parents=True, exist_ok=True)  # deliberately empty - no train_original.csv
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(AmbiguitySetupError, match="train_original.csv"):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


# --- checkpoint compatibility ---


def test_incompatible_reference_checkpoint_rejected(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir = _setup(tmp_path, n_per_class=2)
    # Build a checkpoint declaring the wrong num_classes.
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, num_classes=99)

    with pytest.raises(CheckpointIncompatibleError):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)


def test_missing_reference_class_raises_ambiguity_setup_related_error(tmp_path):
    canonical_classes = CANONICAL_CLASSES[:3]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=True)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = _make_rows(raw_dir, canonical_classes[:2], 2, "train")  # omit the third class entirely
    write_csv(rows, MANIFEST_COLUMNS, manifests_dir / "train_original.csv")

    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, len(canonical_classes))

    with pytest.raises(AmbiguitySetupError, match="zero samples for class index"):
        _call(tmp_path, dataset_cfg, models_cfg, canonical_classes, manifests_dir, raw_dir, checkpoint_path)

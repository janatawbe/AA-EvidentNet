"""Integration tests for the learned class-ambiguity wiring in
src.training.run_aa_evidentnet.run_aa_evidentnet_training
(feature/learned-ambiguity, Phase 1).

Verifies: ambiguity_source="fixed_pairs" performs no setup and requires no
reference checkpoint (existing behavior unchanged); ambiguity_source=
"learned_class" builds the frozen matrix from train_original.csv BEFORE
trainer.fit() runs and installs it into CSSupConLoss; no test manifest is
ever read; failures (missing reference checkpoint config, missing
train_original.csv, smoke-test incompatibility) are explicit rather than
silently falling back to a different mode.

CRITICAL: no test here touches the real dataset, data/raw/, or any real
manifest - every fixture builds its own tiny, self-contained synthetic
dataset/checkpoint under tmp_path, and none of them ever create a
test_original.csv.
"""

import csv
import json

import pytest
import torch
import yaml

from src.losses.cs_supcon import CSSupConLoss
from src.models.factory import create_model
from src.training.ambiguity_setup import LearnedAmbiguityArtifact
from src.training.checkpointing import build_checkpoint, save_checkpoint
from src.training.run_aa_evidentnet import RunAAEvidentNetError, run_aa_evidentnet_training
from tests.conftest import make_image, write_min_dataset_config, write_min_losses_config, write_min_models_config, write_min_training_config

CANONICAL_CLASSES = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
]
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


def _write_manifest(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _build_reference_checkpoint(tmp_path, models_config_path, num_classes, run_id="reference_run"):
    models_config = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    checkpoint = build_checkpoint(
        model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), scheduler=None,
        epoch=5, best_metric=0.8, monitor_metric="val_macro_f1", training_config={}, seed=42,
        model_name="aa_evidentnet", architecture=model.architecture, num_classes=num_classes,
        dataset_manifest_hash="deadbeef", git_commit="abc123",
    )
    checkpoint_dir = tmp_path / "reference_checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


def _tmp_configs(tmp_path, n_per_class=2, losses_overrides=None, write_train_original=True):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in CANONICAL_CLASSES}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(tmp_path, num_classes=10, include_aa_evidentnet=True)
    training_config_path = write_min_training_config(
        tmp_path, overrides={"batch_size": 2, "epochs": 1, "early_stopping": {"enabled": False, "patience": 10}}
    )
    losses_config_path = write_min_losses_config(tmp_path, CANONICAL_CLASSES, overrides=losses_overrides)
    registry_path = tmp_path / "registry.csv"

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    if write_train_original:
        _write_manifest(
            manifests_dir / "train_original.csv", _make_rows(raw_dir, CANONICAL_CLASSES, "train", n_per_class, "trainorig")
        )
    _write_manifest(
        manifests_dir / "train_balanced.csv", _make_rows(raw_dir, CANONICAL_CLASSES, "train", n_per_class, "trainbal")
    )
    _write_manifest(manifests_dir / "val_original.csv", _make_rows(raw_dir, CANONICAL_CLASSES, "val", n_per_class, "val"))

    return dataset_config_path, models_config_path, training_config_path, losses_config_path, registry_path, manifests_dir


# --- fixed_pairs: existing behavior, no setup, no reference checkpoint required ---


def test_fixed_pairs_requires_no_reference_checkpoint_and_performs_no_setup(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, training_cfg, losses_cfg, registry_path, manifests_dir = _tmp_configs(tmp_path)

    import src.training.run_aa_evidentnet as raan

    calls = []
    monkeypatch.setattr(raan, "build_learned_class_ambiguity", lambda *a, **kw: calls.append(1))

    summary = run_aa_evidentnet_training(
        dataset_config_path=dataset_cfg, models_config_path=models_cfg, training_config_path=training_cfg,
        losses_config_path=losses_cfg, registry_path=registry_path, smoke_test=True,
    )

    assert summary.ambiguity_source == "fixed_pairs"
    assert summary.ambiguity_metadata_path is None
    assert calls == []  # build_learned_class_ambiguity never called


# --- learned_class: setup runs before training, matrix installed ---


def test_learned_class_builds_matrix_before_training_and_installs_it(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, training_cfg, losses_cfg_base, registry_path, manifests_dir = _tmp_configs(
        tmp_path, n_per_class=2
    )
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, num_classes=10)
    losses_cfg = write_min_losses_config(
        tmp_path,
        CANONICAL_CLASSES,
        overrides={"cs_supcon": {"ambiguity_source": "learned_class", "reference_checkpoint_path": str(checkpoint_path)}},
        config_name="losses_learned.yaml",
    )

    assert not (manifests_dir / "test_original.csv").exists()

    import src.training.run_aa_evidentnet as raan

    setup_calls = []
    original_build = raan.build_learned_class_ambiguity

    def spy_build(*args, **kwargs):
        artifact = original_build(*args, **kwargs)
        setup_calls.append(artifact)
        return artifact

    install_calls = []
    original_set_matrix = CSSupConLoss.set_learned_ambiguity_matrix

    def spy_set_matrix(self, matrix):
        install_calls.append(matrix.shape)
        return original_set_matrix(self, matrix)

    monkeypatch.setattr(raan, "build_learned_class_ambiguity", spy_build)
    monkeypatch.setattr(CSSupConLoss, "set_learned_ambiguity_matrix", spy_set_matrix)

    fit_calls = []
    original_fit = raan.Trainer.fit

    def spy_fit(self, *a, **kw):
        # By the time Trainer.fit() runs, ambiguity setup must already be done.
        fit_calls.append(len(setup_calls))
        return original_fit(self, *a, **kw)

    monkeypatch.setattr(raan.Trainer, "fit", spy_fit)

    summary = run_aa_evidentnet_training(
        dataset_config_path=dataset_cfg, models_config_path=models_cfg, training_config_path=training_cfg,
        losses_config_path=losses_cfg, registry_path=registry_path, smoke_test=False,
        max_train_steps_per_epoch=1, max_val_steps_per_epoch=1,
    )

    assert len(setup_calls) == 1  # built exactly once
    assert isinstance(setup_calls[0], LearnedAmbiguityArtifact)
    assert len(install_calls) == 1
    assert install_calls[0] == (10, 10)
    assert fit_calls == [1]  # setup had already run by the time Trainer.fit() started

    assert summary.ambiguity_source == "learned_class"
    assert summary.ambiguity_metadata_path is not None
    assert summary.ambiguity_metadata_path.is_file()

    with open(summary.ambiguity_metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)
    assert metadata["ambiguity_source"] == "learned_class"
    assert metadata["reference_checkpoint_path"] == str(checkpoint_path)
    assert len(metadata["class_ambiguity_matrix"]) == 10
    assert metadata["train_manifest_path"].endswith("train_original.csv")
    assert "methodological_caveat" in metadata
    assert not (manifests_dir / "test_original.csv").exists()  # never created by this run


# --- failures are explicit, never a silent fallback ---


def test_smoke_test_with_learned_class_raises_explicit_error(tmp_path):
    checkpoint_placeholder = tmp_path / "does_not_need_to_exist.pt"
    dataset_cfg, models_cfg, training_cfg, losses_cfg_base, registry_path, manifests_dir = _tmp_configs(tmp_path)
    losses_cfg = write_min_losses_config(
        tmp_path,
        CANONICAL_CLASSES,
        overrides={"cs_supcon": {"ambiguity_source": "learned_class", "reference_checkpoint_path": str(checkpoint_placeholder)}},
        config_name="losses_smoke_learned.yaml",
    )

    with pytest.raises(RunAAEvidentNetError, match="smoke_test"):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg, models_config_path=models_cfg, training_config_path=training_cfg,
            losses_config_path=losses_cfg, registry_path=registry_path, smoke_test=True,
        )


def test_learned_class_without_reference_checkpoint_path_fails_clearly(tmp_path):
    dataset_cfg, models_cfg, training_cfg, losses_cfg_base, registry_path, manifests_dir = _tmp_configs(tmp_path)
    losses_cfg = write_min_losses_config(
        tmp_path, CANONICAL_CLASSES, overrides={"cs_supcon": {"ambiguity_source": "learned_class"}},
        config_name="losses_missing_ref.yaml",
    )

    with pytest.raises(RunAAEvidentNetError, match="reference_checkpoint_path"):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg, models_config_path=models_cfg, training_config_path=training_cfg,
            losses_config_path=losses_cfg, registry_path=registry_path, smoke_test=True,
        )


def test_learned_class_missing_train_original_manifest_fails_clearly(tmp_path):
    dataset_cfg, models_cfg, training_cfg, losses_cfg_base, registry_path, manifests_dir = _tmp_configs(
        tmp_path, write_train_original=False
    )
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, num_classes=10)
    losses_cfg = write_min_losses_config(
        tmp_path,
        CANONICAL_CLASSES,
        overrides={"cs_supcon": {"ambiguity_source": "learned_class", "reference_checkpoint_path": str(checkpoint_path)}},
        config_name="losses_missing_manifest.yaml",
    )

    assert not (manifests_dir / "train_original.csv").exists()

    with pytest.raises(RunAAEvidentNetError, match="train_original.csv"):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg, models_config_path=models_cfg, training_config_path=training_cfg,
            losses_config_path=losses_cfg, registry_path=registry_path, smoke_test=False,
            max_train_steps_per_epoch=1, max_val_steps_per_epoch=1,
        )


def test_learned_class_incompatible_reference_checkpoint_fails_clearly(tmp_path):
    dataset_cfg, models_cfg, training_cfg, losses_cfg_base, registry_path, manifests_dir = _tmp_configs(tmp_path)
    # Reference checkpoint declares the wrong num_classes.
    checkpoint_path = _build_reference_checkpoint(tmp_path, models_cfg, num_classes=3)
    losses_cfg = write_min_losses_config(
        tmp_path,
        CANONICAL_CLASSES,
        overrides={"cs_supcon": {"ambiguity_source": "learned_class", "reference_checkpoint_path": str(checkpoint_path)}},
        config_name="losses_bad_ref.yaml",
    )

    with pytest.raises(RunAAEvidentNetError):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg, models_config_path=models_cfg, training_config_path=training_cfg,
            losses_config_path=losses_cfg, registry_path=registry_path, smoke_test=False,
            max_train_steps_per_epoch=1, max_val_steps_per_epoch=1,
        )

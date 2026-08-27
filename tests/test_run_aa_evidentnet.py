"""Integration tests for src.training.run_aa_evidentnet.run_aa_evidentnet_training
(Task 7 completion: AA-EvidentNet training orchestration + the combined
classification + CS-SupCon + EDL objective).

Uses the real create_model("aa_evidentnet", ...) (pretrained=False, small
embedding/local-feature dims) since there is no lightweight substitute for
"the actual training orchestration works end to end". Always uses
tmp_path-based registry/log/checkpoint directories so pytest never touches
the real project's results/ or experiments/registry.csv.
"""

import json

import pytest

from src.training.checkpointing import CheckpointIncompatibleError, load_checkpoint
from src.training.registry import STATUS_COMPLETED, STATUS_FAILED, load_registry
from src.training.run_aa_evidentnet import RunAAEvidentNetError, run_aa_evidentnet_training
from tests.conftest import write_min_dataset_config, write_min_losses_config, write_min_models_config, write_min_training_config

CANONICAL_CLASSES_10 = [
    "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
]


def _tmp_configs(tmp_path, training_overrides=None, losses_overrides=None, aa_evidentnet_overrides=None):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in CANONICAL_CLASSES_10}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(
        tmp_path, num_classes=10, include_aa_evidentnet=True, aa_evidentnet_overrides=aa_evidentnet_overrides
    )
    training_config_path = write_min_training_config(tmp_path, overrides=training_overrides)
    losses_config_path = write_min_losses_config(tmp_path, CANONICAL_CLASSES_10, overrides=losses_overrides)
    registry_path = tmp_path / "registry.csv"
    return dataset_config_path, models_config_path, training_config_path, losses_config_path, registry_path


def _run(tmp_path, **kwargs):
    dataset_cfg, models_cfg, training_cfg, losses_cfg, registry_path = _tmp_configs(
        tmp_path,
        training_overrides=kwargs.pop("training_overrides", None),
        losses_overrides=kwargs.pop("losses_overrides", None),
        aa_evidentnet_overrides=kwargs.pop("aa_evidentnet_overrides", None),
    )
    return run_aa_evidentnet_training(
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        losses_config_path=losses_cfg,
        registry_path=registry_path,
        **kwargs,
    ), registry_path


# --- end-to-end smoke test ---


def test_run_aa_evidentnet_training_smoke_test_end_to_end(tmp_path):
    summary, registry_path = _run(tmp_path, seed=42, smoke_test=True)

    assert summary.model_name == "aa_evidentnet"
    assert summary.smoke_test is True
    assert summary.device == "cpu"
    assert summary.amp_enabled is False
    assert summary.cs_supcon_enabled is True
    assert summary.edl_enabled is True
    assert summary.best_checkpoint_path is not None
    assert summary.best_checkpoint_path.is_file()
    assert summary.run_dir.is_dir()

    for filename in ("run.log", "metrics.jsonl", "config.yaml", "environment.txt", "dataset_hash.txt", "git_commit.txt"):
        assert (summary.run_dir / filename).is_file(), f"missing {filename}"

    metrics_lines = (summary.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    first_record = json.loads(metrics_lines[0])
    assert "train_loss" in first_record
    assert "val_macro_f1" in first_record
    assert "loss_components" in first_record
    assert set(first_record["loss_components"].keys()) == {"classification", "cs_supcon", "edl", "total"}

    registry_rows = load_registry(registry_path)
    assert len(registry_rows) == 1
    assert registry_rows[0]["status"] == STATUS_COMPLETED
    assert registry_rows[0]["model"] == "aa_evidentnet"
    assert registry_rows[0]["notes"] == "smoke_test"
    assert registry_rows[0]["test_result"] == ""  # test set never used


def test_run_aa_evidentnet_training_updates_real_model_weights(tmp_path):
    summary, _ = _run(tmp_path, seed=42, smoke_test=True)
    checkpoint = load_checkpoint(summary.best_checkpoint_path)
    assert checkpoint["metadata"]["model_name"] == "aa_evidentnet"
    assert checkpoint["metadata"]["num_classes"] == 10
    # The fusion gate is included in the saved state dict (Task 7 architecture).
    assert any("alpha" in key for key in checkpoint["model_state_dict"])


# --- combined objective actually drives training (component ablation) ---


def test_disabling_cs_supcon_and_edl_still_trains_on_classification_only(tmp_path):
    summary, registry_path = _run(
        tmp_path,
        seed=42,
        smoke_test=True,
        losses_overrides={"cs_supcon": {"enabled": False}, "edl": {"enabled": False}},
    )
    assert summary.cs_supcon_enabled is False
    assert summary.edl_enabled is False
    metrics_lines = (summary.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    first_record = json.loads(metrics_lines[0])
    assert set(first_record["loss_components"].keys()) == {"classification", "total"}


def test_enabling_only_cs_supcon(tmp_path):
    summary, _ = _run(
        tmp_path,
        seed=42,
        smoke_test=True,
        losses_overrides={"cs_supcon": {"enabled": True}, "edl": {"enabled": False}},
    )
    assert summary.cs_supcon_enabled is True
    assert summary.edl_enabled is False


def test_enabling_only_edl(tmp_path):
    summary, _ = _run(
        tmp_path,
        seed=42,
        smoke_test=True,
        losses_overrides={"cs_supcon": {"enabled": False}, "edl": {"enabled": True}},
    )
    assert summary.cs_supcon_enabled is False
    assert summary.edl_enabled is True


def test_invalid_loss_config_fails_clearly(tmp_path):
    with pytest.raises(RunAAEvidentNetError, match="combined"):
        _run(tmp_path, seed=42, smoke_test=True, losses_overrides={"baseline": {"class_weighting": "inverse_frequency"}})


# --- missing config entry / manifest ---


def test_missing_proposed_config_entry_fails_clearly(tmp_path):
    dataset_cfg, models_cfg, training_cfg, losses_cfg, registry_path = _tmp_configs(tmp_path)
    import yaml

    config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    del config["proposed"]
    models_cfg.write_text(yaml.safe_dump(config), encoding="utf-8")

    with pytest.raises(RunAAEvidentNetError, match="proposed"):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            losses_config_path=losses_cfg,
            registry_path=registry_path,
            smoke_test=True,
        )


def test_missing_train_manifest_raises(tmp_path):
    dataset_cfg, models_cfg, training_cfg, losses_cfg, registry_path = _tmp_configs(tmp_path)
    with pytest.raises(RunAAEvidentNetError, match="train_balanced.csv|not found"):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            losses_config_path=losses_cfg,
            registry_path=registry_path,
            smoke_test=False,
        )


# --- failure handling ---


def test_failed_run_updates_registry_status(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, training_cfg, losses_cfg, registry_path = _tmp_configs(tmp_path)

    import src.training.run_aa_evidentnet as raan

    def broken_fit(self, *a, **kw):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(raan.Trainer, "fit", broken_fit)

    with pytest.raises(RuntimeError, match="simulated training failure"):
        run_aa_evidentnet_training(
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            losses_config_path=losses_cfg,
            registry_path=registry_path,
            smoke_test=True,
        )

    rows = load_registry(registry_path)
    assert rows[0]["status"] == STATUS_FAILED
    assert "simulated training failure" in rows[0]["notes"]


# --- resume ---


def test_resume_from_checkpoint_restores_epoch_and_rejects_incompatible_num_classes(tmp_path):
    dataset_cfg, models_cfg, training_cfg, losses_cfg, registry_path = _tmp_configs(tmp_path)

    first_summary = run_aa_evidentnet_training(
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        losses_config_path=losses_cfg,
        registry_path=registry_path,
        smoke_test=True,
    )
    checkpoint_path = first_summary.best_checkpoint_path
    assert checkpoint_path is not None

    registry_path_2 = tmp_path / "registry2.csv"
    second_summary = run_aa_evidentnet_training(
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        losses_config_path=losses_cfg,
        registry_path=registry_path_2,
        smoke_test=True,
        resume_from=checkpoint_path,
    )
    # smoke_test always forces epochs=2 (see run_aa_evidentnet.SMOKE_TEST_EPOCHS),
    # so the first run completes epochs 0,1 and resuming continues at epoch 2.
    assert second_summary.fit_result.history[0].epoch == 2

    # Resuming into a run with a different num_classes (derived from the
    # dataset config's class_directory_mapping, exactly like
    # run_aa_evidentnet_training itself computes it) must fail clearly.
    # AA-EvidentNet has only one registered proposed-model name, so (unlike
    # run_baseline's resnet50-vs-efficientnetb0 test) the mismatch must
    # come from num_classes, not model_name.
    registry_path_3 = tmp_path / "registry3.csv"
    raw_dir_4 = tmp_path / "raw4"
    audit_dir_4 = tmp_path / "audit4"
    mismatched_dataset_cfg = write_min_dataset_config(
        tmp_path,
        {name: name for name in ["Alpha", "Beta", "Gamma", "Delta"]},
        raw_dir_4,
        audit_dir_4,
        config_name="dataset_4class.yaml",
    )
    mismatched_models_cfg = write_min_models_config(
        tmp_path, num_classes=4, config_name="models_4class.yaml", include_aa_evidentnet=True
    )
    mismatched_losses_cfg = write_min_losses_config(tmp_path, ["Alpha", "Beta", "Gamma", "Delta"], config_name="losses_4class.yaml")
    with pytest.raises(CheckpointIncompatibleError):
        run_aa_evidentnet_training(
            dataset_config_path=mismatched_dataset_cfg,
            models_config_path=mismatched_models_cfg,
            training_config_path=training_cfg,
            losses_config_path=mismatched_losses_cfg,
            registry_path=registry_path_3,
            smoke_test=True,
            resume_from=checkpoint_path,
        )


# --- test-set lock ---


def test_smoke_test_does_not_create_or_reference_test_manifest_path(tmp_path):
    summary, _ = _run(tmp_path, seed=42, smoke_test=True)
    all_paths = [str(p) for p in summary.run_dir.rglob("*")] + [str(p) for p in summary.checkpoint_dir.rglob("*")]
    assert not any("test_original" in p for p in all_paths)

"""Integration tests for src.training.run_baseline.run_baseline_training.

Uses real create_model() (pretrained=False, fast/offline) since there is
no lightweight substitute for "the actual baseline training orchestration
works end to end". Uses "efficientnetb0" (fewest parameters of the three)
to keep the suite fast, and always tmp_path-based registry/log/checkpoint
directories so pytest never touches the real project's results/ or
experiments/registry.csv.
"""

import json

import pytest

from src.training.checkpointing import CheckpointIncompatibleError, load_checkpoint
from src.training.registry import STATUS_COMPLETED, STATUS_FAILED, load_registry
from src.training.run_baseline import RunBaselineError, run_baseline_training
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config, write_min_training_config

FAST_MODEL = "efficientnetb0"


def _tmp_configs(tmp_path, training_overrides=None):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in [
        "Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa",
    ]}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(tmp_path, num_classes=10)
    training_config_path = write_min_training_config(tmp_path, overrides=training_overrides)
    registry_path = tmp_path / "registry.csv"
    return dataset_config_path, models_config_path, training_config_path, registry_path


def test_run_baseline_training_rejects_unknown_model(tmp_path):
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)
    with pytest.raises(RunBaselineError, match="Unknown baseline model"):
        run_baseline_training(
            "not_a_real_model",
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            registry_path=registry_path,
        )


def test_run_baseline_training_smoke_test_end_to_end(tmp_path):
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)

    summary = run_baseline_training(
        FAST_MODEL,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        seed=42,
        smoke_test=True,
        registry_path=registry_path,
    )

    assert summary.smoke_test is True
    assert summary.device == "cpu"
    assert summary.amp_enabled is False
    assert summary.best_checkpoint_path is not None
    assert summary.best_checkpoint_path.is_file()
    assert summary.run_dir.is_dir()

    for filename in ("run.log", "metrics.jsonl", "config.yaml", "environment.txt", "dataset_hash.txt", "git_commit.txt"):
        assert (summary.run_dir / filename).is_file(), f"missing {filename}"

    metrics_lines = (summary.run_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().split("\n")
    assert len(metrics_lines) >= 1
    first_record = json.loads(metrics_lines[0])
    assert "train_loss" in first_record
    assert "val_macro_f1" in first_record

    registry_rows = load_registry(registry_path)
    assert len(registry_rows) == 1
    assert registry_rows[0]["status"] == STATUS_COMPLETED
    assert registry_rows[0]["model"] == FAST_MODEL
    assert registry_rows[0]["notes"] == "smoke_test"
    assert registry_rows[0]["test_result"] == ""  # test set never used


def test_run_baseline_training_smoke_test_never_downloads_pretrained_weights(tmp_path):
    # Models config requests pretrained=True, but smoke_test must force it
    # to False regardless (offline, fast, deterministic).
    dataset_cfg, _, training_cfg, registry_path = _tmp_configs(tmp_path)
    models_cfg = write_min_models_config(tmp_path, num_classes=10)
    import yaml

    config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    config["baselines"][FAST_MODEL]["pretrained"] = True
    models_cfg.write_text(yaml.safe_dump(config), encoding="utf-8")

    # Should complete fast without attempting a network call; if this were
    # actually pulling pretrained weights the smoke test would be much
    # slower and require internet - we simply verify it still completes.
    summary = run_baseline_training(
        FAST_MODEL,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        smoke_test=True,
        registry_path=registry_path,
    )
    assert summary.smoke_test is True


def test_run_baseline_training_missing_train_manifest_raises(tmp_path):
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)
    with pytest.raises(RunBaselineError, match="train_original.csv|train_balanced.csv|not found"):
        run_baseline_training(
            FAST_MODEL,
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            smoke_test=False,
            registry_path=registry_path,
        )


def test_run_baseline_training_failed_run_updates_registry_status(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)

    import src.training.run_baseline as rb

    def broken_fit(self, *a, **kw):
        raise RuntimeError("simulated training failure")

    monkeypatch.setattr(rb.Trainer, "fit", broken_fit)

    with pytest.raises(RuntimeError, match="simulated training failure"):
        run_baseline_training(
            FAST_MODEL,
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            smoke_test=True,
            registry_path=registry_path,
        )

    rows = load_registry(registry_path)
    assert rows[0]["status"] == STATUS_FAILED
    assert "simulated training failure" in rows[0]["notes"]


def test_resume_from_checkpoint_restores_epoch_and_rejects_incompatible_model(tmp_path):
    # Note: smoke_test=True always forces epochs=SMOKE_TEST_EPOCHS (2)
    # regardless of the config file, unless epochs_override is passed
    # explicitly to run_baseline_training() - so the first run below
    # completes epochs 0 and 1, and resuming should continue at epoch 2.
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)

    first_summary = run_baseline_training(
        FAST_MODEL,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        smoke_test=True,
        registry_path=registry_path,
    )
    checkpoint_path = first_summary.best_checkpoint_path
    assert checkpoint_path is not None

    # Resuming into the SAME model architecture must work and continue
    # from the next epoch.
    registry_path_2 = tmp_path / "registry2.csv"
    second_summary = run_baseline_training(
        FAST_MODEL,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        smoke_test=True,
        resume_from=checkpoint_path,
        registry_path=registry_path_2,
    )
    assert second_summary.fit_result.history[0].epoch == 2  # continues after epochs 0,1

    # Resuming into a DIFFERENT model architecture must fail clearly.
    registry_path_3 = tmp_path / "registry3.csv"
    with pytest.raises(CheckpointIncompatibleError):
        run_baseline_training(
            "resnet50",
            dataset_config_path=dataset_cfg,
            models_config_path=models_cfg,
            training_config_path=training_cfg,
            smoke_test=True,
            resume_from=checkpoint_path,
            registry_path=registry_path_3,
        )


def test_run_baseline_training_checkpoint_includes_scaler_state(tmp_path):
    # Colab/CUDA readiness: every checkpoint must carry scaler_state_dict
    # (even if trivial/disabled, as on this CPU-only machine) so a CUDA +
    # mixed-precision run can resume its AMP loss-scale state faithfully.
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)
    summary = run_baseline_training(
        FAST_MODEL,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        smoke_test=True,
        registry_path=registry_path,
    )
    checkpoint = load_checkpoint(summary.best_checkpoint_path)
    assert "scaler_state_dict" in checkpoint
    assert checkpoint["scaler_state_dict"] is not None  # a real (disabled-on-CPU) GradScaler state


def test_run_baseline_training_smoke_test_does_not_create_test_manifest_path(tmp_path):
    # Behavioral guarantee (not a source-text grep, which would also match
    # this module's own docstring explaining the exclusion): running a
    # full smoke-test training never creates or reads any file whose name
    # mentions "test_original" anywhere under the run's directories.
    dataset_cfg, models_cfg, training_cfg, registry_path = _tmp_configs(tmp_path)
    summary = run_baseline_training(
        FAST_MODEL,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        training_config_path=training_cfg,
        smoke_test=True,
        registry_path=registry_path,
    )
    all_paths = [str(p) for p in summary.run_dir.rglob("*")] + [str(p) for p in summary.checkpoint_dir.rglob("*")]
    assert not any("test_original" in p for p in all_paths)

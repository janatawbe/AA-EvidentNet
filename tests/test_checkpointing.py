"""Tests for src.training.checkpointing. Uses a tiny nn.Linear "model" -
no real backbone needed to test save/load/resume/compatibility mechanics."""

import pytest
import torch
import torch.nn as nn

from src.training.checkpointing import (
    CheckpointIncompatibleError,
    assert_checkpoint_compatible,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)


def _tiny_model():
    return nn.Linear(4, 10)


def test_build_checkpoint_contains_required_fields():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    checkpoint = build_checkpoint(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        epoch=3,
        best_metric=0.75,
        monitor_metric="val_macro_f1",
        training_config={"lr": 1e-3},
        seed=42,
        model_name="resnet50",
        architecture="resnet50",
        num_classes=10,
        dataset_manifest_hash="deadbeef",
        git_commit="abc123",
    )

    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint
    assert "scheduler_state_dict" in checkpoint
    metadata = checkpoint["metadata"]
    assert metadata["epoch"] == 3
    assert metadata["best_metric"] == 0.75
    assert metadata["model_name"] == "resnet50"
    assert metadata["num_classes"] == 10
    assert metadata["seed"] == 42
    assert metadata["dataset_manifest_hash"] == "deadbeef"
    assert metadata["git_commit"] == "abc123"
    assert "timestamp_utc" in metadata


def test_save_and_load_checkpoint_roundtrip(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c",
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, path)
    assert path.is_file()

    loaded = load_checkpoint(path)
    assert loaded["metadata"]["epoch"] == 1
    assert set(loaded["model_state_dict"].keys()) == set(model.state_dict().keys())


def test_load_checkpoint_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_checkpoint(tmp_path / "does_not_exist.pt")


def test_assert_checkpoint_compatible_passes_for_matching():
    checkpoint = {"metadata": {"model_name": "resnet50", "num_classes": 10}}
    assert_checkpoint_compatible(checkpoint, "resnet50", 10)  # must not raise


def test_assert_checkpoint_compatible_rejects_wrong_model_name():
    checkpoint = {"metadata": {"model_name": "resnet50", "num_classes": 10}}
    with pytest.raises(CheckpointIncompatibleError, match="model_name"):
        assert_checkpoint_compatible(checkpoint, "efficientnetb0", 10)


def test_assert_checkpoint_compatible_rejects_wrong_num_classes():
    checkpoint = {"metadata": {"model_name": "resnet50", "num_classes": 10}}
    with pytest.raises(CheckpointIncompatibleError, match="num_classes"):
        assert_checkpoint_compatible(checkpoint, "resnet50", 4)


def test_restore_training_state_restores_model_and_optimizer(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Mutate the model's weights so we can detect restoration.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(1.0)
    mutated_state = {k: v.clone() for k, v in model.state_dict().items()}

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=5, best_metric=0.9, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c",
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, path)

    fresh_model = _tiny_model()  # different random init
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    fresh_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fresh_optimizer)

    loaded = load_checkpoint(path)
    state = restore_training_state(loaded, fresh_model, fresh_optimizer, fresh_scheduler)

    assert state["epoch"] == 5
    assert state["best_metric"] == 0.9
    for key, value in fresh_model.state_dict().items():
        assert torch.equal(value, mutated_state[key])

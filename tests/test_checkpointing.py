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


# --- AMP GradScaler state (Colab/CUDA-readiness: matters only for CUDA +
# mixed_precision runs - on CPU the scaler is always disabled and this is
# an inert no-op, but the checkpoint schema must still round-trip it
# correctly so a CUDA resume doesn't lose the adaptive loss-scale state). ---


def test_build_checkpoint_omits_scaler_state_when_no_scaler_given():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c",
    )
    assert "scaler_state_dict" in checkpoint
    assert checkpoint["scaler_state_dict"] is None


def test_build_checkpoint_includes_scaler_state_when_scaler_given():
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = torch.amp.GradScaler(device="cuda", enabled=False)  # enabled=False: safe on CPU-only machines

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c", scaler=scaler,
    )
    assert checkpoint["scaler_state_dict"] is not None
    assert checkpoint["scaler_state_dict"] == scaler.state_dict()


def test_restore_training_state_restores_scaler_when_present(tmp_path):
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = torch.amp.GradScaler(device="cuda", enabled=False)

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c", scaler=scaler,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, path)

    fresh_scaler = torch.amp.GradScaler(device="cuda", enabled=False)
    loaded = load_checkpoint(path)
    # Must not raise, and must actually apply the saved scaler state.
    restore_training_state(loaded, model, optimizer, scheduler, scaler=fresh_scaler)
    assert fresh_scaler.state_dict() == scaler.state_dict()


def test_restore_training_state_without_scaler_arg_is_backward_compatible(tmp_path):
    # A checkpoint written WITH scaler state must still restore correctly
    # when the caller doesn't pass a scaler at all (old call sites).
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)
    scaler = torch.amp.GradScaler(device="cuda", enabled=False)

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=2, best_metric=0.6, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c", scaler=scaler,
    )
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(checkpoint, path)

    loaded = load_checkpoint(path)
    state = restore_training_state(loaded, model, optimizer, scheduler)  # no scaler= at all
    assert state["epoch"] == 2


def test_restore_training_state_old_checkpoint_without_scaler_key_still_works(tmp_path):
    # Simulates a checkpoint written before this parameter existed: no
    # "scaler_state_dict" key at all in the saved dict.
    model = _tiny_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    checkpoint = build_checkpoint(
        model, optimizer, scheduler, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture="resnet50",
        num_classes=10, dataset_manifest_hash="h", git_commit="c",
    )
    del checkpoint["scaler_state_dict"]  # simulate a genuinely pre-existing old checkpoint
    path = tmp_path / "old_checkpoint.pt"
    save_checkpoint(checkpoint, path)

    loaded = load_checkpoint(path)
    fresh_scaler = torch.amp.GradScaler(device="cuda", enabled=False)
    state = restore_training_state(loaded, model, optimizer, scheduler, scaler=fresh_scaler)
    assert state["epoch"] == 1

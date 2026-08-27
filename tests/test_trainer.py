"""Tests for src.training.trainer.Trainer: the reusable train/validate
engine. Uses a tiny synthetic nn.Module + in-memory dict-dataset - no real
backbone or real dataset needed, since Trainer only requires
`model(images) -> logits`."""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from src.training.trainer import (
    EpochResult,
    Trainer,
    TrainerError,
    TrainingConfig,
    _is_better,
    _monitor_value,
    build_optimizer,
    build_scheduler,
    resolve_device,
)


class _TinyClassifier(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(3 * 8 * 8, num_classes)

    def forward(self, images):
        return self.linear(self.flatten(images))


class _DictDataset(Dataset):
    """Mimics RetinalDataset's dict-sample shape without touching disk."""

    def __init__(self, n, num_classes=10, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 8, 8, generator=gen)
        self.labels = torch.randint(0, num_classes, (n,), generator=gen)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": int(self.labels[idx])}


def _make_trainer(train_n=16, val_n=8, batch_size=4, config_overrides=None, on_epoch_end=None):
    train_loader = DataLoader(_DictDataset(train_n, seed=1), batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(_DictDataset(val_n, seed=2), batch_size=batch_size, shuffle=False)
    model = _TinyClassifier()
    config = TrainingConfig(**{**dict(epochs=2, batch_size=batch_size, mixed_precision=False), **(config_overrides or {})})
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=torch.device("cpu"),
        on_epoch_end=on_epoch_end,
    )
    return trainer


# --- device resolution ---


def test_resolve_device_cpu():
    assert resolve_device("cpu").type == "cpu"


def test_resolve_device_auto_falls_back_to_cpu_without_cuda():
    device = resolve_device("auto")
    assert device.type in ("cpu", "cuda")  # cuda only if actually available
    if not torch.cuda.is_available():
        assert device.type == "cpu"


def test_resolve_device_cuda_raises_clearly_if_unavailable():
    if torch.cuda.is_available():
        pytest.skip("CUDA is available in this environment; nothing to test here")
    with pytest.raises(TrainerError, match="CUDA"):
        resolve_device("cuda")


def test_resolve_device_unknown_raises():
    with pytest.raises(TrainerError, match="Unknown device"):
        resolve_device("tpu")


# --- config parsing ---


def test_training_config_from_dict_uses_defaults_for_missing_keys():
    config = TrainingConfig.from_dict({})
    assert config.seed == 42
    assert config.optimizer_name == "adamw"
    assert config.scheduler_name == "reduce_on_plateau"
    assert config.monitor_metric == "val_macro_f1"
    assert config.mode == "max"


def test_training_config_from_dict_honors_provided_values():
    config = TrainingConfig.from_dict({"epochs": 5, "optimizer": {"lr": 0.01}, "gradient_clip_norm": None})
    assert config.epochs == 5
    assert config.lr == 0.01
    assert config.gradient_clip_norm is None


# --- optimizer / scheduler construction ---


def test_build_optimizer_rejects_unsupported_name():
    model = _TinyClassifier()
    config = TrainingConfig(optimizer_name="sgd")
    with pytest.raises(TrainerError, match="Unsupported optimizer"):
        build_optimizer(model, config)


def test_build_scheduler_rejects_unsupported_name():
    model = _TinyClassifier()
    config = TrainingConfig(scheduler_name="cosine")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(TrainerError, match="Unsupported scheduler"):
        build_scheduler(optimizer, config)


# --- monitor value / is_better helpers ---


def test_monitor_value_strips_val_prefix():
    assert _monitor_value({"macro_f1": 0.7}, "val_macro_f1") == 0.7


def test_monitor_value_missing_key_raises():
    with pytest.raises(TrainerError, match="not found"):
        _monitor_value({"loss": 1.0}, "val_macro_f1")


def test_is_better_max_mode():
    assert _is_better(0.6, 0.5, "max") is True
    assert _is_better(0.4, 0.5, "max") is False


def test_is_better_min_mode():
    assert _is_better(0.4, 0.5, "min") is True
    assert _is_better(0.6, 0.5, "min") is False


def test_is_better_always_true_when_no_prior_best():
    assert _is_better(0.0, None, "max") is True


# --- trainer initialization ---


def test_trainer_initializes_and_resolves_amp():
    trainer = _make_trainer()
    assert trainer.device.type == "cpu"
    assert trainer.amp_enabled is False  # never true on CPU regardless of config


def test_trainer_amp_enabled_requested_but_cpu_stays_disabled():
    trainer = _make_trainer(config_overrides={"mixed_precision": True})
    assert trainer.amp_enabled is False


# --- one training / validation epoch ---


def test_train_one_epoch_returns_expected_metric_keys():
    trainer = _make_trainer()
    metrics = trainer.train_one_epoch()
    assert set(metrics.keys()) == {"loss", "accuracy", "macro_f1"}


def test_validate_one_epoch_returns_expected_metric_keys_including_balanced_accuracy():
    trainer = _make_trainer()
    metrics = trainer.validate_one_epoch()
    assert set(metrics.keys()) == {"loss", "accuracy", "macro_f1", "balanced_accuracy"}


def test_train_one_epoch_updates_model_weights():
    trainer = _make_trainer()
    before = copy.deepcopy(trainer.model.state_dict())
    trainer.train_one_epoch()
    after = trainer.model.state_dict()
    changed = any(not torch.equal(before[k], after[k]) for k in before)
    assert changed, "model weights should change after a real forward/backward/optimizer-step epoch"


def test_validate_one_epoch_does_not_update_model_weights():
    trainer = _make_trainer()
    before = copy.deepcopy(trainer.model.state_dict())
    trainer.validate_one_epoch()
    after = trainer.model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)


def test_max_steps_per_epoch_limits_batches_processed():
    trainer = _make_trainer(train_n=32, batch_size=4)  # 8 batches available
    calls = []
    original_forward_loss = trainer._forward_loss

    def spy(batch):
        calls.append(1)
        return original_forward_loss(batch)

    trainer._forward_loss = spy
    trainer.train_one_epoch(max_steps=3)
    assert len(calls) == 3


# --- gradient accumulation ---


def test_gradient_accumulation_steps_optimizer_at_correct_intervals():
    trainer = _make_trainer(train_n=16, batch_size=2, config_overrides={"gradient_accumulation_steps": 3})
    step_calls = []
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step
    trainer.train_one_epoch()
    # 16 samples / batch_size 2 = 8 batches; accum_steps=3 -> steps at
    # batches 3, 6, and the final (8th, partial-accumulation) batch = 3 steps.
    assert len(step_calls) == 3


def test_gradient_accumulation_of_one_steps_every_batch():
    trainer = _make_trainer(train_n=8, batch_size=2, config_overrides={"gradient_accumulation_steps": 1})
    step_calls = []
    original_step = trainer.optimizer.step

    def counting_step(*args, **kwargs):
        step_calls.append(1)
        return original_step(*args, **kwargs)

    trainer.optimizer.step = counting_step
    trainer.train_one_epoch()
    assert len(step_calls) == 4  # 8 samples / batch_size 2 = 4 batches, all stepped


# --- gradient clipping ---


def test_gradient_clip_norm_caps_gradient_norm():
    trainer = _make_trainer(config_overrides={"gradient_clip_norm": 0.01, "lr": 10.0})
    # Force huge gradients by scaling up loss artificially via a big lr is
    # not directly observable on grad norm; instead check clip_grad_norm_
    # was actually invoked by monkeypatching it.
    calls = []
    import torch.nn.utils as utils

    original_clip = utils.clip_grad_norm_

    def spy_clip(parameters, max_norm, *a, **kw):
        calls.append(max_norm)
        return original_clip(parameters, max_norm, *a, **kw)

    utils.clip_grad_norm_ = spy_clip
    try:
        trainer.train_one_epoch()
    finally:
        utils.clip_grad_norm_ = original_clip
    assert calls == [0.01] * len(calls)
    assert len(calls) > 0


def test_gradient_clip_disabled_never_calls_clip():
    trainer = _make_trainer(config_overrides={"gradient_clip_norm": None})
    calls = []
    import torch.nn.utils as utils

    original_clip = utils.clip_grad_norm_

    def spy_clip(*a, **kw):
        calls.append(1)
        return original_clip(*a, **kw)

    utils.clip_grad_norm_ = spy_clip
    try:
        trainer.train_one_epoch()
    finally:
        utils.clip_grad_norm_ = original_clip
    assert calls == []


# --- scheduler stepping ---


def test_scheduler_last_epoch_advances_after_fit():
    trainer = _make_trainer(config_overrides={"epochs": 3, "early_stopping_enabled": False})
    result = trainer.fit()
    assert len(result.history) == 3
    assert trainer.scheduler.last_epoch == 3


# --- early stopping ---


def test_early_stopping_triggers_after_patience_epochs_without_improvement():
    trainer = _make_trainer(config_overrides={"epochs": 20, "early_stopping_enabled": True, "early_stopping_patience": 2})
    # Force validation metrics to never improve after epoch 0.
    call_count = {"n": 0}

    def fake_validate(max_steps=None):
        call_count["n"] += 1
        value = 0.9 if call_count["n"] == 1 else 0.1
        return {"loss": 1.0, "accuracy": value, "macro_f1": value, "balanced_accuracy": value}

    trainer.validate_one_epoch = fake_validate
    result = trainer.fit()

    assert result.best_epoch == 0
    assert result.stopped_epoch == 2  # stops patience(=2) epochs after best (epoch 0)
    assert "early stopping" in result.stopping_reason


def test_no_early_stopping_runs_full_epoch_budget_when_disabled():
    trainer = _make_trainer(config_overrides={"epochs": 3, "early_stopping_enabled": False})
    result = trainer.fit()
    assert len(result.history) == 3
    assert result.stopping_reason == "completed all configured epochs"


def test_fit_calls_on_epoch_end_with_epoch_result():
    received = []
    trainer = _make_trainer(config_overrides={"epochs": 2}, on_epoch_end=lambda r: received.append(r))
    trainer.fit()
    assert len(received) == 2
    assert all(isinstance(r, EpochResult) for r in received)
    assert received[0].epoch == 0
    assert received[1].epoch == 1


def test_fit_result_tracks_best_metric_and_is_best_flag():
    trainer = _make_trainer(config_overrides={"epochs": 2})
    result = trainer.fit()
    assert result.best_metric is not None
    assert result.history[0].is_best is True  # first epoch always "best" (no prior)

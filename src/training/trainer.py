"""The reusable training engine every model (baselines now, AA-EvidentNet
later) trains through. Nothing here is architecture-specific - it only
requires a `model(images) -> logits` callable (exactly what
src.models.base.TimmBackboneModel provides).

Device/AMP handling: mixed precision is only ever actually enabled when
running on CUDA. Requesting mixed_precision=True on a CPU device does not
error - it is silently (but loggably, via Trainer.amp_enabled) downgraded
to disabled, since CUDA AMP has no CPU equivalent here and must never be
falsely reported as active.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.training.metrics import MetricAccumulator

DEFAULT_BETAS = (0.9, 0.999)


class TrainerError(Exception):
    """Raised for a misconfigured trainer (e.g. an unsupported optimizer/
    scheduler name) - never silently falls back to a different one."""


@dataclass
class TrainingConfig:
    seed: int = 42
    batch_size: int = 16
    epochs: int = 50
    num_workers: int = 4

    optimizer_name: str = "adamw"
    lr: float = 3.0e-4
    weight_decay: float = 1.0e-4
    betas: tuple = DEFAULT_BETAS
    eps: float = 1.0e-8

    scheduler_name: str = "reduce_on_plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 5
    scheduler_min_lr: float = 1.0e-6

    early_stopping_enabled: bool = True
    early_stopping_patience: int = 10

    monitor_metric: str = "val_macro_f1"
    mode: str = "max"

    gradient_clip_norm: Optional[float] = 1.0
    gradient_accumulation_steps: int = 1

    mixed_precision: bool = True
    checkpoint_frequency: int = 1

    @classmethod
    def from_dict(cls, config: Dict[str, Any]) -> "TrainingConfig":
        optimizer_cfg = config.get("optimizer", {}) or {}
        scheduler_cfg = config.get("scheduler", {}) or {}
        early_stopping_cfg = config.get("early_stopping", {}) or {}

        return cls(
            seed=config.get("seed", 42),
            batch_size=config.get("batch_size", 16),
            epochs=config.get("epochs", 50),
            num_workers=config.get("num_workers", 4),
            optimizer_name=optimizer_cfg.get("name", "adamw"),
            lr=float(optimizer_cfg.get("lr", 3.0e-4)),
            weight_decay=float(optimizer_cfg.get("weight_decay", 1.0e-4)),
            betas=tuple(optimizer_cfg.get("betas", DEFAULT_BETAS)),
            eps=float(optimizer_cfg.get("eps", 1.0e-8)),
            scheduler_name=scheduler_cfg.get("name", "reduce_on_plateau"),
            scheduler_factor=float(scheduler_cfg.get("factor", 0.5)),
            scheduler_patience=int(scheduler_cfg.get("patience", 5)),
            scheduler_min_lr=float(scheduler_cfg.get("min_lr", 1.0e-6)),
            early_stopping_enabled=bool(early_stopping_cfg.get("enabled", True)),
            early_stopping_patience=int(early_stopping_cfg.get("patience", 10)),
            monitor_metric=config.get("monitor_metric", "val_macro_f1"),
            mode=config.get("mode", "max"),
            gradient_clip_norm=config.get("gradient_clip_norm", 1.0),
            gradient_accumulation_steps=int(config.get("gradient_accumulation_steps", 1)),
            mixed_precision=bool(config.get("mixed_precision", True)),
            checkpoint_frequency=int(config.get("checkpoint_frequency", 1)),
        )


def resolve_device(requested: str = "auto") -> torch.device:
    """'cpu' -> cpu. 'cuda' -> cuda, or raise TrainerError if unavailable
    (an explicit request deserves a clear failure, not a silent fallback).
    'auto' -> cuda if available, else cpu, with no error either way."""
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise TrainerError("--device cuda was requested but CUDA is not available on this machine")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    raise TrainerError(f"Unknown device '{requested}' (must be 'auto', 'cpu', or 'cuda')")


def build_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer_name.lower() != "adamw":
        raise TrainerError(f"Unsupported optimizer '{config.optimizer_name}' (only 'adamw' is implemented)")
    return torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay, betas=config.betas, eps=config.eps
    )


def build_scheduler(optimizer: torch.optim.Optimizer, config: TrainingConfig):
    if config.scheduler_name.lower() != "reduce_on_plateau":
        raise TrainerError(f"Unsupported scheduler '{config.scheduler_name}' (only 'reduce_on_plateau' is implemented)")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode=config.mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        min_lr=config.scheduler_min_lr,
    )


def _monitor_value(val_metrics: Dict[str, float], monitor_metric: str) -> float:
    """monitor_metric is written as 'val_<key>' by convention (e.g.
    'val_macro_f1'); only validation metrics are ever used for early
    stopping / scheduler / checkpoint selection - never train or test."""
    key = monitor_metric[len("val_") :] if monitor_metric.startswith("val_") else monitor_metric
    if key not in val_metrics:
        raise TrainerError(f"monitor_metric '{monitor_metric}' not found among validation metrics {list(val_metrics.keys())}")
    return val_metrics[key]


def _is_better(current: float, best: Optional[float], mode: str) -> bool:
    if best is None:
        return True
    return current > best if mode == "max" else current < best


@dataclass
class EpochResult:
    epoch: int
    train_metrics: Dict[str, float]
    val_metrics: Dict[str, float]
    lr: float
    elapsed_seconds: float
    is_best: bool = False
    monitor_value: float = 0.0


@dataclass
class FitResult:
    history: List[EpochResult] = field(default_factory=list)
    best_epoch: int = 0
    best_metric: Optional[float] = None
    stopped_epoch: int = 0
    stopping_reason: str = "completed all configured epochs"
    amp_enabled: bool = False
    device: str = "cpu"


class Trainer:
    """Reusable train/validate engine. Owns no dataset/model-specific
    logic - the caller (e.g. src/training/run_baseline.py) is responsible
    for constructing the model/dataloaders/optimizer/scheduler and for
    checkpointing/logging; Trainer.fit() calls back into caller-supplied
    hooks for those side effects so it stays reusable across models."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainingConfig,
        device: Optional[torch.device] = None,
        criterion: Optional[nn.Module] = None,
        on_epoch_end: Optional[Callable[[EpochResult], None]] = None,
        start_epoch: int = 0,
        initial_best_metric: Optional[float] = None,
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device or resolve_device("auto")
        self.criterion = criterion or nn.CrossEntropyLoss()
        self.on_epoch_end = on_epoch_end
        self.start_epoch = start_epoch
        self.best_metric = initial_best_metric

        self.model.to(self.device)

        self.amp_enabled = bool(config.mixed_precision) and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler(device="cuda", enabled=self.amp_enabled)

    def _forward_loss(self, batch: Dict[str, Any]):
        images = batch["image"].to(self.device, non_blocking=True)
        labels = torch.as_tensor(batch["label"]).to(self.device, non_blocking=True)
        with torch.amp.autocast(device_type=self.device.type, enabled=self.amp_enabled):
            logits = self.model(images)
            loss = self.criterion(logits, labels)
        return logits, loss, labels, images.size(0)

    def train_one_epoch(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.train()
        accumulator = MetricAccumulator()
        accum_steps = max(1, self.config.gradient_accumulation_steps)
        self.optimizer.zero_grad(set_to_none=True)

        num_batches = len(self.train_loader)
        for step, batch in enumerate(self.train_loader):
            if max_steps is not None and step >= max_steps:
                break

            logits, loss, labels, batch_size = self._forward_loss(batch)
            self.scaler.scale(loss / accum_steps).backward()

            is_last_batch = step == num_batches - 1 or (max_steps is not None and step == max_steps - 1)
            should_step = ((step + 1) % accum_steps == 0) or is_last_batch
            if should_step:
                if self.config.gradient_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)

            preds = logits.detach().argmax(dim=1)
            accumulator.update(loss.item(), batch_size, preds, labels)

        return accumulator.compute()

    @torch.no_grad()
    def validate_one_epoch(self, max_steps: Optional[int] = None) -> Dict[str, float]:
        self.model.eval()
        accumulator = MetricAccumulator()
        for step, batch in enumerate(self.val_loader):
            if max_steps is not None and step >= max_steps:
                break
            logits, loss, labels, batch_size = self._forward_loss(batch)
            preds = logits.argmax(dim=1)
            accumulator.update(loss.item(), batch_size, preds, labels)
        return accumulator.compute(include_balanced_accuracy=True)

    def fit(self, max_train_steps_per_epoch: Optional[int] = None, max_val_steps_per_epoch: Optional[int] = None) -> FitResult:
        import time

        result = FitResult(amp_enabled=self.amp_enabled, device=str(self.device))
        best_epoch = 0
        epochs_without_improvement = 0

        for epoch in range(self.start_epoch, self.start_epoch + self.config.epochs):
            start_time = time.time()
            train_metrics = self.train_one_epoch(max_steps=max_train_steps_per_epoch)
            val_metrics = self.validate_one_epoch(max_steps=max_val_steps_per_epoch)
            elapsed = time.time() - start_time

            current_lr = self.optimizer.param_groups[0]["lr"]
            monitor_value = _monitor_value(val_metrics, self.config.monitor_metric)

            improved = _is_better(monitor_value, self.best_metric, self.config.mode)
            if improved:
                self.best_metric = monitor_value
                best_epoch = epoch
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            # Step the plateau scheduler AFTER computing `improved` against
            # the pre-step best, so scheduler behavior never influences
            # (or is influenced by) the early-stopping/checkpoint decision.
            self.scheduler.step(monitor_value)

            epoch_result = EpochResult(
                epoch=epoch,
                train_metrics=train_metrics,
                val_metrics=val_metrics,
                lr=current_lr,
                elapsed_seconds=elapsed,
                is_best=improved,
                monitor_value=monitor_value,
            )
            result.history.append(epoch_result)
            if self.on_epoch_end is not None:
                self.on_epoch_end(epoch_result)

            result.best_epoch = best_epoch
            result.best_metric = self.best_metric
            result.stopped_epoch = epoch

            if self.config.early_stopping_enabled and epochs_without_improvement >= self.config.early_stopping_patience:
                result.stopping_reason = (
                    f"early stopping: no improvement in '{self.config.monitor_metric}' for "
                    f"{epochs_without_improvement} epochs (patience={self.config.early_stopping_patience})"
                )
                break
        else:
            result.stopping_reason = "completed all configured epochs"

        return result

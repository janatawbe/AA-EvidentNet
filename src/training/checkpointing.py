"""Checkpoint build/save/load/compatibility-checking, shared by every model
the trainer will ever train (baselines now, AA-EvidentNet later).

A checkpoint is a single torch.save()'d dict containing not just the raw
state_dicts but enough provenance (model name, num_classes, seed, dataset
manifest hash, git commit, timestamp, full training config) to know
exactly what produced it without consulting anything else.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointIncompatibleError(Exception):
    """Raised when a checkpoint's architecture/class-count/etc. does not
    match what the caller is trying to resume into. Never silently
    ignored or coerced."""


@dataclass
class CheckpointMetadata:
    schema_version: int
    model_name: str
    architecture: str
    num_classes: int
    seed: int
    epoch: int
    best_metric: Optional[float]
    monitor_metric: str
    dataset_manifest_hash: str
    git_commit: Optional[str]
    timestamp_utc: str
    training_config: Dict[str, Any] = field(default_factory=dict)


def build_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    best_metric: Optional[float],
    monitor_metric: str,
    training_config: Dict[str, Any],
    seed: int,
    model_name: str,
    architecture: str,
    num_classes: int,
    dataset_manifest_hash: str,
    git_commit: Optional[str],
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """`scaler` (optional): the AMP `torch.amp.GradScaler` in use, if any.
    Saving its state matters only for CUDA+mixed-precision runs (on CPU the
    scaler is always disabled and its state is trivial) - without it, a
    CUDA run resumed from a checkpoint would restart AMP's adaptive loss
    scale from its default rather than where it left off. Harmless to omit
    (pass None, the default) for CPU-only runs or callers that predate
    this parameter."""
    metadata = CheckpointMetadata(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        model_name=model_name,
        architecture=architecture,
        num_classes=num_classes,
        seed=seed,
        epoch=epoch,
        best_metric=best_metric,
        monitor_metric=monitor_metric,
        dataset_manifest_hash=dataset_manifest_hash,
        git_commit=git_commit,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        training_config=training_config,
    )
    return {
        "metadata": asdict(metadata),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
    }


def save_checkpoint(checkpoint: Dict[str, Any], path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def load_checkpoint(path: Union[str, Path], map_location: str = "cpu") -> Dict[str, Any]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    # weights_only=False: this checkpoint intentionally stores plain-Python
    # metadata (dicts/strings/floats) alongside tensors, and is always one
    # this project wrote itself - never an untrusted third-party file.
    return torch.load(path, map_location=map_location, weights_only=False)


def assert_checkpoint_compatible(checkpoint: Dict[str, Any], model_name: str, num_classes: int) -> None:
    metadata = checkpoint.get("metadata", {})
    errors = []
    if metadata.get("model_name") != model_name:
        errors.append(f"checkpoint model_name '{metadata.get('model_name')}' != requested '{model_name}'")
    if metadata.get("num_classes") != num_classes:
        errors.append(f"checkpoint num_classes {metadata.get('num_classes')} != requested {num_classes}")
    if errors:
        raise CheckpointIncompatibleError(
            "Checkpoint is incompatible with the requested run:\n" + "\n".join(f"  - {e}" for e in errors)
        )


def restore_training_state(
    checkpoint: Dict[str, Any],
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    scaler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Restore model/optimizer/scheduler/AMP-scaler in place; return the
    epoch/best metric to resume from. Caller is responsible for calling
    assert_checkpoint_compatible() first.

    `scaler` (optional): the AMP `torch.amp.GradScaler` to restore state
    into. Checkpoints written before this parameter existed simply have no
    `scaler_state_dict` key (`.get(...)` returns None), so old checkpoints
    still resume correctly - the scaler just starts from its default
    state, same as always."""
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and checkpoint.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    if scaler is not None and checkpoint.get("scaler_state_dict") is not None:
        scaler.load_state_dict(checkpoint["scaler_state_dict"])

    metadata = checkpoint.get("metadata", {})
    return {"epoch": metadata.get("epoch", 0), "best_metric": metadata.get("best_metric")}

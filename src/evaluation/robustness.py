"""Robustness evaluation for already-frozen, already-finally-tested
checkpoints -- a later, additional test-time analysis, entirely separate
from Task 8's final held-out test evaluation (src/evaluation/final_test.py).

This module NEVER reads, writes, modifies, or overwrites anything Task 8
already produced: it does not touch results/raw_predictions/, does not
touch any existing results/tables/<eval_run_id>/ directory, and does not
update experiments/registry.csv. It writes only to its own,
separate results/robustness/<robustness_run_id>/ directory.

Reused, unmodified, from Task 8's final_test.py and the rest of the
codebase (see module docstring conventions there): the same frozen-model
loading (create_model / load_checkpoint / assert_checkpoint_compatible /
restore_training_state), the same canonical class ordering, and the same
test-manifest safeguard (RetinalDataset.from_manifest(...,
expected_split="test", require_all_original=True)). No optimizer, no
scheduler, and no backward pass exist anywhere in this module; inference
runs under torch.inference_mode() with model.eval() already set, exactly
like final_test.py.

Degradations are applied ONLY in memory, on an already-loaded [0,1]
(post-ToTensor, pre-Normalize) image tensor, in the evaluation loop below
-- data/raw/ is opened strictly read-only (via RetinalDataset, exactly as
final_test.py already does) and data/manifests/test_original.csv is never
written to. Severities are FIXED, predefined values (see
DEFAULT_DEGRADATION_SEVERITIES below and configs/evaluation.yaml:
robustness.degradations) -- never tuned from any observed result.
"""

import hashlib
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.records import write_csv
from src.data.transforms import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    DEFAULT_RESIZE_SIZE,
    build_pre_normalize_transform,
    normalize_tensor,
)
from src.evaluation.final_test import ALL_MODEL_NAMES, _effective_model_config
from src.evaluation.metrics import compute_overall_metrics
from src.models.factory import create_model
from src.training.checkpointing import (
    assert_checkpoint_compatible,
    load_checkpoint,
    restore_training_state,
)
from src.training.logging import generate_run_id
from src.training.trainer import resolve_device
from src.utils.config import hash_config, load_config
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file
from src.utils.seeding import DEFAULT_SEED, set_seed

CLEAN_REFERENCE_LABEL = "clean_reference"

# FIXED, predefined severities -- NOT tuned from any observed test
# performance. This constant is also the fallback default when
# configs/evaluation.yaml: robustness.degradations is absent; in normal
# operation the config file (which states these same values explicitly)
# is what run_robustness_evaluation actually reads, consistent with this
# project's existing config-driven convention (e.g.
# src/losses/cs_supcon.py: DEFAULT_TEMPERATURE mirrored in
# configs/losses.yaml).
DEFAULT_DEGRADATION_SEVERITIES: Dict[str, List[float]] = {
    "brightness": [0.70, 0.85, 1.15, 1.30],
    "contrast": [0.70, 0.85, 1.15, 1.30],
    "gaussian_noise": [0.02, 0.05, 0.10],
    "gaussian_blur": [0.5, 1.0, 2.0],
    "reduced_resolution": [168, 112, 56],
}

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 2


class RobustnessError(Exception):
    """Raised for a robustness-specific problem: an unknown model name, an
    unknown/invalid degradation or severity, or a missing test manifest.
    Checkpoint incompatibility is raised by the reused
    `assert_checkpoint_compatible` as `CheckpointIncompatibleError`."""


def _derive_noise_seed(base_seed: int, sample_id: str, severity: float) -> int:
    """Deterministic per-(base_seed, sample, severity) integer seed for
    reproducible Gaussian noise -- independent of batch order, device, or
    num_workers, since it never depends on anything but these three
    values. Uses the project's existing hashing convention (SHA-256) so
    reruns with the same evaluation seed always draw identical noise for
    the same sample at the same severity."""
    material = f"{base_seed}:{sample_id}:gaussian_noise:{severity}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return int(digest[:8], 16)  # fits well within torch.Generator.manual_seed's valid range


def _validate_condition(degradation: str, severity: Any, degradation_table: Dict[str, List[float]]) -> None:
    if degradation not in degradation_table:
        raise RobustnessError(f"Unknown degradation '{degradation}'. Known: {sorted(degradation_table)}")
    if severity not in degradation_table[degradation]:
        raise RobustnessError(
            f"Severity {severity!r} is not one of the fixed, predefined severities for "
            f"'{degradation}': {degradation_table[degradation]}"
        )


def apply_degradation(
    image: torch.Tensor,
    degradation: str,
    severity: Any,
    *,
    degradation_table: Dict[str, List[float]] = DEFAULT_DEGRADATION_SEVERITIES,
    base_seed: Optional[int] = None,
    sample_id: Optional[str] = None,
) -> torch.Tensor:
    """Apply one named degradation at one fixed severity to a single
    image tensor `[C, H, W]` in `[0, 1]` (post-ToTensor, pre-Normalize).
    Returns a new tensor of the same shape, values clamped to `[0, 1]`.

    Raises RobustnessError for an unknown degradation name or a severity
    not present in `degradation_table` for that degradation -- severities
    are fixed/predefined, never arbitrary.
    """
    _validate_condition(degradation, severity, degradation_table)
    if image.dim() != 3:
        raise RobustnessError(f"apply_degradation expects a single [C, H, W] image tensor, got shape {tuple(image.shape)}")

    if degradation == "brightness":
        return TF.adjust_brightness(image, float(severity)).clamp(0.0, 1.0)

    if degradation == "contrast":
        return TF.adjust_contrast(image, float(severity)).clamp(0.0, 1.0)

    if degradation == "gaussian_noise":
        if base_seed is None or sample_id is None:
            raise RobustnessError("gaussian_noise requires base_seed and sample_id for reproducible, sample-specific noise")
        generator = torch.Generator().manual_seed(_derive_noise_seed(base_seed, sample_id, severity))
        noise = torch.randn(image.shape, generator=generator) * float(severity)
        return (image + noise).clamp(0.0, 1.0)

    if degradation == "gaussian_blur":
        sigma = float(severity)
        # Standard, PROVISIONAL "+-3 sigma" kernel-size rule (always odd)
        # -- not tuned, just a conventional coverage choice.
        kernel_size = int(2 * math.ceil(3 * sigma) + 1)
        return TF.gaussian_blur(image, kernel_size=[kernel_size, kernel_size], sigma=[sigma, sigma]).clamp(0.0, 1.0)

    if degradation == "reduced_resolution":
        target_size = int(severity)
        _, h, w = image.shape
        downsampled = F.interpolate(image.unsqueeze(0), size=(target_size, target_size), mode="bilinear", align_corners=False)
        restored = F.interpolate(downsampled, size=(h, w), mode="bilinear", align_corners=False)
        return restored.squeeze(0).clamp(0.0, 1.0)

    raise RobustnessError(f"No implementation for degradation '{degradation}'")  # unreachable given _validate_condition above


def _resolve_num_workers(device: torch.device, num_workers_override: Optional[int], config_num_workers: int) -> int:
    if num_workers_override is not None:
        return num_workers_override
    if device.type == "cpu":
        return 0
    return config_num_workers


def _iter_conditions(degradation_table: Dict[str, List[float]], include_clean_reference: bool):
    if include_clean_reference:
        yield CLEAN_REFERENCE_LABEL, None
    for degradation, severities in degradation_table.items():
        for severity in severities:
            yield degradation, severity


@dataclass
class RobustnessSummary:
    robustness_run_id: str
    model_name: str
    checkpoint_path: str
    checkpoint_hash: str
    device: str
    num_samples: int
    class_names: List[str]
    metrics_path: str
    metadata_path: str
    rows: List[Dict[str, Any]]


def run_robustness_evaluation(
    model_name: str,
    checkpoint_path: Union[str, Path],
    dataset_config_path: Union[str, Path] = "configs/dataset.yaml",
    models_config_path: Union[str, Path] = "configs/models.yaml",
    evaluation_config_path: Union[str, Path] = "configs/evaluation.yaml",
    seed: int = DEFAULT_SEED,
    device_override: Optional[str] = None,
    num_workers_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    include_clean_reference: bool = True,
) -> RobustnessSummary:
    if model_name not in ALL_MODEL_NAMES:
        raise RobustnessError(f"Unknown model '{model_name}'. Known models: {list(ALL_MODEL_NAMES)}")

    set_seed(seed)

    dataset_config = load_config(dataset_config_path)
    models_config = load_config(models_config_path)
    evaluation_config = load_config(evaluation_config_path)
    robustness_cfg = evaluation_config.get("robustness", {}) or {}
    degradation_table = robustness_cfg.get("degradations", DEFAULT_DEGRADATION_SEVERITIES)

    device = resolve_device(device_override or "auto")
    num_workers = _resolve_num_workers(device, num_workers_override, robustness_cfg.get("num_workers", DEFAULT_NUM_WORKERS))
    batch_size = batch_size_override if batch_size_override is not None else robustness_cfg.get("batch_size", DEFAULT_BATCH_SIZE)

    canonical_classes = sorted(dataset_config["class_directory_mapping"].keys())
    num_classes = len(canonical_classes)

    image_cfg = dataset_config.get("image", {}) or {}
    image_size = image_cfg.get("size", DEFAULT_IMAGE_SIZE)
    resize_size = image_cfg.get("resize_size", DEFAULT_RESIZE_SIZE)
    normalize_mean = image_cfg.get("normalize_mean", DEFAULT_NORMALIZE_MEAN)
    normalize_std = image_cfg.get("normalize_std", DEFAULT_NORMALIZE_STD)
    # Deliberately NOT build_transforms_from_config()'s normalized eval
    # transform: degradations must be applied in [0,1] pixel-value space,
    # before normalization (see module docstring) - normalize_tensor() is
    # applied explicitly, per-sample, after the degradation below.
    pre_normalize_transform = build_pre_normalize_transform(image_size, resize_size)

    raw_dir = Path(dataset_config["paths"]["raw_dir"])
    processed_train_dir = Path(dataset_config["paths"]["processed_dir"]) / "train"
    manifests_dir = Path(dataset_config["paths"]["manifests_dir"])
    test_manifest_path = manifests_dir / "test_original.csv"
    if not test_manifest_path.is_file():
        raise RobustnessError(f"{test_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    assert_checkpoint_compatible(checkpoint, model_name, num_classes)
    checkpoint_hash = hash_file(checkpoint_path)
    checkpoint_metadata = checkpoint.get("metadata", {}) or {}
    training_run_id = checkpoint_path.parent.name  # inferred from path convention, see final_test.py

    effective_models_config = _effective_model_config(model_name, models_config)
    model = create_model(model_name, effective_models_config)
    restore_training_state(checkpoint, model)  # weights only - never modified afterward
    model.to(device)
    model.eval()

    test_dataset = RetinalDataset.from_manifest(
        test_manifest_path,
        canonical_classes,
        raw_dir,
        processed_train_dir,
        transform=pre_normalize_transform,
        expected_split="test",
        require_all_original=True,
    )
    test_manifest_hash = hash_file(test_manifest_path)
    num_samples = len(test_dataset)

    is_aa_evidentnet = model_name == "aa_evidentnet"
    rows: List[Dict[str, Any]] = []

    for degradation, severity in _iter_conditions(degradation_table, include_clean_reference):
        loader = build_eval_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)
        y_true: List[int] = []
        y_pred: List[int] = []
        probabilities_all: List[List[float]] = []
        uncertainties: List[float] = []

        with torch.inference_mode():
            for batch in loader:
                raw_images = batch["image"]  # [B, C, H, W] in [0, 1], not yet normalized
                labels = batch["label"]
                sample_ids = batch["original_id"]

                processed = []
                for i in range(raw_images.size(0)):
                    img = raw_images[i]
                    if degradation != CLEAN_REFERENCE_LABEL:
                        img = apply_degradation(
                            img, degradation, severity,
                            degradation_table=degradation_table, base_seed=seed, sample_id=sample_ids[i],
                        )
                    img = normalize_tensor(img, normalize_mean, normalize_std)
                    processed.append(img)
                images = torch.stack(processed, dim=0).to(device)

                if is_aa_evidentnet:
                    output = model(images, return_features=True)
                    logits = output.logits
                    uncertainties.extend(output.uncertainty.detach().cpu().tolist())
                else:
                    logits = model(images)

                probabilities = torch.softmax(logits, dim=1)
                preds = torch.argmax(logits, dim=1)

                labels_list = labels.detach().cpu().tolist() if torch.is_tensor(labels) else list(labels)
                y_true.extend(int(v) for v in labels_list)
                y_pred.extend(preds.detach().cpu().tolist())
                probabilities_all.extend(probabilities.detach().cpu().tolist())

        overall = compute_overall_metrics(y_true, y_pred, probabilities_all, num_classes)
        mean_uncertainty = float(np.mean(uncertainties)) if is_aa_evidentnet and uncertainties else None
        rows.append(
            {
                "model": model_name,
                "degradation": degradation,
                "severity": "" if severity is None else severity,
                "n": len(y_true),
                "accuracy": overall["accuracy"],
                "balanced_accuracy": overall["balanced_accuracy"],
                "macro_f1": overall["macro_f1"],
                "mean_uncertainty": "" if mean_uncertainty is None else mean_uncertainty,
            }
        )

    # --- outputs: entirely separate from Task 8's results/raw_predictions/
    # and results/tables/<finaltest_run_id>/ - a fresh directory root. ---
    robustness_run_id = generate_run_id(f"robustness_{model_name}", seed, smoke_test=False)
    output_paths_cfg = evaluation_config.get("output_paths", {}) or {}
    robustness_base_dir = Path(output_paths_cfg.get("robustness_dir", "results/robustness")) / robustness_run_id
    robustness_base_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = robustness_base_dir / "robustness_metrics.csv"
    metrics_columns = ["model", "degradation", "severity", "n", "accuracy", "balanced_accuracy", "macro_f1", "mean_uncertainty"]
    write_csv(rows, metrics_columns, metrics_path)

    config_hash = hash_config({"dataset": dataset_config, "models": models_config, "evaluation": evaluation_config})
    metadata = {
        "robustness_run_id": robustness_run_id,
        "model_name": model_name,
        "model_architecture": checkpoint_metadata.get("architecture"),
        "training_run_id_inferred_from_checkpoint_path": training_run_id,
        "training_seed": checkpoint_metadata.get("seed"),
        "eval_invocation_seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_best_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_monitor_metric": checkpoint_metadata.get("monitor_metric"),
        "checkpoint_best_metric": checkpoint_metadata.get("best_metric"),
        "test_manifest_path": str(test_manifest_path),
        "test_manifest_sha256": test_manifest_hash,
        "class_names": canonical_classes,
        "num_classes": num_classes,
        "num_samples": num_samples,
        "seed": seed,
        "device": str(device),
        "degradations": degradation_table,
        "include_clean_reference": include_clean_reference,
        "dataset_config_path": str(dataset_config_path),
        "models_config_path": str(models_config_path),
        "evaluation_config_path": str(evaluation_config_path),
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(metrics_path),
    }
    metadata_path = robustness_base_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    metadata["metadata_path"] = str(metadata_path)

    return RobustnessSummary(
        robustness_run_id=robustness_run_id,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        device=str(device),
        num_samples=num_samples,
        class_names=canonical_classes,
        metrics_path=str(metrics_path),
        metadata_path=str(metadata_path),
        rows=rows,
    )

"""Feature-distance OOD detector combined with EDL uncertainty, for the
already-frozen, already-finally-tested AA-EvidentNet checkpoint only.

This is a NEW evaluation component, entirely separate from Task 8's clean
final-test evaluation (src/evaluation/final_test.py) and from the
robustness evaluation (src/evaluation/robustness.py, whose degradation
definitions/severities and frozen-model-loading helpers this module reuses
unmodified). It never retrains, fine-tunes, or otherwise modifies
AA-EvidentNet's weights, and never overwrites anything either of those two
modules produced.

Motivation: under severe Gaussian noise (std=0.10), AA-EvidentNet's own
EDL uncertainty was observed to DECREASE even as accuracy collapsed
(robustness evaluation) - i.e. the model becomes confidently wrong rather
than appropriately uncertain. This module adds a second, independent
OOD-awareness signal (distance, in the model's own fused-embedding space,
to the nearest per-class training prototype) and combines it with EDL
uncertainty, in the hope that at least one of the two signals rises when
the other fails.

Method (train/val only - never test - for every calibration decision):
  1. Reuse the frozen checkpoint's fused embedding
     (AAEvidentNetOutput.embedding, `forward(images, return_features=True)`)
     - the same shared global+local representation the classifier and EDL
     head already use; no new model code, no new forward path.
  2. Forward-pass train_original.csv (the SAME manifest training already
     used) once, with no augmentation/degradation, to compute one class
     prototype per class = the mean fused embedding of that class's
     training-original images.
  3. Forward-pass val_original.csv once to compute, for every validation
     sample: its EDL uncertainty, its COSINE distance (1 - cosine
     similarity) to the nearest class prototype, and whether the frozen
     checkpoint's own prediction was correct.
  4. Fit min-max normalization for both signals FROM VAL_ORIGINAL'S OWN
     DISTRIBUTION (not train - train samples define the prototypes, so
     their own distances to "their" prototype are biased low and would
     make an unrepresentatively tight normalization range). Normalized
     values are floored at 0 but deliberately NOT capped above 1, so a
     severely out-of-distribution test-time sample can (and should) push
     a normalized score arbitrarily high.
  5. Choose the combination weight in
     `combined = normalized_edl + weight * normalized_ood`
     via a small, fixed grid search, selecting the weight that maximizes
     error-detection AUROC (is the frozen checkpoint's prediction wrong?)
     on VAL_ORIGINAL's own predictions - never on test. Ties are broken
     toward the smallest weight (grid iterated in ascending order, first
     max wins).

Once calibrated (steps 1-5 above), the method is frozen and evaluated,
read-only, on data/manifests/test_original.csv (clean) and on every
robustness condition from src/evaluation/robustness.py - test_original.csv
is never used for calibration, only for this final, one-shot evaluation
step, using RetinalDataset's same expected_split="test",
require_all_original=True safeguard as final_test.py/robustness.py.

Outputs are written only under results/ood_uncertainty/<run_id>/ - never
into results/raw_predictions/, any results/tables/<eval_run_id>/, or
results/robustness/<robustness_run_id>/, and experiments/registry.csv is
never read or written by this module.
"""

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.records import write_csv
from src.data.transforms import (
    DEFAULT_IMAGE_SIZE,
    DEFAULT_NORMALIZE_MEAN,
    DEFAULT_NORMALIZE_STD,
    DEFAULT_RESIZE_SIZE,
    build_pre_normalize_transform,
    build_transforms_from_config,
    normalize_tensor,
)
from src.evaluation.final_test import _effective_model_config
from src.evaluation.robustness import (
    CLEAN_REFERENCE_LABEL,
    DEFAULT_DEGRADATION_SEVERITIES,
    _iter_conditions,
    apply_degradation,
)
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

# This feature is only defined for AA-EvidentNet: it is the only registered
# model with a fused embedding + EDL evidential head. Baselines have
# neither, so there is nothing for this module to combine.
MODEL_NAME = "aa_evidentnet"

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 2

# FIXED, predefined candidate weights for combine_weight selection - not an
# arbitrary continuous search, so the calibration procedure is exactly
# reproducible and auditable (same philosophy as robustness.py's fixed
# degradation severities). The value actually chosen is always the one
# that maximizes val_original error-detection AUROC (see
# select_combine_weight below); this list only bounds the search space.
DEFAULT_WEIGHT_GRID: List[float] = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]

# Fallback only - normal operation reads configs/evaluation.yaml:
# selective_prediction.coverage_levels (already used, unmodified, by that
# section's own placeholder docs).
DEFAULT_COVERAGE_LEVELS: List[float] = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]


class OODUncertaintyError(Exception):
    """Raised for an OOD-uncertainty-specific problem: a non-AA-EvidentNet
    model name, a missing train/val/test manifest, or an internal
    invariant violation (e.g. a training class with zero samples, making
    its prototype undefined). Checkpoint incompatibility is raised by the
    reused `assert_checkpoint_compatible` as `CheckpointIncompatibleError`,
    not wrapped here."""


@dataclass
class NormalizationParams:
    min: float
    max: float


@dataclass
class OODCalibration:
    """The frozen, train/val-only-derived calibration: prototypes,
    normalization ranges, and the combination weight. Computed fresh on
    every call (cheap, deterministic, no randomness) rather than cached to
    a separate artifact file, so every run's metadata.json fully documents
    exactly how these numbers were derived from the checkpoint + manifests
    used that run."""

    prototypes: np.ndarray  # [num_classes, embedding_dim]
    class_sample_counts: List[int]
    edl_norm: NormalizationParams
    ood_norm: NormalizationParams
    weight: float
    weight_grid_results: List[Dict[str, Any]]
    train_num_samples: int
    val_num_samples: int


@dataclass
class OODUncertaintySummary:
    run_id: str
    model_name: str
    checkpoint_path: str
    checkpoint_hash: str
    device: str
    num_test_samples: int
    class_names: List[str]
    weight: float
    metrics_path: str
    risk_coverage_path: str
    figure_path: str
    metadata_path: str
    rows: List[Dict[str, Any]]


def _resolve_num_workers(device: torch.device, num_workers_override: Optional[int], config_num_workers: int) -> int:
    if num_workers_override is not None:
        return num_workers_override
    if device.type == "cpu":
        return 0
    return config_num_workers


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def nearest_prototype_cosine_distance(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Cosine distance (`1 - cosine_similarity`) from each row of
    `embeddings` [N, D] to its NEAREST row of `prototypes` [K, D] (i.e. the
    minimum distance across all K class prototypes). Neither array needs
    to be pre-normalized - L2-normalization happens here. Returns an
    [N]-shaped array in [0, 2] (0 = identical direction, 2 = opposite
    direction)."""
    if embeddings.ndim != 2:
        raise OODUncertaintyError(f"embeddings must be 2-D [N, D], got shape {embeddings.shape}")
    if prototypes.ndim != 2:
        raise OODUncertaintyError(f"prototypes must be 2-D [K, D], got shape {prototypes.shape}")
    if embeddings.shape[1] != prototypes.shape[1]:
        raise OODUncertaintyError(
            f"embedding dim {embeddings.shape[1]} != prototype dim {prototypes.shape[1]}"
        )
    embeddings_unit = _l2_normalize_rows(embeddings.astype(np.float64))
    prototypes_unit = _l2_normalize_rows(prototypes.astype(np.float64))
    similarity = embeddings_unit @ prototypes_unit.T  # [N, K]
    distance = 1.0 - similarity
    return distance.min(axis=1)


def _fit_minmax(values: np.ndarray) -> NormalizationParams:
    return NormalizationParams(min=float(np.min(values)), max=float(np.max(values)))


def _apply_minmax(values: np.ndarray, params: NormalizationParams) -> np.ndarray:
    """`clip((x - min) / (max - min), 0, None)` - floored at 0 (a value
    below the val-fitted minimum normalizes to 0, never negative), but
    deliberately NOT capped above 1, so a severely out-of-distribution
    sample can push its normalized score arbitrarily high (the exact
    signal this module exists to produce). A degenerate (zero-span) fitted
    range - e.g. every val value identical, which can happen with a tiny
    synthetic test fixture - normalizes every value to 0.0 rather than
    dividing by zero."""
    span = params.max - params.min
    values = np.asarray(values, dtype=np.float64)
    if span <= 0:
        return np.zeros_like(values)
    return np.clip((values - params.min) / span, 0.0, None)


def _error_detection_auroc(is_error: np.ndarray, score: np.ndarray) -> Optional[float]:
    positives = int(is_error.sum())
    negatives = int(is_error.size - positives)
    if positives == 0 or negatives == 0:
        return None
    return float(roc_auc_score(is_error, score))


def _error_detection_auprc(is_error: np.ndarray, score: np.ndarray) -> Optional[float]:
    positives = int(is_error.sum())
    if positives == 0:
        return None
    return float(average_precision_score(is_error, score))


def select_combine_weight(
    is_error_val: np.ndarray,
    normalized_edl_val: np.ndarray,
    normalized_ood_val: np.ndarray,
    weight_grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> Tuple[float, List[Dict[str, Any]]]:
    """Grid search over `weight_grid`, maximizing error-detection AUROC of
    `normalized_edl_val + weight * normalized_ood_val` against
    `is_error_val`, computed ENTIRELY on val_original - never on test.
    Ties (including "AUROC undefined for every candidate", e.g. a val set
    with zero misclassifications) are broken toward the smallest weight,
    since `weight_grid` is iterated in ascending order and only a STRICT
    improvement replaces the current best. Returns (chosen_weight,
    per-candidate results) so the full search is always recorded in
    metadata.json, not just the winner."""
    results: List[Dict[str, Any]] = []
    best_weight = float(weight_grid[0])
    best_score = -1.0
    for weight in weight_grid:
        combined = normalized_edl_val + float(weight) * normalized_ood_val
        auroc = _error_detection_auroc(is_error_val, combined)
        results.append({"weight": float(weight), "val_error_detection_auroc": auroc})
        candidate_score = auroc if auroc is not None else -1.0
        if candidate_score > best_score:
            best_score = candidate_score
            best_weight = float(weight)
    return best_weight, results


def compute_class_prototypes(
    model: torch.nn.Module, loader, device: torch.device, num_classes: int
) -> Tuple[np.ndarray, List[int]]:
    """One forward pass over `loader` (train_original.csv), accumulating
    the mean fused embedding (`AAEvidentNetOutput.embedding`) per class.
    Raises OODUncertaintyError if any of the `num_classes` classes has zero
    samples in this loader - a prototype is then genuinely undefined,
    never silently fabricated as a zero vector."""
    sums: Optional[np.ndarray] = None
    counts = np.zeros(num_classes, dtype=np.int64)

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

            output = model(images, return_features=True)
            embeddings = output.embedding.detach().cpu().numpy()
            if sums is None:
                sums = np.zeros((num_classes, embeddings.shape[1]), dtype=np.float64)

            for i in range(embeddings.shape[0]):
                class_idx = int(labels_np[i])
                sums[class_idx] += embeddings[i]
                counts[class_idx] += 1

    if sums is None:
        raise OODUncertaintyError("train_original.csv produced zero batches - cannot compute class prototypes")

    missing = [k for k in range(num_classes) if counts[k] == 0]
    if missing:
        raise OODUncertaintyError(
            f"train_original.csv has zero samples for class index/es {missing} - a prototype is undefined for "
            "these classes; refusing to fabricate one"
        )

    prototypes = sums / counts[:, None]
    return prototypes, counts.tolist()


def calibrate_ood_uncertainty(
    model: torch.nn.Module,
    device: torch.device,
    canonical_classes: Sequence[str],
    raw_dir: Path,
    processed_train_dir: Path,
    manifests_dir: Path,
    eval_transform,
    batch_size: int,
    num_workers: int,
    weight_grid: Sequence[float] = DEFAULT_WEIGHT_GRID,
) -> OODCalibration:
    """Steps 1-5 of the module docstring's method: prototypes from
    train_original.csv, normalization + combine weight from
    val_original.csv. Never reads test_original.csv."""
    num_classes = len(canonical_classes)

    train_manifest_path = manifests_dir / "train_original.csv"
    if not train_manifest_path.is_file():
        raise OODUncertaintyError(f"{train_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first.")
    val_manifest_path = manifests_dir / "val_original.csv"
    if not val_manifest_path.is_file():
        raise OODUncertaintyError(f"{val_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first.")

    train_dataset = RetinalDataset.from_manifest(
        train_manifest_path, canonical_classes, raw_dir, processed_train_dir,
        transform=eval_transform, expected_split="train", require_all_original=True,
    )
    train_loader = build_eval_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)
    prototypes, class_sample_counts = compute_class_prototypes(model, train_loader, device, num_classes)

    val_dataset = RetinalDataset.from_manifest(
        val_manifest_path, canonical_classes, raw_dir, processed_train_dir,
        transform=eval_transform, expected_split="val", require_all_original=True,
    )
    val_loader = build_eval_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    val_uncertainty: List[float] = []
    val_ood_distance: List[float] = []
    val_is_error: List[int] = []

    with torch.inference_mode():
        for batch in val_loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

            output = model(images, return_features=True)
            embeddings = output.embedding.detach().cpu().numpy()
            preds = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
            uncertainty = output.uncertainty.detach().cpu().numpy()
            distances = nearest_prototype_cosine_distance(embeddings, prototypes)

            val_uncertainty.extend(uncertainty.tolist())
            val_ood_distance.extend(distances.tolist())
            val_is_error.extend((preds != labels_np).astype(int).tolist())

    val_uncertainty_arr = np.asarray(val_uncertainty, dtype=np.float64)
    val_ood_distance_arr = np.asarray(val_ood_distance, dtype=np.float64)
    val_is_error_arr = np.asarray(val_is_error, dtype=np.int64)

    edl_norm = _fit_minmax(val_uncertainty_arr)
    ood_norm = _fit_minmax(val_ood_distance_arr)

    normalized_edl_val = _apply_minmax(val_uncertainty_arr, edl_norm)
    normalized_ood_val = _apply_minmax(val_ood_distance_arr, ood_norm)

    weight, weight_grid_results = select_combine_weight(val_is_error_arr, normalized_edl_val, normalized_ood_val, weight_grid)

    return OODCalibration(
        prototypes=prototypes,
        class_sample_counts=class_sample_counts,
        edl_norm=edl_norm,
        ood_norm=ood_norm,
        weight=weight,
        weight_grid_results=weight_grid_results,
        train_num_samples=len(train_dataset),
        val_num_samples=len(val_dataset),
    )


def _process_condition(
    model: torch.nn.Module,
    device: torch.device,
    test_dataset: RetinalDataset,
    degradation: str,
    severity: Any,
    degradation_table: Dict[str, List[float]],
    seed: int,
    normalize_mean: Sequence[float],
    normalize_std: Sequence[float],
    calibration: OODCalibration,
    batch_size: int,
    num_workers: int,
) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    loader = build_eval_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)
    y_true: List[int] = []
    y_pred: List[int] = []
    raw_uncertainty: List[float] = []
    raw_ood_distance: List[float] = []

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

            output = model(images, return_features=True)
            preds = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
            embeddings = output.embedding.detach().cpu().numpy()
            uncertainty = output.uncertainty.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

            distances = nearest_prototype_cosine_distance(embeddings, calibration.prototypes)

            y_true.extend(int(v) for v in labels_np)
            y_pred.extend(int(v) for v in preds)
            raw_uncertainty.extend(uncertainty.tolist())
            raw_ood_distance.extend(distances.tolist())

    y_true_arr = np.asarray(y_true, dtype=np.int64)
    y_pred_arr = np.asarray(y_pred, dtype=np.int64)
    raw_uncertainty_arr = np.asarray(raw_uncertainty, dtype=np.float64)
    raw_ood_distance_arr = np.asarray(raw_ood_distance, dtype=np.float64)
    is_error = (y_true_arr != y_pred_arr).astype(np.int64)

    normalized_edl = _apply_minmax(raw_uncertainty_arr, calibration.edl_norm)
    normalized_ood = _apply_minmax(raw_ood_distance_arr, calibration.ood_norm)
    combined = normalized_edl + calibration.weight * normalized_ood

    n = len(y_true_arr)
    accuracy = float((y_true_arr == y_pred_arr).mean()) if n > 0 else None

    row: Dict[str, Any] = {
        "model": MODEL_NAME,
        "degradation": degradation,
        "severity": "" if severity is None else severity,
        "n": n,
        "accuracy": accuracy,
        "mean_edl_uncertainty": float(np.mean(normalized_edl)) if n > 0 else None,
        "mean_ood_score": float(np.mean(normalized_ood)) if n > 0 else None,
        "mean_combined_score": float(np.mean(combined)) if n > 0 else None,
        "raw_mean_edl_uncertainty": float(np.mean(raw_uncertainty_arr)) if n > 0 else None,
        "raw_mean_ood_distance": float(np.mean(raw_ood_distance_arr)) if n > 0 else None,
    }
    for score_name, score_values in (("edl", normalized_edl), ("ood", normalized_ood), ("combined", combined)):
        row[f"{score_name}_error_auroc"] = _error_detection_auroc(is_error, score_values)
        row[f"{score_name}_error_auprc"] = _error_detection_auprc(is_error, score_values)

    raw = {
        "is_error": is_error,
        "normalized_edl": normalized_edl,
        "normalized_ood": normalized_ood,
        "combined": combined,
    }
    return row, raw


def _corruption_strength(degradation: str, severity: Any, image_size: int = DEFAULT_IMAGE_SIZE) -> float:
    """A monotonic "how severe is this condition" convention used ONLY for
    the severity-vs-score correlation analysis below - it plays no role in
    the actual degradation, normalization, or combination logic anywhere
    else in this module. brightness/contrast severities are factors
    centered on 1.0 (no change), so strength is distance from 1.0;
    gaussian_noise/gaussian_blur severities already increase monotonically
    with corruption; reduced_resolution severities are target pixel sizes,
    so SMALLER means MORE corruption - strength is `image_size - target`."""
    if degradation in ("brightness", "contrast"):
        return abs(float(severity) - 1.0)
    if degradation in ("gaussian_noise", "gaussian_blur"):
        return float(severity)
    if degradation == "reduced_resolution":
        return float(image_size) - float(severity)
    raise OODUncertaintyError(f"no corruption-strength convention defined for degradation '{degradation}'")


def _spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation via pandas' tie-aware `.rank()` (average
    ranks for ties) followed by an ordinary Pearson correlation of the
    ranks - mathematically identical to Spearman's rho, reusing an
    already-declared project dependency (pandas) rather than adding scipy
    or hand-rolling tie handling. Returns None when undefined (fewer than
    2 points, or zero variance in either ranked series)."""
    if len(x) < 2 or len(y) < 2:
        return None
    x_ranks = pd.Series(x).rank().to_numpy()
    y_ranks = pd.Series(y).rank().to_numpy()
    if np.std(x_ranks) == 0 or np.std(y_ranks) == 0:
        return None
    return float(np.corrcoef(x_ranks, y_ranks)[0, 1])


def _compute_severity_correlations(
    rows: List[Dict[str, Any]], degradation_table: Dict[str, List[float]], image_size: int = DEFAULT_IMAGE_SIZE
) -> List[Dict[str, Any]]:
    """Per degradation family, per score, the Spearman correlation between
    `_corruption_strength(...)` and the condition's mean score - answering
    "does the score increase appropriately as this degradation gets more
    severe?" The clean_reference row (strength 0.0) is included as the
    anchor for every family when present, since "no corruption" is the
    natural starting point of each family's severity trend."""
    clean_rows = [r for r in rows if r["degradation"] == CLEAN_REFERENCE_LABEL]
    correlation_rows: List[Dict[str, Any]] = []
    for degradation in degradation_table:
        family_rows = [r for r in rows if r["degradation"] == degradation]
        if not family_rows:
            continue
        strengths = [_corruption_strength(degradation, r["severity"], image_size) for r in family_rows]
        combined_rows = family_rows
        combined_strengths = strengths
        if clean_rows:
            combined_rows = clean_rows + family_rows
            combined_strengths = [0.0] + strengths
        for score_name in ("mean_edl_uncertainty", "mean_ood_score", "mean_combined_score"):
            values = [r[score_name] for r in combined_rows]
            correlation_rows.append(
                {
                    "degradation": degradation,
                    "score": score_name,
                    "spearman_correlation_vs_severity": _spearman(combined_strengths, values),
                    "n_conditions": len(combined_rows),
                }
            )
    return correlation_rows


def _compute_risk_coverage(raw: Dict[str, np.ndarray], coverage_levels: Sequence[float]) -> List[Dict[str, Any]]:
    """Selective risk/coverage on the clean test condition only (the
    standard use case: how many of the most-confident samples can be
    auto-accepted, and what's the resulting error rate among them), for
    each of the three scores. Samples are ranked by ASCENDING score (lower
    = more confident, both for EDL uncertainty and for the OOD/combined
    scores by construction) and the lowest-`coverage` fraction is
    retained."""
    is_error = raw["is_error"]
    n = len(is_error)
    rows: List[Dict[str, Any]] = []
    for score_name, key in (("edl", "normalized_edl"), ("ood", "normalized_ood"), ("combined", "combined")):
        scores = raw[key]
        order = np.argsort(scores, kind="stable")
        sorted_is_error = is_error[order]
        for coverage in coverage_levels:
            k = max(int(round(float(coverage) * n)), 0)
            if k == 0:
                rows.append({"score_type": score_name, "coverage": coverage, "n_retained": 0, "risk": "", "accuracy_retained": ""})
                continue
            retained_errors = sorted_is_error[:k]
            risk = float(retained_errors.mean())
            rows.append(
                {
                    "score_type": score_name,
                    "coverage": coverage,
                    "n_retained": k,
                    "risk": risk,
                    "accuracy_retained": 1.0 - risk,
                }
            )
    return rows


def _make_figure(rows: List[Dict[str, Any]], degradation_table: Dict[str, List[float]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")  # headless-safe: never opens a display, safe under pytest/CI/Colab
    import matplotlib.pyplot as plt

    clean_rows = [r for r in rows if r["degradation"] == CLEAN_REFERENCE_LABEL]
    degradations = [d for d in degradation_table if any(r["degradation"] == d for r in rows)]
    n_panels = max(len(degradations), 1)
    ncols = min(3, n_panels)
    nrows = math.ceil(n_panels / ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)

    series = (
        ("mean_edl_uncertainty", "EDL uncertainty", "-o"),
        ("mean_ood_score", "OOD score (cosine)", "-s"),
        ("mean_combined_score", "Combined", "-^"),
    )

    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx >= len(degradations):
            ax.axis("off")
            continue
        degradation = degradations[idx]
        family_rows = sorted((r for r in rows if r["degradation"] == degradation), key=lambda r: float(r["severity"]))
        severities = [float(r["severity"]) for r in family_rows]
        for score_name, label, style in series:
            values = [r[score_name] for r in family_rows]
            ax.plot(severities, values, style, label=label)
            if clean_rows and clean_rows[0][score_name] is not None:
                ax.axhline(clean_rows[0][score_name], linestyle="--", alpha=0.4, linewidth=1)
        ax.set_title(degradation)
        ax.set_xlabel("severity")
        ax.set_ylabel("normalized score")
        ax.legend(fontsize=8)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def run_ood_uncertainty_evaluation(
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
) -> OODUncertaintySummary:
    if model_name != MODEL_NAME:
        raise OODUncertaintyError(
            f"OOD/EDL combination is only defined for '{MODEL_NAME}' (fused embedding + evidential head); "
            f"got '{model_name}'. Baselines have neither."
        )

    set_seed(seed)

    dataset_config = load_config(dataset_config_path)
    models_config = load_config(models_config_path)
    evaluation_config = load_config(evaluation_config_path)
    ood_cfg = evaluation_config.get("ood_uncertainty", {}) or {}
    degradation_table = ood_cfg.get("degradations", DEFAULT_DEGRADATION_SEVERITIES)
    weight_grid = ood_cfg.get("weight_grid", DEFAULT_WEIGHT_GRID)
    coverage_levels = (evaluation_config.get("selective_prediction", {}) or {}).get("coverage_levels", DEFAULT_COVERAGE_LEVELS)

    device = resolve_device(device_override or "auto")
    num_workers = _resolve_num_workers(device, num_workers_override, ood_cfg.get("num_workers", DEFAULT_NUM_WORKERS))
    batch_size = batch_size_override if batch_size_override is not None else ood_cfg.get("batch_size", DEFAULT_BATCH_SIZE)

    canonical_classes = sorted(dataset_config["class_directory_mapping"].keys())
    num_classes = len(canonical_classes)

    image_cfg = dataset_config.get("image", {}) or {}
    image_size = image_cfg.get("size", DEFAULT_IMAGE_SIZE)
    resize_size = image_cfg.get("resize_size", DEFAULT_RESIZE_SIZE)
    normalize_mean = image_cfg.get("normalize_mean", DEFAULT_NORMALIZE_MEAN)
    normalize_std = image_cfg.get("normalize_std", DEFAULT_NORMALIZE_STD)
    # Train/val: the ordinary, already-normalized eval transform (no
    # degradation ever applied to them). Test: the pre-normalize transform
    # only, so degradations can be applied in [0,1] space exactly like
    # robustness.py, before normalize_tensor() is called explicitly below.
    _, eval_transform = build_transforms_from_config(dataset_config)
    pre_normalize_transform = build_pre_normalize_transform(image_size, resize_size)

    raw_dir = Path(dataset_config["paths"]["raw_dir"])
    processed_train_dir = Path(dataset_config["paths"]["processed_dir"]) / "train"
    manifests_dir = Path(dataset_config["paths"]["manifests_dir"])
    test_manifest_path = manifests_dir / "test_original.csv"
    if not test_manifest_path.is_file():
        raise OODUncertaintyError(f"{test_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    assert_checkpoint_compatible(checkpoint, model_name, num_classes)
    checkpoint_hash = hash_file(checkpoint_path)
    checkpoint_metadata = checkpoint.get("metadata", {}) or {}
    training_run_id = checkpoint_path.parent.name

    effective_models_config = _effective_model_config(model_name, models_config)
    model = create_model(model_name, effective_models_config)
    restore_training_state(checkpoint, model)  # weights only - never modified afterward
    model.to(device)
    model.eval()

    # --- calibration: train_original.csv (prototypes) + val_original.csv
    # (normalization + combine weight). Never reads test_original.csv. ---
    calibration = calibrate_ood_uncertainty(
        model=model, device=device, canonical_classes=canonical_classes, raw_dir=raw_dir,
        processed_train_dir=processed_train_dir, manifests_dir=manifests_dir, eval_transform=eval_transform,
        batch_size=batch_size, num_workers=num_workers, weight_grid=weight_grid,
    )

    # --- frozen-method evaluation: clean test_original.csv + every
    # robustness condition. ---
    test_dataset = RetinalDataset.from_manifest(
        test_manifest_path, canonical_classes, raw_dir, processed_train_dir,
        transform=pre_normalize_transform, expected_split="test", require_all_original=True,
    )
    test_manifest_hash = hash_file(test_manifest_path)

    rows: List[Dict[str, Any]] = []
    condition_raw: Dict[Tuple[str, Any], Dict[str, np.ndarray]] = {}
    for degradation, severity in _iter_conditions(degradation_table, include_clean_reference):
        row, raw = _process_condition(
            model, device, test_dataset, degradation, severity, degradation_table, seed,
            normalize_mean, normalize_std, calibration, batch_size, num_workers,
        )
        rows.append(row)
        condition_raw[(degradation, "" if severity is None else severity)] = raw

    correlation_rows = _compute_severity_correlations(rows, degradation_table, image_size)

    risk_coverage_rows: List[Dict[str, Any]] = []
    clean_key = (CLEAN_REFERENCE_LABEL, "")
    if clean_key in condition_raw:
        risk_coverage_rows = _compute_risk_coverage(condition_raw[clean_key], coverage_levels)

    # --- outputs: entirely separate from final_test's and robustness's. ---
    run_id = generate_run_id(f"ood_uncertainty_{model_name}", seed, smoke_test=False)
    output_paths_cfg = evaluation_config.get("output_paths", {}) or {}
    base_dir = Path(output_paths_cfg.get("ood_uncertainty_dir", "results/ood_uncertainty")) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = base_dir / "metrics.csv"
    metrics_columns = [
        "model", "degradation", "severity", "n", "accuracy",
        "mean_edl_uncertainty", "mean_ood_score", "mean_combined_score",
        "raw_mean_edl_uncertainty", "raw_mean_ood_distance",
        "edl_error_auroc", "edl_error_auprc",
        "ood_error_auroc", "ood_error_auprc",
        "combined_error_auroc", "combined_error_auprc",
    ]
    write_csv(rows, metrics_columns, metrics_path)

    risk_coverage_path = base_dir / "selective_risk_coverage.csv"
    risk_coverage_columns = ["score_type", "coverage", "n_retained", "risk", "accuracy_retained"]
    write_csv(risk_coverage_rows, risk_coverage_columns, risk_coverage_path)

    figure_path = base_dir / "severity_vs_score.png"
    _make_figure(rows, degradation_table, figure_path)

    config_hash = hash_config({"dataset": dataset_config, "models": models_config, "evaluation": evaluation_config})
    train_manifest_path = manifests_dir / "train_original.csv"
    val_manifest_path = manifests_dir / "val_original.csv"
    metadata = {
        "run_id": run_id,
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
        "train_manifest_path": str(train_manifest_path),
        "train_manifest_sha256": hash_file(train_manifest_path),
        "val_manifest_path": str(val_manifest_path),
        "val_manifest_sha256": hash_file(val_manifest_path),
        "test_manifest_path": str(test_manifest_path),
        "test_manifest_sha256": test_manifest_hash,
        "class_names": canonical_classes,
        "num_classes": num_classes,
        "train_num_samples": calibration.train_num_samples,
        "val_num_samples": calibration.val_num_samples,
        "num_test_samples": len(test_dataset),
        "class_sample_counts_train": calibration.class_sample_counts,
        "prototype_distance_metric": "cosine",
        "edl_normalization": {"min": calibration.edl_norm.min, "max": calibration.edl_norm.max, "fit_on": "val_original.csv"},
        "ood_normalization": {"min": calibration.ood_norm.min, "max": calibration.ood_norm.max, "fit_on": "val_original.csv"},
        "combine_weight": calibration.weight,
        "combine_weight_grid": list(weight_grid),
        "combine_weight_grid_results": calibration.weight_grid_results,
        "combine_weight_selection_criterion": (
            "maximize error-detection AUROC (is the checkpoint's own prediction wrong?) on val_original.csv only; "
            "ties broken toward the smallest weight"
        ),
        "severity_correlations": correlation_rows,
        "coverage_levels": list(coverage_levels),
        "degradations": degradation_table,
        "include_clean_reference": include_clean_reference,
        "seed": seed,
        "device": str(device),
        "dataset_config_path": str(dataset_config_path),
        "models_config_path": str(models_config_path),
        "evaluation_config_path": str(evaluation_config_path),
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "metrics_path": str(metrics_path),
        "risk_coverage_path": str(risk_coverage_path),
        "figure_path": str(figure_path),
    }
    metadata_path = base_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return OODUncertaintySummary(
        run_id=run_id,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        device=str(device),
        num_test_samples=len(test_dataset),
        class_names=canonical_classes,
        weight=calibration.weight,
        metrics_path=str(metrics_path),
        risk_coverage_path=str(risk_coverage_path),
        figure_path=str(figure_path),
        metadata_path=str(metadata_path),
        rows=rows,
    )

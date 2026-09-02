"""Phase 3: one-time orchestration that builds the frozen, train-only
class-affinity ambiguity quantities (margin_scale, boundary_gap_scale, the
class-affinity matrix) from a REFERENCE checkpoint's embeddings over
train_original.csv - a separate, additional research direction alongside
Phase 1's class-PROTOTYPE ambiguity (src/training/ambiguity_setup.py) and
Phase 2's cross-class NEIGHBORHOOD ambiguity
(src/training/neighborhood_ambiguity_setup.py), both left completely
unmodified and intact.

Pipeline (see src/losses/class_affinity_ambiguity.py for the exact math):

    existing frozen reference checkpoint (read-only)
        -> forward pass over train_original.csv (eval mode, no augmentation)
           (src.models.prototypes.extract_embeddings - the SAME shared
           extraction loop Phase 1/Phase 2/OOD-uncertainty use)
        -> self-excluded class affinities for every train sample
           (compute_class_affinities(..., exclude_self=True))
        -> margin_scale = 95th percentile of train top1-top2 margins
        -> boundary_gap_scale = 95th percentile of train boundary gaps
           (ANALYSIS ONLY - label-aware, never an inference-time score)
        -> class-affinity matrix (compute_class_affinity_matrix)

THIS IS ANALYSIS/RESEARCH ONLY: nothing here is (or is meant to be)
installed into CSSupConLoss or wired into src.training.run_aa_evidentnet.

Read-only guarantees (identical to Phase 1's/Phase 2's setup modules): the
reference checkpoint is loaded into its own throwaway model instance;
every parameter's `requires_grad` is forced to `False`; inference runs
under `torch.inference_mode()` with `model.eval()`; only
train_original.csv is read (never train_balanced.csv, val_original.csv,
or test_original.csv).
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.transforms import build_transforms_from_config
from src.losses.class_affinity_ambiguity import (
    DEFAULT_M,
    DEFAULT_SCALE_PERCENTILE,
    DEFAULT_TEMPERATURE,
    ClassAffinityAmbiguityError,
    compute_class_affinities,
    compute_class_affinity_matrix,
    compute_label_aware_boundary_gap,
    compute_top_affinities,
    fit_boundary_gap_scale,
    fit_margin_scale,
)
from src.models.factory import create_model
from src.models.prototypes import PrototypeComputationError, extract_embeddings
from src.training.checkpointing import assert_checkpoint_compatible, load_checkpoint, restore_training_state
from src.utils.hashing import hash_file

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


class ClassAffinityAmbiguitySetupError(Exception):
    """Raised for a problem building the frozen class-affinity ambiguity
    quantities: a missing train_original.csv, or a numerically-degenerate
    train-derived scale (re-raised from
    `src.losses.class_affinity_ambiguity.ClassAffinityAmbiguityError` /
    `src.models.prototypes.PrototypeComputationError`). Checkpoint
    incompatibility is raised by the reused `assert_checkpoint_compatible`
    as `CheckpointIncompatibleError`, not wrapped here."""


@dataclass
class ClassAffinityAmbiguityArtifact:
    """Everything a subsequent validation-analysis step needs from this
    one-time setup step. `train_embeddings`/`train_labels` are kept (not
    just a compressed summary) because validation-time class affinities
    must be computed against the full set of individual TRAIN embeddings."""

    matrix_numpy: np.ndarray  # [K, K], symmetric, bounded [0,1], zero diagonal
    margin_scale: float
    boundary_gap_scale: float
    train_affinities_self_excluded: np.ndarray  # [N, K]
    train_embeddings: np.ndarray
    train_labels: np.ndarray
    m: int
    temperature: float
    scale_percentile: float
    reference_checkpoint_path: str
    reference_checkpoint_sha256: str
    reference_model_name: str
    reference_checkpoint_architecture: Any
    train_manifest_path: str
    train_manifest_sha256: str
    canonical_classes: List[str]
    num_train_samples: int


def build_class_affinity_ambiguity(
    reference_checkpoint_path: Union[str, Path],
    reference_model_name: str,
    models_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    canonical_classes: Sequence[str],
    raw_dir: Union[str, Path],
    processed_train_dir: Union[str, Path],
    train_manifest_path: Union[str, Path],
    device: torch.device,
    m: int = DEFAULT_M,
    temperature: float = DEFAULT_TEMPERATURE,
    scale_percentile: float = DEFAULT_SCALE_PERCENTILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> ClassAffinityAmbiguityArtifact:
    """Build the frozen class-affinity ambiguity matrix and train-derived
    scales from a REFERENCE checkpoint's embeddings over
    train_original.csv only.

    The reference checkpoint is loaded read-only into its own model
    instance (never a model being trained - this module trains nothing)
    and is used strictly under `torch.inference_mode()`: no optimizer is
    ever constructed, no backward pass ever occurs, and its weights are
    discarded once this function returns.
    """
    canonical_classes = list(canonical_classes)
    num_classes = len(canonical_classes)
    train_manifest_path = Path(train_manifest_path)
    if not train_manifest_path.is_file():
        raise ClassAffinityAmbiguitySetupError(
            f"{train_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first."
        )

    checkpoint_path = Path(reference_checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    assert_checkpoint_compatible(checkpoint, reference_model_name, num_classes)
    checkpoint_hash = hash_file(checkpoint_path)
    checkpoint_metadata = checkpoint.get("metadata", {}) or {}

    reference_model = create_model(reference_model_name, models_config)
    restore_training_state(checkpoint, reference_model)  # weights only, read-only from here on
    reference_model.to(device)
    reference_model.eval()
    for parameter in reference_model.parameters():
        parameter.requires_grad_(False)

    _, eval_transform = build_transforms_from_config(dataset_config)
    train_dataset = RetinalDataset.from_manifest(
        train_manifest_path,
        canonical_classes,
        raw_dir,
        processed_train_dir,
        transform=eval_transform,
        expected_split="train",
        require_all_original=True,
    )
    loader = build_eval_dataloader(train_dataset, batch_size=batch_size, num_workers=num_workers)

    try:
        extracted = extract_embeddings(reference_model, loader, device)
    except PrototypeComputationError as e:
        raise ClassAffinityAmbiguitySetupError(str(e)) from e

    try:
        train_affinities = compute_class_affinities(
            extracted.embeddings, extracted.embeddings, extracted.labels, num_classes, m=m, exclude_self=True
        )
        train_top = compute_top_affinities(train_affinities)
        margin_scale = fit_margin_scale(train_top.raw_margin, percentile=scale_percentile)

        train_boundary_gap = compute_label_aware_boundary_gap(train_affinities, extracted.labels)
        boundary_gap_scale = fit_boundary_gap_scale(train_boundary_gap, percentile=scale_percentile)

        matrix_numpy = compute_class_affinity_matrix(train_affinities, extracted.labels, num_classes)
    except ClassAffinityAmbiguityError as e:
        raise ClassAffinityAmbiguitySetupError(str(e)) from e

    return ClassAffinityAmbiguityArtifact(
        matrix_numpy=matrix_numpy,
        margin_scale=margin_scale,
        boundary_gap_scale=boundary_gap_scale,
        train_affinities_self_excluded=train_affinities,
        train_embeddings=extracted.embeddings,
        train_labels=extracted.labels,
        m=m,
        temperature=temperature,
        scale_percentile=scale_percentile,
        reference_checkpoint_path=str(checkpoint_path),
        reference_checkpoint_sha256=checkpoint_hash,
        reference_model_name=reference_model_name,
        reference_checkpoint_architecture=checkpoint_metadata.get("architecture"),
        train_manifest_path=str(train_manifest_path),
        train_manifest_sha256=hash_file(train_manifest_path),
        canonical_classes=canonical_classes,
        num_train_samples=len(train_dataset),
    )

"""Phase 2: one-time orchestration that builds the frozen, cross-class
NEIGHBORHOOD ambiguity matrix (and per-train-sample neighborhood-ambiguity
fields) from a REFERENCE checkpoint's embeddings over train_original.csv -
a separate, additional research direction alongside Phase 1's class-
PROTOTYPE ambiguity (src/training/ambiguity_setup.py, left completely
unmodified and intact for comparison).

Pipeline (see src/losses/neighborhood_ambiguity.py for the exact math):

    existing frozen reference checkpoint (read-only)
        -> forward pass over train_original.csv (eval mode, no augmentation)
           (src.models.prototypes.extract_embeddings - the SAME shared
           extraction loop Phase 1/OOD-uncertainty use, never a new copy)
        -> cross-class k-NN (find_cross_class_neighbors)
        -> per-sample neighborhood ambiguity (compute_sample_neighborhood_ambiguity)
        -> class-level neighborhood ambiguity matrix (compute_neighborhood_class_matrix)

THIS IS ANALYSIS/RESEARCH ONLY: unlike Phase 1's `build_learned_class_ambiguity`,
nothing here is (or is meant to be) installed into CSSupConLoss or wired
into src.training.run_aa_evidentnet - there is no training-time consumer
of this module yet. It exists so the neighborhood-based approach can be
evaluated (src/evaluation/neighborhood_ambiguity_validation.py) before any
decision is made about whether it should ever influence training.

Read-only guarantees (identical to Phase 1's ambiguity_setup.py): the
reference checkpoint is loaded into its own throwaway model instance,
never the model being trained (there is no "model being trained" in this
module at all); every parameter's `requires_grad` is forced to `False`;
inference runs under `torch.inference_mode()` with `model.eval()`; only
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
from src.losses.neighborhood_ambiguity import (
    DEFAULT_K,
    CrossClassNeighbors,
    NeighborhoodAmbiguityError,
    SampleNeighborhoodAmbiguityResult,
    compute_neighborhood_class_matrix,
    compute_sample_neighborhood_ambiguity,
    find_cross_class_neighbors,
)
from src.models.factory import create_model
from src.models.prototypes import PrototypeComputationError, extract_embeddings
from src.training.checkpointing import assert_checkpoint_compatible, load_checkpoint, restore_training_state
from src.utils.hashing import hash_file

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


class NeighborhoodAmbiguitySetupError(Exception):
    """Raised for a problem building the frozen neighborhood-ambiguity
    matrix: a missing train_original.csv, or too few cross-class
    candidates for the requested k (re-raised from
    `src.losses.neighborhood_ambiguity.NeighborhoodAmbiguityError` /
    `src.models.prototypes.PrototypeComputationError`). Checkpoint
    incompatibility is raised by the reused `assert_checkpoint_compatible`
    as `CheckpointIncompatibleError`, not wrapped here."""


@dataclass
class NeighborhoodAmbiguityArtifact:
    """Everything a subsequent validation-analysis step (or, in a later,
    not-yet-implemented phase, a training-time consumer) needs from this
    one-time setup step. `train_embeddings`/`train_labels` are kept (not
    just a compressed summary, unlike Phase 1's per-class prototypes)
    because neighborhood-based validation ambiguity must compare each
    validation embedding against the full set of individual TRAIN
    embeddings, not a per-class mean."""

    matrix_numpy: np.ndarray  # [K, K], symmetric, bounded [0,1], zero diagonal
    sample_ambiguity: SampleNeighborhoodAmbiguityResult
    neighbors: CrossClassNeighbors
    train_embeddings: np.ndarray
    train_labels: np.ndarray
    k: int
    reference_checkpoint_path: str
    reference_checkpoint_sha256: str
    reference_model_name: str
    reference_checkpoint_architecture: Any
    train_manifest_path: str
    train_manifest_sha256: str
    canonical_classes: List[str]
    num_train_samples: int


def build_neighborhood_class_ambiguity(
    reference_checkpoint_path: Union[str, Path],
    reference_model_name: str,
    models_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    canonical_classes: Sequence[str],
    raw_dir: Union[str, Path],
    processed_train_dir: Union[str, Path],
    train_manifest_path: Union[str, Path],
    device: torch.device,
    k: int = DEFAULT_K,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> NeighborhoodAmbiguityArtifact:
    """Build the frozen cross-class neighborhood ambiguity matrix (and
    per-sample neighborhood-ambiguity fields) from a REFERENCE checkpoint's
    embeddings over train_original.csv only.

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
        raise NeighborhoodAmbiguitySetupError(
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
        raise NeighborhoodAmbiguitySetupError(str(e)) from e

    try:
        neighbors = find_cross_class_neighbors(extracted.embeddings, extracted.labels, k=k)
        sample_ambiguity = compute_sample_neighborhood_ambiguity(neighbors, num_classes)
        matrix_numpy = compute_neighborhood_class_matrix(neighbors, extracted.labels, num_classes)
    except NeighborhoodAmbiguityError as e:
        raise NeighborhoodAmbiguitySetupError(str(e)) from e

    return NeighborhoodAmbiguityArtifact(
        matrix_numpy=matrix_numpy,
        sample_ambiguity=sample_ambiguity,
        neighbors=neighbors,
        train_embeddings=extracted.embeddings,
        train_labels=extracted.labels,
        k=k,
        reference_checkpoint_path=str(checkpoint_path),
        reference_checkpoint_sha256=checkpoint_hash,
        reference_model_name=reference_model_name,
        reference_checkpoint_architecture=checkpoint_metadata.get("architecture"),
        train_manifest_path=str(train_manifest_path),
        train_manifest_sha256=hash_file(train_manifest_path),
        canonical_classes=canonical_classes,
        num_train_samples=len(train_dataset),
    )

"""One-time orchestration for the learned class-ambiguity mechanism
(Phase 1) - builds the frozen class-ambiguity matrix (and the margin
normalization needed for sample-ambiguity analysis) BEFORE a new
ambiguity-aware AA-EvidentNet training run begins.

Pipeline (see src/losses/ambiguity.py for the exact math, and
REPRODUCIBILITY.md for the full methodological discussion):

    existing frozen reference checkpoint (read-only)
        -> forward pass over train_original.csv (eval mode, no augmentation)
        -> class prototypes (src.models.prototypes.compute_class_prototypes)
        -> class-ambiguity matrix (src.losses.ambiguity.compute_class_ambiguity_matrix)
        -> margin normalization for sample-ambiguity analysis
           (src.losses.ambiguity.fit_margin_normalization, same forward pass)

This is called exactly ONCE, before src.training.trainer.Trainer.fit() is
ever invoked for the new run - there is no periodic recomputation, no
warm-start-then-freeze epoch, and no change to Trainer or its
on_epoch_end hook. The reference checkpoint is loaded into its own,
throwaway model instance that is never the model being trained, is never
optimized, never receives gradients, and is discarded once this module's
function returns; only train_original.csv is read here (never
train_balanced.csv, val_original.csv, or test_original.csv).

METHODOLOGICAL CAVEAT (see REPRODUCIBILITY.md for the full discussion):
the reference checkpoint's own embedding space was itself shaped by the
EXISTING fixed-hard-pair CS-SupCon objective (ambiguity_weight on 3
clinician-picked pairs). The learned matrix built here therefore reflects
class geometry AFTER that existing correction has already partially acted
- it is not a from-scratch, assumption-free measurement of natural class
confusability. This is disclosed, not hidden, and is not claimed to be
independent of the previous ambiguity mechanism.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.transforms import build_transforms_from_config
from src.losses.ambiguity import (
    MarginNormalization,
    class_ambiguity_matrix_to_buffer,
    compute_class_ambiguity_matrix,
    compute_raw_margins,
    fit_margin_normalization,
)
from src.models.factory import create_model
from src.models.prototypes import compute_class_prototypes
from src.training.checkpointing import assert_checkpoint_compatible, load_checkpoint, restore_training_state
from src.utils.hashing import hash_file

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


class AmbiguitySetupError(Exception):
    """Raised for a problem building the frozen learned class-ambiguity
    matrix: a missing/incompatible reference checkpoint, or a missing
    train_original.csv. Checkpoint incompatibility beyond a missing file
    is raised by the reused `assert_checkpoint_compatible` as
    `CheckpointIncompatibleError`, not wrapped here."""


@dataclass
class LearnedAmbiguityArtifact:
    """Everything the new training run and its reproducibility metadata
    need from this one-time setup step."""

    matrix_buffer: torch.Tensor  # [K, K], non-trainable (requires_grad=False)
    matrix_numpy: np.ndarray
    prototypes: np.ndarray
    class_sample_counts: List[int]
    margin_normalization: MarginNormalization
    reference_checkpoint_path: str
    reference_checkpoint_sha256: str
    reference_model_name: str
    reference_checkpoint_architecture: Any
    train_manifest_path: str
    train_manifest_sha256: str
    canonical_classes: List[str]
    num_train_samples: int


def build_learned_class_ambiguity(
    reference_checkpoint_path: Union[str, Path],
    reference_model_name: str,
    models_config: Dict[str, Any],
    dataset_config: Dict[str, Any],
    canonical_classes: Sequence[str],
    raw_dir: Union[str, Path],
    processed_train_dir: Union[str, Path],
    train_manifest_path: Union[str, Path],
    device: torch.device,
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> LearnedAmbiguityArtifact:
    """Build the frozen class-ambiguity matrix and sample-ambiguity margin
    normalization from a REFERENCE checkpoint's embeddings over
    train_original.csv only.

    The reference checkpoint (`reference_checkpoint_path`,
    `reference_model_name`) is loaded read-only into its own model
    instance - built fresh via `create_model`, never the model instance
    the caller is about to train - and is used strictly under
    `torch.inference_mode()` with `model.eval()`: no optimizer is ever
    constructed for it, no backward pass ever touches it, and its weights
    are discarded (garbage-collected) once this function returns. Nothing
    about the model the caller subsequently trains is read, modified, or
    depended upon here.
    """
    canonical_classes = list(canonical_classes)
    num_classes = len(canonical_classes)
    train_manifest_path = Path(train_manifest_path)
    if not train_manifest_path.is_file():
        raise AmbiguitySetupError(
            f"{train_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first."
        )

    checkpoint_path = Path(reference_checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    assert_checkpoint_compatible(checkpoint, reference_model_name, num_classes)
    checkpoint_hash = hash_file(checkpoint_path)
    checkpoint_metadata = checkpoint.get("metadata", {}) or {}

    # A dedicated, throwaway model instance - distinct from (and never
    # reused as) the model the caller trains afterward.
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

    prototypes, class_sample_counts = compute_class_prototypes(reference_model, loader, device, num_classes)
    matrix_numpy = compute_class_ambiguity_matrix(prototypes)
    matrix_buffer = class_ambiguity_matrix_to_buffer(matrix_numpy)

    # Second, separate pass over the SAME train_original.csv loader (never
    # validation/test) to fit the sample-ambiguity margin normalization -
    # kept as a distinct pass rather than fused into compute_class_prototypes
    # above, since the two serve different purposes (class prototypes vs.
    # a per-sample margin distribution) and this keeps each step simple
    # and independently testable, at the cost of one extra (cheap, CPU-
    # feasible) forward pass over train_original.csv.
    embeddings_chunks = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            output = reference_model(images, return_features=True)
            embeddings_chunks.append(output.embedding.detach().cpu().numpy())
    all_embeddings = np.concatenate(embeddings_chunks, axis=0)
    raw_margins = compute_raw_margins(all_embeddings, prototypes)
    margin_normalization = fit_margin_normalization(raw_margins)

    return LearnedAmbiguityArtifact(
        matrix_buffer=matrix_buffer,
        matrix_numpy=matrix_numpy,
        prototypes=prototypes,
        class_sample_counts=class_sample_counts,
        margin_normalization=margin_normalization,
        reference_checkpoint_path=str(checkpoint_path),
        reference_checkpoint_sha256=checkpoint_hash,
        reference_model_name=reference_model_name,
        reference_checkpoint_architecture=checkpoint_metadata.get("architecture"),
        train_manifest_path=str(train_manifest_path),
        train_manifest_sha256=hash_file(train_manifest_path),
        canonical_classes=canonical_classes,
        num_train_samples=len(train_dataset),
    )

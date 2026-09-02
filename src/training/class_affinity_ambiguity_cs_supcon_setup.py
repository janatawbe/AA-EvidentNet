"""Experimental (feature/learned-ambiguity, Phase 3-experimental)
orchestration: install Phase 3's continuous class-affinity ambiguity
matrix into CS-SupCon's `ambiguity_source="learned_class_affinity"` mode,
as a controlled-experiment alternative to the existing fixed-hard-pair
weighting - mirroring src/training/ambiguity_setup.py's role for Phase 1's
`ambiguity_source="learned_class"` mode, but reusing Phase 3's OWN,
completely unmodified matrix-construction pipeline
(src/training/class_affinity_ambiguity_setup.py: build_class_affinity_ambiguity)
rather than re-deriving its equations.

This module performs NO new math. It is a thin wrapper:

    build_class_affinity_ambiguity(...)   [UNCHANGED, Phase 3's own function]
        -> matrix_numpy  [K, K], symmetric, bounded [0,1], zero diagonal,
           built from a REFERENCE checkpoint's embeddings over
           train_original.csv only (m=5, temperature=0.1, predetermined
           BEFORE the Phase 3 validation analysis - see
           src/losses/class_affinity_ambiguity.py)
        -> class_ambiguity_matrix_to_buffer(...)   [UNCHANGED, src/losses/ambiguity.py,
           already generic to any [K, K] numpy matrix - not specific to
           Phase 1's own prototype-based matrix despite living in that
           module]
        -> a frozen, non-trainable torch.Tensor ready for
           CSSupConLoss.set_learned_ambiguity_matrix(...)

The resulting matrix is built exactly ONCE, before optimizer/scheduler/
Trainer are ever constructed (see src/training/run_aa_evidentnet.py's
"learned_class_affinity" branch), never recomputed during training, and
never receives gradients (register_buffer, requires_grad=False - verified
by tests/test_class_affinity_ambiguity_cs_supcon_setup.py).

The reference checkpoint (`reference_checkpoint_path`, `reference_model_name`)
is loaded read-only into its own throwaway model instance - identical
read-only guarantees to every other setup module in this project (eval
mode, every parameter's requires_grad forced False, torch.inference_mode()
throughout, weights discarded once the function returns) - and is
completely independent of whatever model the caller subsequently trains:
this module never touches, initializes, or depends on the experimental
model in any way.

Only train_original.csv is read (never train_balanced.csv,
val_original.csv, or test_original.csv) - this module's public function
has no test-manifest parameter at all.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union

import numpy as np
import torch

from src.losses.ambiguity import class_ambiguity_matrix_to_buffer
from src.losses.class_affinity_ambiguity import DEFAULT_M, DEFAULT_SCALE_PERCENTILE, DEFAULT_TEMPERATURE
from src.training.class_affinity_ambiguity_setup import (
    ClassAffinityAmbiguitySetupError,
    build_class_affinity_ambiguity,
)

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0

MATRIX_CONSTRUCTION_METHOD = (
    "phase3_class_affinity_matrix: A[a,b] built by "
    "src.training.class_affinity_ambiguity_setup.build_class_affinity_ambiguity "
    "(directed(a->b) = mean over train samples of class a of their self-excluded "
    "top-m cosine affinity to class b; symmetrized (directed(a->b)+directed(b->a))/2, "
    "diagonal=0; min-max rescaled using only its own off-diagonal entries) - "
    "train_original.csv only, unmodified from Phase 3."
)

# Re-exported so callers of this module never need to reach back into
# src.training.class_affinity_ambiguity_setup just to catch this error.
__all__ = [
    "ClassAffinityAmbiguityCSSupConArtifact",
    "MATRIX_CONSTRUCTION_METHOD",
    "build_class_affinity_ambiguity_for_cs_supcon",
    "ClassAffinityAmbiguitySetupError",
]


@dataclass
class ClassAffinityAmbiguityCSSupConArtifact:
    """Everything a controlled experimental training run and its
    reproducibility metadata need to install Phase 3's class-affinity
    matrix into `CSSupConLoss(ambiguity_source="learned_class_affinity")`."""

    matrix_buffer: torch.Tensor  # [K, K], non-trainable (requires_grad=False)
    matrix_numpy: np.ndarray  # [K, K], symmetric, bounded [0,1], zero diagonal
    matrix_construction_method: str
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


def build_class_affinity_ambiguity_for_cs_supcon(
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
) -> ClassAffinityAmbiguityCSSupConArtifact:
    """Build Phase 3's frozen class-affinity matrix (via the EXISTING,
    completely unmodified
    src.training.class_affinity_ambiguity_setup.build_class_affinity_ambiguity
    - never a re-derivation of its equations) and wrap it as a frozen,
    non-trainable buffer ready for
    `CSSupConLoss.set_learned_ambiguity_matrix(...)`.

    `m`/`temperature`/`scale_percentile` default to Phase 3's own
    predetermined constants (src.losses.class_affinity_ambiguity -
    m=5 was fixed before the Phase 3 validation analysis ever ran), so
    calling this with no overrides reuses exactly the same construction
    Phase 3 already validated - not a new, separately-tuned configuration.

    Has no test-manifest parameter. Raises
    ClassAffinityAmbiguitySetupError (re-raised, unchanged, from the
    wrapped call) for a missing train_original.csv or a numerically
    degenerate train-derived scale.
    """
    artifact = build_class_affinity_ambiguity(
        reference_checkpoint_path=reference_checkpoint_path,
        reference_model_name=reference_model_name,
        models_config=models_config,
        dataset_config=dataset_config,
        canonical_classes=canonical_classes,
        raw_dir=raw_dir,
        processed_train_dir=processed_train_dir,
        train_manifest_path=train_manifest_path,
        device=device,
        m=m,
        temperature=temperature,
        scale_percentile=scale_percentile,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    matrix_buffer = class_ambiguity_matrix_to_buffer(artifact.matrix_numpy)

    return ClassAffinityAmbiguityCSSupConArtifact(
        matrix_buffer=matrix_buffer,
        matrix_numpy=artifact.matrix_numpy,
        matrix_construction_method=MATRIX_CONSTRUCTION_METHOD,
        m=artifact.m,
        temperature=artifact.temperature,
        scale_percentile=artifact.scale_percentile,
        reference_checkpoint_path=artifact.reference_checkpoint_path,
        reference_checkpoint_sha256=artifact.reference_checkpoint_sha256,
        reference_model_name=artifact.reference_model_name,
        reference_checkpoint_architecture=artifact.reference_checkpoint_architecture,
        train_manifest_path=artifact.train_manifest_path,
        train_manifest_sha256=artifact.train_manifest_sha256,
        canonical_classes=artifact.canonical_classes,
        num_train_samples=artifact.num_train_samples,
    )

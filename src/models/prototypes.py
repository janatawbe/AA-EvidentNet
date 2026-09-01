"""Shared class-prototype and cosine-distance utilities.

Used by two independent, unrelated consumers that must never depend on
each other:

  - src/evaluation/ood_uncertainty.py (post-hoc feature-distance OOD
    detection on an already-frozen, already-finally-tested checkpoint)
  - src/losses/ambiguity.py / src/training/ambiguity_setup.py (the learned
    class-ambiguity mechanism, computed once before a NEW training run
    begins)

Both need exactly the same two primitives - "mean embedding per class from
a dataloader" and "cosine distance from an embedding to its nearest
prototype" - so this module exists purely to hold that shared,
dependency-free implementation once. It imports nothing from
src/evaluation/ or src/losses/ (avoiding an upward/backward layering
dependency in either direction) and is otherwise unopinionated about why a
caller wants prototypes.
"""

from typing import List, Tuple

import numpy as np
import torch


class PrototypeComputationError(Exception):
    """Raised for a prototype/cosine-distance-specific problem: a
    malformed (non-2D, or mismatched-dimension) input, or a class with
    zero samples in the provided dataloader (a prototype is then
    genuinely undefined - never fabricated as a zero vector)."""


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
        raise PrototypeComputationError(f"embeddings must be 2-D [N, D], got shape {embeddings.shape}")
    if prototypes.ndim != 2:
        raise PrototypeComputationError(f"prototypes must be 2-D [K, D], got shape {prototypes.shape}")
    if embeddings.shape[1] != prototypes.shape[1]:
        raise PrototypeComputationError(
            f"embedding dim {embeddings.shape[1]} != prototype dim {prototypes.shape[1]}"
        )
    embeddings_unit = _l2_normalize_rows(embeddings.astype(np.float64))
    prototypes_unit = _l2_normalize_rows(prototypes.astype(np.float64))
    similarity = embeddings_unit @ prototypes_unit.T  # [N, K]
    distance = 1.0 - similarity
    return distance.min(axis=1)


def compute_class_prototypes(
    model: torch.nn.Module, loader, device: torch.device, num_classes: int
) -> Tuple[np.ndarray, List[int]]:
    """One forward pass over `loader`, accumulating the mean fused
    embedding (`AAEvidentNetOutput.embedding`, via
    `model(images, return_features=True)`) per class. Caller is
    responsible for ensuring `loader` yields undistorted, non-augmented
    samples (e.g. built from train_original.csv with an eval-style
    transform) and for putting `model` in eval mode / running under
    `torch.inference_mode()` beforehand if that guarantee matters to the
    caller - this function itself always runs its forward passes under
    `torch.inference_mode()` regardless.

    Raises PrototypeComputationError if any of the `num_classes` classes
    has zero samples in this loader - a prototype is then genuinely
    undefined, never silently fabricated as a zero vector.
    """
    sums = None
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
        raise PrototypeComputationError("the dataloader produced zero batches - cannot compute class prototypes")

    missing = [k for k in range(num_classes) if counts[k] == 0]
    if missing:
        raise PrototypeComputationError(
            f"zero samples for class index/es {missing} - a prototype is undefined for "
            "these classes; refusing to fabricate one"
        )

    prototypes = sums / counts[:, None]
    return prototypes, counts.tolist()

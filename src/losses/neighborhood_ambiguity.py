"""Phase 2: learned class-level ambiguity from cross-class NEIGHBORHOOD
structure in the frozen AA-EvidentNet embedding space - a separate,
additional research direction alongside Phase 1's class-PROTOTYPE
ambiguity (src/losses/ambiguity.py), which remains unchanged and intact.

Motivation (see REPRODUCIBILITY.md for the full discussion): Phase 1's
validation results showed the largest observed validation confusion
(Healthy <-> Glaucoma, 35 confusions) was NOT strongly identified by
class-prototype cosine similarity - averaging every sample in a class
into a single centroid can hide difficult boundary regions where only a
SUBSET of each class's samples actually sit near the other class. This
module tests whether looking at cross-class NEAREST NEIGHBORS (a local,
sample-level notion of overlap) captures that boundary structure better.
Phase 1 is kept exactly as-is specifically so the two can be compared.

This module is PURE MATH ONLY - no I/O, no checkpoints, no dataloaders -
matching src/losses/ambiguity.py's/cs_supcon.py's/evidential.py's existing
purity convention. Orchestration (loading the reference checkpoint,
extracting embeddings) lives in
src/training/neighborhood_ambiguity_setup.py and
src/evaluation/neighborhood_ambiguity_validation.py.

THIS PHASE IS ANALYSIS/RESEARCH ONLY: nothing here is wired into
CSSupConLoss or any training objective. No performance or novelty claim
is made - the entire point of this module is to MEASURE whether
neighborhood-based ambiguity is more informative than Phase 1's
prototype-based ambiguity, not to assume the answer.

Efficiency note: for N train samples and embedding dim D, this module
computes similarity via ordinary [N, D] @ [D, N] matrix multiplication
(never an [N, N, D] tensor) - for this project's real dataset sizes
(N=3075 train, N=880 val), the resulting [N, N] similarity matrix is a
few tens of MB, well within a CPU-only environment's memory.

Equations
---------

A. Cross-class k-NN (train_original.csv only, `find_cross_class_neighbors`):
for every sample i, among all OTHER samples j with y_j != y_i, take the k
with the highest cosine similarity to i.

B. Sample-level neighborhood ambiguity (train, `compute_sample_neighborhood_ambiguity`):
from the k cross-class neighbors of each train sample - nearest competing
class (class of the single most-similar cross-class neighbor), mean
top-k cross-class similarity, per-class neighbor fraction, strongest
competing class (mode of the k neighbors' classes) and its fraction/mean
similarity.

C. Class-level neighborhood ambiguity matrix (train, `compute_neighborhood_class_matrix`):

    For directed a -> b:
        score(a->b) = mean over samples i with y_i=a of
                      [ sum over i's top-k cross-class neighbors j with y_j=b
                        of max(0, cosine(z_i, z_j)) / k ]

    (bounded in [0, 1] by construction: at most k terms, each in [0, 1/k])

        A_sym[a,b] = (score(a->b) + score(b->a)) / 2      for a != b
        A_sym[a,a] = 0

    Then min-max rescaled using ONLY the off-diagonal entries of A_sym
    itself (entirely train-derived - no validation or test data enters
    this step) so the full [0, 1] range is used for interpretability:

        A[a,b] = (A_sym[a,b] - m) / (M - m)     where m, M = min, max over a != b
        A[a,a] = 0

D. Validation sample ambiguity (`compute_validation_neighborhood_ambiguity`):
for each validation embedding, the k overall nearest TRAIN embeddings
(NOT restricted to a different class - unlike A above, a validation
sample deep inside its own class's neighborhood should correctly show
low ambiguity, which requires letting same-class train neighbors count).
Let p_c = fraction of the k neighbors with class c (a probability
distribution over classes, entirely determined by the training neighbor
pool). Two diagnostics, both naturally bounded in [0, 1] as probabilities/
proportions - by construction, neither requires any additional train-
fitted normalization constant:

    margin  = p_(1) - p_(2)              (top-1 minus top-2 class proportions)
    ambiguity_margin  = 1 - margin        in [0, 1]     <- PRIMARY score
    H       = -sum_c p_c * log(p_c)
    ambiguity_entropy = H / log(K)        in [0, 1]     <- secondary diagnostic

`ambiguity_margin` is chosen as the primary score for direct consistency
with Phase 1's own choice of a top-1-vs-top-2 margin as its primary score
(see src/losses/ambiguity.py) - the two are computed identically in spirit
(a rank-1-vs-rank-2 gap) even though rank here comes from neighbor-class
proportions rather than prototype cosine similarity, which is exactly
what makes the two phases' scores comparable. `strongest_class`/
`competing_class` are rank-1/rank-2 of p_c, with ties broken toward the
lower class index (a stable sort, fully deterministic) - identical in
spirit to Phase 1's competing-class rule (computed purely from embedding
geometry, never from a true or predicted label).
"""

import math
from dataclasses import dataclass

import numpy as np

DEFAULT_K = 10


class NeighborhoodAmbiguityError(Exception):
    """Raised for a neighborhood-ambiguity-specific problem: a malformed
    embeddings/labels array, an invalid k, or too few cross-class
    candidates to find k neighbors for some sample (never silently
    returning fewer than k neighbors)."""


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def _validate_embeddings_labels(embeddings: np.ndarray, labels: np.ndarray) -> int:
    if embeddings.ndim != 2:
        raise NeighborhoodAmbiguityError(f"embeddings must be 2-D [N, D], got shape {embeddings.shape}")
    labels = np.asarray(labels)
    if labels.ndim != 1 or labels.shape[0] != embeddings.shape[0]:
        raise NeighborhoodAmbiguityError(
            f"labels must be 1-D with length N={embeddings.shape[0]}, got shape {labels.shape}"
        )
    return embeddings.shape[0]


@dataclass
class CrossClassNeighbors:
    """Per-sample top-k cross-class neighbors, for train_original.csv
    embeddings only. All arrays are [N, k]; `neighbor_indices` indexes
    back into the SAME `embeddings`/`labels` arrays passed to
    `find_cross_class_neighbors`."""

    neighbor_indices: np.ndarray
    neighbor_similarities: np.ndarray
    neighbor_classes: np.ndarray
    k: int


def find_cross_class_neighbors(embeddings: np.ndarray, labels: np.ndarray, k: int = DEFAULT_K) -> CrossClassNeighbors:
    """For every sample i, find the k samples j (j != i, y_j != y_i) with
    the highest cosine similarity to i. Never includes i itself or any
    same-class sample. Raises NeighborhoodAmbiguityError if k < 1 or if
    any sample has fewer than k eligible (different-class) candidates in
    the whole array - a silently-truncated neighbor list would make every
    downstream mean/fraction computation quietly inconsistent across
    samples, so this fails loudly instead."""
    n = _validate_embeddings_labels(embeddings, labels)
    if k < 1:
        raise NeighborhoodAmbiguityError(f"k must be >= 1, got {k}")
    labels = np.asarray(labels)

    unit = _l2_normalize_rows(embeddings.astype(np.float64))
    similarity = unit @ unit.T  # [N, N] - never an [N, N, D] tensor

    same_class = labels[:, None] == labels[None, :]
    self_mask = np.eye(n, dtype=bool)
    excluded = same_class | self_mask

    num_eligible = (~excluded).sum(axis=1)
    insufficient = np.where(num_eligible < k)[0]
    if insufficient.size > 0:
        raise NeighborhoodAmbiguityError(
            f"k={k} cross-class neighbors requested, but sample index {int(insufficient[0])} has only "
            f"{int(num_eligible[insufficient[0]])} eligible (different-class) candidates available"
        )

    masked_similarity = np.where(excluded, -np.inf, similarity)
    # Stable sort (descending) so ties break deterministically toward the
    # lower original index, matching this project's existing convention
    # (src/losses/ambiguity.py's _top2_similarity uses the same approach).
    order = np.argsort(-masked_similarity, axis=1, kind="stable")[:, :k]
    rows = np.arange(n)[:, None]

    neighbor_similarities = similarity[rows, order]
    neighbor_classes = labels[order]

    return CrossClassNeighbors(neighbor_indices=order, neighbor_similarities=neighbor_similarities, neighbor_classes=neighbor_classes, k=k)


@dataclass
class SampleNeighborhoodAmbiguityResult:
    """Per-train-sample outputs (see module docstring, section B)."""

    nearest_competing_class: np.ndarray  # [N] int - class of the single closest cross-class neighbor
    mean_topk_similarity: np.ndarray  # [N] float - mean raw cosine similarity over all k neighbors
    class_fraction: np.ndarray  # [N, K] float - fraction of the k neighbors belonging to each class
    strongest_competing_class: np.ndarray  # [N] int - mode class among the k neighbors
    strongest_competing_fraction: np.ndarray  # [N] float, in [0, 1]
    strongest_competing_mean_similarity: np.ndarray  # [N] float - mean similarity restricted to that class's neighbors


def compute_sample_neighborhood_ambiguity(neighbors: CrossClassNeighbors, num_classes: int) -> SampleNeighborhoodAmbiguityResult:
    """Derive the per-sample neighborhood-ambiguity fields from an
    already-computed `CrossClassNeighbors` (see `find_cross_class_neighbors`)."""
    if num_classes < 2:
        raise NeighborhoodAmbiguityError(f"num_classes must be >= 2, got {num_classes}")
    n, k = neighbors.neighbor_classes.shape

    nearest_competing_class = neighbors.neighbor_classes[:, 0].copy()
    mean_topk_similarity = neighbors.neighbor_similarities.mean(axis=1)

    class_fraction = np.zeros((n, num_classes), dtype=np.float64)
    for c in range(num_classes):
        class_fraction[:, c] = (neighbors.neighbor_classes == c).sum(axis=1) / k

    # Stable descending sort of class_fraction -> ties broken toward the
    # lower class index, deterministic.
    order = np.argsort(-class_fraction, axis=1, kind="stable")
    strongest_competing_class = order[:, 0]
    rows = np.arange(n)
    strongest_competing_fraction = class_fraction[rows, strongest_competing_class]

    strongest_competing_mean_similarity = np.zeros(n, dtype=np.float64)
    for i in range(n):
        mask = neighbors.neighbor_classes[i] == strongest_competing_class[i]
        strongest_competing_mean_similarity[i] = neighbors.neighbor_similarities[i, mask].mean()

    return SampleNeighborhoodAmbiguityResult(
        nearest_competing_class=nearest_competing_class,
        mean_topk_similarity=mean_topk_similarity,
        class_fraction=class_fraction,
        strongest_competing_class=strongest_competing_class,
        strongest_competing_fraction=strongest_competing_fraction,
        strongest_competing_mean_similarity=strongest_competing_mean_similarity,
    )


def compute_neighborhood_class_matrix(neighbors: CrossClassNeighbors, labels: np.ndarray, num_classes: int) -> np.ndarray:
    """A[a,b] (see module docstring, section C) - symmetric, bounded
    [0, 1], zero diagonal, deterministic given fixed neighbors/labels.
    Rescaling uses only the matrix's own off-diagonal entries (train-
    derived; no validation or test data enters this computation)."""
    if num_classes < 2:
        raise NeighborhoodAmbiguityError(f"num_classes must be >= 2, got {num_classes}")
    labels = np.asarray(labels)
    n, k = neighbors.neighbor_classes.shape

    rectified = np.clip(neighbors.neighbor_similarities, 0.0, None) / k  # [N, k]
    anchor_classes_flat = np.repeat(labels, k)
    neighbor_classes_flat = neighbors.neighbor_classes.reshape(-1)
    rectified_flat = rectified.reshape(-1)

    directed_sum = np.zeros((num_classes, num_classes), dtype=np.float64)
    np.add.at(directed_sum, (anchor_classes_flat, neighbor_classes_flat), rectified_flat)

    class_counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    missing = np.where(class_counts == 0)[0]
    if missing.size > 0:
        raise NeighborhoodAmbiguityError(f"zero samples for class index/es {missing.tolist()} - cannot compute directed scores")

    directed_mean = directed_sum / class_counts[:, None]  # score(a -> b)

    symmetric = (directed_mean + directed_mean.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)

    off_diagonal_mask = ~np.eye(num_classes, dtype=bool)
    off_diagonal_values = symmetric[off_diagonal_mask]
    m, big_m = float(off_diagonal_values.min()), float(off_diagonal_values.max())

    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    if big_m > m:
        matrix[off_diagonal_mask] = (symmetric[off_diagonal_mask] - m) / (big_m - m)
    # else: every off-diagonal entry is already identical (a degenerate,
    # uninformative case) - leave the matrix at all-zero rather than
    # dividing by zero, consistent with this project's existing
    # degenerate-normalization convention (src/losses/ambiguity.py's
    # `_apply_margin_normalization`, src/evaluation/ood_uncertainty.py's
    # `_apply_minmax`).
    return matrix


@dataclass
class ValidationNeighborhoodAmbiguityResult:
    """Per-validation-sample outputs (see module docstring, section D).
    Computed purely from each sample's embedding geometry against TRAIN
    embeddings only - never from a true or predicted label."""

    strongest_class: np.ndarray  # [N] int - mode class among the k nearest TRAIN neighbors (any class, not cross-class-only)
    competing_class: np.ndarray  # [N] int - 2nd-most-frequent class among the k neighbors
    class_fraction: np.ndarray  # [N, K] float
    margin: np.ndarray  # [N] float, in [0, 1] - p_(1) - p_(2)
    ambiguity_margin: np.ndarray  # [N] float, in [0, 1] - PRIMARY score - 1 - margin
    entropy_nats: np.ndarray  # [N] float
    ambiguity_entropy: np.ndarray  # [N] float, in [0, 1] - secondary diagnostic


def compute_validation_neighborhood_ambiguity(
    val_embeddings: np.ndarray, train_embeddings: np.ndarray, train_labels: np.ndarray, num_classes: int, k: int = DEFAULT_K
) -> ValidationNeighborhoodAmbiguityResult:
    """For each validation embedding, find the k overall nearest TRAIN
    embeddings (same-class train neighbors ARE allowed here, unlike
    `find_cross_class_neighbors` - see module docstring section D for
    why), and compute the neighbor-class distribution + both ambiguity
    diagnostics from it. `train_embeddings`/`train_labels` must be
    train_original.csv only; validation samples are never compared
    against other validation samples."""
    if val_embeddings.ndim != 2:
        raise NeighborhoodAmbiguityError(f"val_embeddings must be 2-D [N, D], got shape {val_embeddings.shape}")
    if train_embeddings.ndim != 2:
        raise NeighborhoodAmbiguityError(f"train_embeddings must be 2-D [M, D], got shape {train_embeddings.shape}")
    if val_embeddings.shape[1] != train_embeddings.shape[1]:
        raise NeighborhoodAmbiguityError(
            f"val embedding dim {val_embeddings.shape[1]} != train embedding dim {train_embeddings.shape[1]}"
        )
    train_labels = np.asarray(train_labels)
    if train_labels.shape[0] != train_embeddings.shape[0]:
        raise NeighborhoodAmbiguityError("train_labels length must match train_embeddings")
    if num_classes < 2:
        raise NeighborhoodAmbiguityError(f"num_classes must be >= 2, got {num_classes}")
    num_train = train_embeddings.shape[0]
    if k < 1 or k > num_train:
        raise NeighborhoodAmbiguityError(f"k must be in [1, {num_train}] (number of train samples), got {k}")

    val_unit = _l2_normalize_rows(val_embeddings.astype(np.float64))
    train_unit = _l2_normalize_rows(train_embeddings.astype(np.float64))
    similarity = val_unit @ train_unit.T  # [Nval, Ntrain] - never an [Nval, Ntrain, D] tensor

    order = np.argsort(-similarity, axis=1, kind="stable")[:, :k]
    neighbor_classes = train_labels[order]  # [Nval, k]

    n_val = val_embeddings.shape[0]
    class_fraction = np.zeros((n_val, num_classes), dtype=np.float64)
    for c in range(num_classes):
        class_fraction[:, c] = (neighbor_classes == c).sum(axis=1) / k

    rank_order = np.argsort(-class_fraction, axis=1, kind="stable")
    strongest_class = rank_order[:, 0]
    competing_class = rank_order[:, 1]
    rows = np.arange(n_val)
    p1 = class_fraction[rows, strongest_class]
    p2 = class_fraction[rows, competing_class]
    margin = p1 - p2
    ambiguity_margin = 1.0 - margin

    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(class_fraction > 0, np.log(class_fraction), 0.0)
    entropy_nats = -(class_fraction * log_p).sum(axis=1)
    ambiguity_entropy = entropy_nats / math.log(num_classes)

    return ValidationNeighborhoodAmbiguityResult(
        strongest_class=strongest_class,
        competing_class=competing_class,
        class_fraction=class_fraction,
        margin=margin,
        ambiguity_margin=ambiguity_margin,
        entropy_nats=entropy_nats,
        ambiguity_entropy=ambiguity_entropy,
    )

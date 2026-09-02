"""Phase 3: continuous class-affinity ambiguity in the frozen AA-EvidentNet
embedding space - a third, additional, research-only ambiguity direction
alongside Phase 1's class-PROTOTYPE ambiguity (src/losses/ambiguity.py)
and Phase 2's cross-class NEIGHBORHOOD ambiguity
(src/losses/neighborhood_ambiguity.py), both of which remain completely
unmodified and intact.

Motivation (see REPRODUCIBILITY.md for the full discussion): Phase 1
(one class centroid per class) and Phase 2 (discrete top-k neighbor
votes) both struggled to strongly flag the single largest real validation
confusion (Healthy <-> Glaucoma, 35 confusions), and Phase 2's own
per-sample ambiguity became extremely sparse (median 0.0 for both correct
and incorrect predictions). Phase 3 tests a CONTINUOUS, per-class
affinity in between those two extremes: for every class, the mean
similarity to that class's m closest training samples (m=5, fixed, not
tuned) - richer than a single centroid, denser than a discrete neighbor
vote. Whether this is actually more informative is exactly what the
validation analysis (src/evaluation/class_affinity_ambiguity_validation.py)
measures - this module makes no claim about the answer.

This module is PURE MATH ONLY - no I/O, no checkpoints, no dataloaders -
matching every other loss module's purity convention. Orchestration lives
in src/training/class_affinity_ambiguity_setup.py (train-only, builds the
frozen quantities) and
src/evaluation/class_affinity_ambiguity_validation.py (val-only analysis).

THIS PHASE IS ANALYSIS/RESEARCH ONLY: nothing here is wired into
CSSupConLoss or any training objective. No performance or novelty claim
is made anywhere.

Equations
---------

A. Class affinity (`compute_class_affinities`): for query embedding z_i
and class c,

    a_i,c = mean(top-m cosine similarities between z_i and every
                 TRAIN embedding of class c)

using min(m, available candidates) when a class has fewer than m
eligible candidates (defensive - the real 10-class dataset has well over
m=5 original samples in every class; this only matters for tiny test
fixtures). When computing this for a TRAIN sample against its OWN class
(`exclude_self=True`), the sample's own entry is excluded first (a
sample's cosine similarity with itself is always exactly 1.0, which would
otherwise trivially and meaninglessly dominate its own class's affinity).

B. Primary sample ambiguity (`compute_top_affinities` +
`fit_margin_scale` + `compute_primary_ambiguity`): sort the K class
affinities descending; a1/a2 = top two values (top1_class/top2_class =
their classes).

    margin_i     = a1 - a2                                   raw margin
    ambiguity_i  = 1 - clip(margin_i / margin_scale, 0, 1)    in [0, 1], PRIMARY score

`margin_scale` is the 95th percentile of TRAIN samples' own (self-
excluded) top1-top2 margins - fit ONCE from train_original.csv only,
never from validation or test. Raises ClassAffinityAmbiguityError if
`margin_scale` is numerically zero (a degenerate training population
would otherwise silently divide by ~0).

C. Secondary diagnostics (`compute_entropy_diagnostic`): normalized
entropy of `softmax(affinities / temperature)`, temperature=0.1 fixed
(not tuned) - reported alongside the primary margin score, never
replacing it, exactly like Phase 1's/Phase 2's own entropy diagnostics.

D. Label-aware boundary score (ANALYSIS ONLY - requires the true label,
so it is NEVER a candidate inference-time score):

    own_affinity        = a_i,y                              (affinity to the TRUE class y)
    best_other_affinity = max_{c != y} a_i,c
    boundary_gap_i      = own_affinity - best_other_affinity

    label_aware_ambiguity_i = 1 - clip(boundary_gap_i / boundary_gap_scale, 0, 1)

deliberately the SAME simple train-scaled transform as the primary score
(no new trainable function introduced) - `boundary_gap_scale` is the 95th
percentile of TRAIN samples' own (self-excluded, using each train
sample's own true label) boundary gaps, fit once from train_original.csv
only.

E. Class-level affinity matrix (`compute_class_affinity_matrix`): for
train samples in class a, the mean of their (self-excluded) affinity to
class b is a directed score; symmetrized and rescaled exactly like Phase
1's/Phase 2's own matrices:

    directed(a->b) = mean over train samples i in class a of a_i,b
    A_sym[a,b]     = (directed(a->b) + directed(b->a)) / 2      for a != b
    A_sym[a,a]     = 0
    A[a,b]         = min-max rescale of A_sym's OWN off-diagonal entries to [0,1]

Never built from a confusion matrix or validation data - the exact same
methodological discipline as Phase 1/Phase 2.
"""

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

DEFAULT_M = 5
DEFAULT_TEMPERATURE = 0.1
DEFAULT_SCALE_PERCENTILE = 95.0


class ClassAffinityAmbiguityError(Exception):
    """Raised for a class-affinity-ambiguity-specific problem: a malformed
    embeddings/labels array, a class with zero eligible candidates, or a
    numerically-zero train-derived scale (which would otherwise silently
    divide by ~0 rather than failing loudly)."""


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def compute_class_affinities(
    query_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    reference_labels: np.ndarray,
    num_classes: int,
    m: int = DEFAULT_M,
    exclude_self: bool = False,
) -> np.ndarray:
    """a_i,c = mean of the top-min(m, available) cosine similarities
    between query row i and `reference_embeddings` rows labeled c.

    `exclude_self=True` means `query_embeddings` IS `reference_embeddings`
    (row-aligned - query row i's own entry in the reference pool is row i)
    and each row's own entry is excluded before computing its OWN class's
    affinity - used only when deriving train-only scales from
    train_original.csv against itself. It must never be used for
    validation embeddings (which are never present in the reference pool
    to begin with, so no exclusion is ever needed there).
    """
    if query_embeddings.ndim != 2:
        raise ClassAffinityAmbiguityError(f"query_embeddings must be 2-D [Q, D], got shape {query_embeddings.shape}")
    if reference_embeddings.ndim != 2:
        raise ClassAffinityAmbiguityError(f"reference_embeddings must be 2-D [N, D], got shape {reference_embeddings.shape}")
    if query_embeddings.shape[1] != reference_embeddings.shape[1]:
        raise ClassAffinityAmbiguityError(
            f"query embedding dim {query_embeddings.shape[1]} != reference embedding dim {reference_embeddings.shape[1]}"
        )
    reference_labels = np.asarray(reference_labels)
    if reference_labels.shape[0] != reference_embeddings.shape[0]:
        raise ClassAffinityAmbiguityError("reference_labels length must match reference_embeddings")
    if num_classes < 2:
        raise ClassAffinityAmbiguityError(f"num_classes must be >= 2, got {num_classes}")
    if m < 1:
        raise ClassAffinityAmbiguityError(f"m must be >= 1, got {m}")
    if exclude_self and query_embeddings.shape[0] != reference_embeddings.shape[0]:
        raise ClassAffinityAmbiguityError("exclude_self=True requires query_embeddings and reference_embeddings to be the same set (row-aligned)")

    query_unit = _l2_normalize_rows(query_embeddings.astype(np.float64))
    reference_unit = _l2_normalize_rows(reference_embeddings.astype(np.float64))
    similarity = query_unit @ reference_unit.T  # [Q, N] - never a [Q, N, D] tensor

    if exclude_self:
        np.fill_diagonal(similarity, -np.inf)

    num_query = similarity.shape[0]
    affinities = np.zeros((num_query, num_classes), dtype=np.float64)

    for c in range(num_classes):
        class_mask = reference_labels == c
        count_c = int(class_mask.sum())
        if count_c == 0:
            raise ClassAffinityAmbiguityError(f"reference set has zero samples for class index {c} - affinity is undefined")

        class_similarity = similarity[:, class_mask]  # [Q, count_c]
        sorted_desc = -np.sort(-class_similarity, axis=1)  # [Q, count_c], descending

        if exclude_self:
            own_class_rows = reference_labels == c  # query row i (== reference row i) whose own class is c
            count_per_row = np.where(own_class_rows, count_c - 1, count_c)
        else:
            count_per_row = np.full(num_query, count_c)

        if np.any(count_per_row <= 0):
            raise ClassAffinityAmbiguityError(
                f"class index {c} has too few samples ({count_c}) to exclude a sample's own entry and still "
                "have at least one eligible candidate"
            )

        take_per_row = np.minimum(m, count_per_row)
        positions = np.arange(count_c)[None, :]
        valid = positions < take_per_row[:, None]
        masked_vals = np.where(valid, sorted_desc, 0.0)
        affinities[:, c] = masked_vals.sum(axis=1) / take_per_row

    return affinities


@dataclass
class TopAffinities:
    top1_affinity: np.ndarray  # [Q]
    top1_class: np.ndarray  # [Q] int
    top2_affinity: np.ndarray  # [Q]
    top2_class: np.ndarray  # [Q] int
    raw_margin: np.ndarray  # [Q], = top1_affinity - top2_affinity


def compute_top_affinities(affinities: np.ndarray) -> TopAffinities:
    """Sort each row's class affinities descending; top1/top2 are rank-1/
    rank-2 (ties broken toward the lower class index via a stable sort -
    deterministic, and computed purely from affinity geometry, never from
    a true or predicted label)."""
    if affinities.ndim != 2 or affinities.shape[1] < 2:
        raise ClassAffinityAmbiguityError(f"affinities must be 2-D [Q, K] with K >= 2, got shape {affinities.shape}")
    order = np.argsort(-affinities, axis=1, kind="stable")
    rows = np.arange(affinities.shape[0])
    top1_class = order[:, 0]
    top2_class = order[:, 1]
    top1_affinity = affinities[rows, top1_class]
    top2_affinity = affinities[rows, top2_class]
    return TopAffinities(
        top1_affinity=top1_affinity,
        top1_class=top1_class,
        top2_affinity=top2_affinity,
        top2_class=top2_class,
        raw_margin=top1_affinity - top2_affinity,
    )


def fit_margin_scale(train_margins: np.ndarray, percentile: float = DEFAULT_SCALE_PERCENTILE) -> float:
    """The `percentile`-th percentile (95th by default) of TRAIN samples'
    own top1-top2 affinity margins - MUST be called with train_original.csv
    margins only (never validation or test). Raises
    ClassAffinityAmbiguityError if the result is numerically zero (would
    otherwise silently divide by ~0 downstream)."""
    scale = float(np.percentile(train_margins, percentile))
    if scale <= 0.0:
        raise ClassAffinityAmbiguityError(
            f"train-derived margin_scale is numerically zero or negative ({scale}) - refusing to divide by it"
        )
    return scale


def compute_primary_ambiguity(margin: np.ndarray, scale: float) -> np.ndarray:
    """ambiguity = 1 - clip(margin / scale, 0, 1), in [0, 1]. `scale` must
    already be train-derived (see `fit_margin_scale`) - this function does
    not fit it itself, so the identical frozen scale can be reused across
    both validation and (for reporting/analysis) train margins."""
    if scale <= 0.0:
        raise ClassAffinityAmbiguityError(f"scale must be > 0, got {scale}")
    normalized = np.clip(np.asarray(margin, dtype=np.float64) / scale, 0.0, 1.0)
    return 1.0 - normalized


def compute_entropy_diagnostic(affinities: np.ndarray, temperature: float = DEFAULT_TEMPERATURE) -> np.ndarray:
    """Normalized entropy of softmax(affinities / temperature), in [0, 1]
    - a SECONDARY diagnostic (see module docstring section C), never the
    primary ambiguity score."""
    if temperature <= 0:
        raise ClassAffinityAmbiguityError(f"temperature must be > 0, got {temperature}")
    num_classes = affinities.shape[1]
    scaled = affinities / temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # log-sum-exp stability
    exp_scaled = np.exp(scaled)
    probabilities = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probabilities > 0, np.log(probabilities), 0.0)
    entropy_nats = -(probabilities * log_p).sum(axis=1)
    return entropy_nats / math.log(num_classes)


def compute_label_aware_boundary_gap(affinities: np.ndarray, true_labels: np.ndarray) -> np.ndarray:
    """boundary_gap_i = a_i,y - max_{c != y} a_i,c (see module docstring
    section D). ANALYSIS ONLY - requires the true label `true_labels`, so
    this quantity (and `compute_label_aware_ambiguity` built from it) must
    NEVER be proposed as an inference-time uncertainty/ambiguity score."""
    true_labels = np.asarray(true_labels)
    if true_labels.shape[0] != affinities.shape[0]:
        raise ClassAffinityAmbiguityError("true_labels length must match affinities' first dimension")
    num_query = affinities.shape[0]
    rows = np.arange(num_query)
    own_affinity = affinities[rows, true_labels]
    masked = affinities.copy()
    masked[rows, true_labels] = -np.inf
    best_other_affinity = masked.max(axis=1)
    return own_affinity - best_other_affinity


def fit_boundary_gap_scale(train_boundary_gaps: np.ndarray, percentile: float = DEFAULT_SCALE_PERCENTILE) -> float:
    """The `percentile`-th percentile of TRAIN samples' own boundary gaps
    - fit once from train_original.csv only. Raises
    ClassAffinityAmbiguityError if numerically zero or negative."""
    scale = float(np.percentile(train_boundary_gaps, percentile))
    if scale <= 0.0:
        raise ClassAffinityAmbiguityError(
            f"train-derived boundary_gap_scale is numerically zero or negative ({scale}) - refusing to divide by it"
        )
    return scale


def compute_label_aware_ambiguity(boundary_gap: np.ndarray, scale: float) -> np.ndarray:
    """label_aware_ambiguity = 1 - clip(boundary_gap / scale, 0, 1), in
    [0, 1] - the SAME simple train-scaled transform as
    `compute_primary_ambiguity`, deliberately, rather than a new
    trainable function. ANALYSIS ONLY (see module docstring section D) -
    never an inference-time score, since it requires the true label."""
    if scale <= 0.0:
        raise ClassAffinityAmbiguityError(f"scale must be > 0, got {scale}")
    normalized = np.clip(np.asarray(boundary_gap, dtype=np.float64) / scale, 0.0, 1.0)
    return 1.0 - normalized


def compute_class_affinity_matrix(train_affinities_self_excluded: np.ndarray, train_labels: np.ndarray, num_classes: int) -> np.ndarray:
    """A[a,b] (see module docstring section E) - symmetric, bounded
    [0, 1], zero diagonal, deterministic given fixed
    `train_affinities_self_excluded` (the [N, K] output of
    `compute_class_affinities(..., exclude_self=True)` over
    train_original.csv). Rescaling uses only the matrix's own off-diagonal
    entries (train-derived; no validation or test data enters this
    computation)."""
    if num_classes < 2:
        raise ClassAffinityAmbiguityError(f"num_classes must be >= 2, got {num_classes}")
    train_labels = np.asarray(train_labels)
    if train_labels.shape[0] != train_affinities_self_excluded.shape[0]:
        raise ClassAffinityAmbiguityError("train_labels length must match train_affinities_self_excluded")

    directed = np.zeros((num_classes, num_classes), dtype=np.float64)
    for a in range(num_classes):
        class_mask = train_labels == a
        if not np.any(class_mask):
            raise ClassAffinityAmbiguityError(f"zero train samples for class index {a} - cannot compute directed scores")
        directed[a, :] = train_affinities_self_excluded[class_mask].mean(axis=0)

    symmetric = (directed + directed.T) / 2.0
    np.fill_diagonal(symmetric, 0.0)

    off_diagonal_mask = ~np.eye(num_classes, dtype=bool)
    off_diagonal_values = symmetric[off_diagonal_mask]
    low, high = float(off_diagonal_values.min()), float(off_diagonal_values.max())

    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    if high > low:
        matrix[off_diagonal_mask] = (symmetric[off_diagonal_mask] - low) / (high - low)
    # else: degenerate (every off-diagonal entry identical) - leave at
    # all-zero rather than dividing by zero, matching Phase 1's/Phase 2's
    # own degenerate-normalization convention.
    return matrix


@dataclass
class SampleClassAffinityResult:
    """Per-sample outputs - see module docstring sections A-D. Populated
    fully for validation samples (including the ANALYSIS-ONLY label-aware
    fields, since validation labels are available for analysis); train
    samples only need `affinities`/`TopAffinities` to derive the two
    train-only scales (see src/training/class_affinity_ambiguity_setup.py)."""

    affinities: np.ndarray  # [Q, K]
    top1_affinity: np.ndarray
    top1_class: np.ndarray
    top2_affinity: np.ndarray
    top2_class: np.ndarray
    raw_margin: np.ndarray
    ambiguity: np.ndarray  # PRIMARY score, in [0, 1]
    entropy: np.ndarray  # secondary diagnostic, in [0, 1]
    boundary_gap: Optional[np.ndarray] = None  # ANALYSIS ONLY - requires true labels
    label_aware_ambiguity: Optional[np.ndarray] = None  # ANALYSIS ONLY - requires true labels


def compute_sample_class_affinity_result(
    affinities: np.ndarray,
    margin_scale: float,
    temperature: float = DEFAULT_TEMPERATURE,
    true_labels: Optional[np.ndarray] = None,
    boundary_gap_scale: Optional[float] = None,
) -> SampleClassAffinityResult:
    """Bundle the full per-sample result from an already-computed
    affinity matrix (see `compute_class_affinities`). `true_labels`/
    `boundary_gap_scale` are optional - when both are given, the ANALYSIS-
    ONLY label-aware fields are also populated; when either is omitted,
    they are left as `None` (e.g. for train-sample-only scale-fitting
    calls that don't need the full bundle)."""
    top = compute_top_affinities(affinities)
    ambiguity = compute_primary_ambiguity(top.raw_margin, margin_scale)
    entropy = compute_entropy_diagnostic(affinities, temperature)

    boundary_gap = None
    label_aware_ambiguity = None
    if true_labels is not None and boundary_gap_scale is not None:
        boundary_gap = compute_label_aware_boundary_gap(affinities, true_labels)
        label_aware_ambiguity = compute_label_aware_ambiguity(boundary_gap, boundary_gap_scale)

    return SampleClassAffinityResult(
        affinities=affinities,
        top1_affinity=top.top1_affinity,
        top1_class=top.top1_class,
        top2_affinity=top.top2_affinity,
        top2_class=top.top2_class,
        raw_margin=top.raw_margin,
        ambiguity=ambiguity,
        entropy=entropy,
        boundary_gap=boundary_gap,
        label_aware_ambiguity=label_aware_ambiguity,
    )

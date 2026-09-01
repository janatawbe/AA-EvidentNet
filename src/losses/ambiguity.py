"""Learned class-level ambiguity + sample-level ambiguity (analysis-only,
Phase 1) - explicit ambiguity modeling for AA-EvidentNet, distinct from
EDL uncertainty and from OOD detection.

Three deliberately separate concepts (never combined into one score here):

  - CLASS ambiguity: "which disease classes have overlapping learned
    representations, in general?" A static, symmetric K x K matrix,
    computed once from class prototypes and then frozen.
  - SAMPLE ambiguity: "which competing disease classes does THIS
    particular image resemble?" A per-image scalar (+ competing class +
    similarity vector), computed independently of the class matrix's
    values (both are derived from the same prototypes, but a sample's
    score depends only on its own embedding's geometry, never on which
    class pairs happen to be flagged class-ambiguous).
  - EDL uncertainty (src/losses/evidential.py) and OOD (src/evaluation/
    ood_uncertainty.py) are untouched by this module and are not computed
    here at all.

This module is pure math only - no I/O, no checkpoint loading, no
dataloaders (matching src/losses/cs_supcon.py and src/losses/evidential.py's
existing convention). Loading a reference checkpoint and building a
train_original.csv dataloader is orchestration, done by
src/training/ambiguity_setup.py, which calls
src.models.prototypes.compute_class_prototypes and then the functions
below.

Equations
---------

Class-level ambiguity matrix (given prototypes P_1..P_K, each the mean
fused embedding of class k over train_original.csv, from a FROZEN,
already-fully-trained reference representation - see
src/training/ambiguity_setup.py and REPRODUCIBILITY.md for exactly which
checkpoint and why):

    S(a,b) = (P_a . P_b) / (||P_a|| ||P_b||)      cosine similarity, [-1, 1]
    A[a,b] = max(0, S(a,b))     for a != b         rectified, [0, 1]
    A[a,a] = 0                                      diagonal, explicit, never queried

Symmetric by construction, bounded, deterministic given fixed prototypes.
Computed purely from embedding geometry - NEVER from a confusion matrix,
predictions, or errors, which is what keeps it conceptually distinct from
uncertainty (a confusion-matrix-based measure would conflate "these
classes look alike" with "the model gets these wrong," which is a
different question).

Sample-level ambiguity (Phase 1: ANALYSIS ONLY - see module docstring of
src/training/ambiguity_setup.py and REPRODUCIBILITY.md for why this does
not yet influence any loss):

    sim_i,k       = cosine_similarity(z_i, P_k)         for k = 1..K
    raw_margin_i  = sim_i,(1) - sim_i,(2)                 top-2 gap, [0, 2]
    ambiguity_i   = 1 - clip((raw_margin_i - margin_min) / (margin_max - margin_min), 0, 1)

`margin_min`/`margin_max` are fit once from train_original.csv's own
realized raw_margin distribution (never from validation or test), exactly
mirroring src/evaluation/ood_uncertainty.py's established min-max
normalization convention. Floored at 0 but NOT capped above 1, so a
sample whose margin falls below the training-observed minimum reads as
maximally ambiguous rather than clipped to an arbitrary boundary.

`competing_class_i = argmax_{k != top1} sim_i,k` - the class achieving the
second-highest RAW cosine similarity. This identity is temperature-
independent (softmax is a rank-preserving monotonic transform), so it is
computed directly from raw similarities, never from a softmax output.

A secondary, OPTIONAL diagnostic (never the primary scalar, reported
alongside it): the full competing-class distribution
`q_i = softmax_k(sim_i,k / temperature)` and its normalized entropy
`H_i = -sum_k q_i,k * log(q_i,k) / log(K)`, in [0, 1]. This exists because
the raw-margin score cannot distinguish "a sharp two-way tie" from "a
diffuse four-way confusion" (both can produce a similarly small raw
margin) - entropy is reported specifically to make that distinction
visible during validation analysis, not to replace the margin as the
primary score.
"""

import math
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import torch

VALID_AMBIGUITY_SOURCES = ("fixed_pairs", "learned_class")

DEFAULT_AMBIGUITY_SCALE = 1.0


class AmbiguityConfigError(Exception):
    """Raised for an invalid learned-ambiguity configuration: an unknown
    `ambiguity_source`, a non-positive `ambiguity_scale`, or (deliberately)
    the recognized-but-not-yet-implemented `learned_class_sample` value.
    Never silently ignored or defaulted."""


class AmbiguityComputationError(Exception):
    """Raised for an ambiguity-computation-specific problem: a malformed
    prototypes/embeddings array, or a request for a quantity that is
    undefined given the inputs (e.g. fewer than 2 classes for a
    competing-class computation)."""


@dataclass(frozen=True)
class AmbiguitySettings:
    ambiguity_source: str
    ambiguity_scale: float


def load_ambiguity_settings(cs_supcon_section: Dict[str, Any]) -> AmbiguitySettings:
    """Parse+validate the `ambiguity_source`/`ambiguity_scale` fields of
    configs/losses.yaml: cs_supcon. `learned_class_sample` is recognized
    (not treated as an unknown-value typo) but explicitly rejected as not
    yet implemented in this phase - see module docstring - rather than
    silently falling back to a different mode."""
    ambiguity_source = cs_supcon_section.get("ambiguity_source", "fixed_pairs") or "fixed_pairs"
    ambiguity_scale = float(cs_supcon_section.get("ambiguity_scale", DEFAULT_AMBIGUITY_SCALE))

    if ambiguity_source == "learned_class_sample":
        raise AmbiguityConfigError(
            "cs_supcon.ambiguity_source='learned_class_sample' is not implemented in this phase. "
            "Sample-level ambiguity is analysis-only (src/evaluation/ambiguity_validation.py) and does not "
            "yet influence the training loss. Use 'fixed_pairs' or 'learned_class' instead."
        )
    if ambiguity_source not in VALID_AMBIGUITY_SOURCES:
        raise AmbiguityConfigError(
            f"cs_supcon.ambiguity_source='{ambiguity_source}' is not recognized. "
            f"Valid values: {VALID_AMBIGUITY_SOURCES}."
        )
    if ambiguity_source == "learned_class" and ambiguity_scale <= 0:
        raise AmbiguityConfigError(f"cs_supcon.ambiguity_scale must be > 0, got {ambiguity_scale}")

    return AmbiguitySettings(ambiguity_source=ambiguity_source, ambiguity_scale=ambiguity_scale)


def _l2_normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, eps, None)


def compute_class_ambiguity_matrix(prototypes: np.ndarray) -> np.ndarray:
    """A[a,b] = max(0, cosine_similarity(P_a, P_b)) for a != b, A[a,a] = 0.
    `prototypes`: [K, D] array (one row per class). Returns a [K, K]
    float64 array - symmetric, bounded [0, 1], deterministic given fixed
    prototypes."""
    if prototypes.ndim != 2:
        raise AmbiguityComputationError(f"prototypes must be 2-D [K, D], got shape {prototypes.shape}")
    num_classes = prototypes.shape[0]
    if num_classes == 0:
        raise AmbiguityComputationError("prototypes must have K > 0 classes")

    unit = _l2_normalize_rows(prototypes.astype(np.float64))
    similarity = unit @ unit.T  # [K, K], diagonal is exactly 1.0 (self-similarity)
    matrix = np.clip(similarity, 0.0, None)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def class_ambiguity_matrix_to_buffer(matrix: np.ndarray) -> torch.Tensor:
    """Convert a numpy class-ambiguity matrix to a non-trainable torch
    tensor (`requires_grad=False`) suitable for `register_buffer` on an
    `nn.Module` - moves with `.to(device)` automatically, is included in
    `state_dict()`, but is never updated by an optimizer or backward
    pass."""
    return torch.tensor(matrix, dtype=torch.float32, requires_grad=False)


@dataclass
class SampleAmbiguityResult:
    """Per-sample ambiguity outputs - analysis-only in this phase (see
    module docstring). `top_class`/`competing_class` are class INDICES
    (canonical ordering); `similarity` is the full [K] raw cosine-
    similarity vector to every prototype (the "prototype similarity
    vector" requested for reporting)."""

    top_class: np.ndarray  # [N] int
    competing_class: np.ndarray  # [N] int
    raw_margin: np.ndarray  # [N] float, in [0, 2]
    ambiguity: np.ndarray  # [N] float, in [0, 1]
    similarity: np.ndarray  # [N, K] float, raw cosine similarities
    entropy: np.ndarray  # [N] float, in [0, 1] - secondary diagnostic


@dataclass(frozen=True)
class MarginNormalization:
    margin_min: float
    margin_max: float


def fit_margin_normalization(raw_margins: np.ndarray) -> MarginNormalization:
    """Fit `margin_min`/`margin_max` from a distribution of raw top-2
    margins - MUST be called with train_original.csv's own raw margins
    only (never validation or test), so the resulting normalization
    reflects the same development-only representation the class matrix
    was built from."""
    return MarginNormalization(margin_min=float(np.min(raw_margins)), margin_max=float(np.max(raw_margins)))


def _apply_margin_normalization(raw_margins: np.ndarray, normalization: MarginNormalization) -> np.ndarray:
    """ambiguity = 1 - clip((raw_margin - min) / (max - min), 0, 1). A
    degenerate (zero-span) fitted range normalizes every value to
    ambiguity=0 (maximally UNambiguous, i.e. `raw_margin` reduces to a
    constant `_apply_minmax`-style edge case) rather than dividing by
    zero - consistent with how src/evaluation/ood_uncertainty.py's
    `_apply_minmax` already handles this."""
    span = normalization.margin_max - normalization.margin_min
    raw_margins = np.asarray(raw_margins, dtype=np.float64)
    if span <= 0:
        return np.zeros_like(raw_margins)
    normalized_margin = np.clip((raw_margins - normalization.margin_min) / span, 0.0, 1.0)
    return 1.0 - normalized_margin


def _validate_embeddings_and_prototypes(embeddings: np.ndarray, prototypes: np.ndarray) -> int:
    if embeddings.ndim != 2:
        raise AmbiguityComputationError(f"embeddings must be 2-D [N, D], got shape {embeddings.shape}")
    if prototypes.ndim != 2:
        raise AmbiguityComputationError(f"prototypes must be 2-D [K, D], got shape {prototypes.shape}")
    if embeddings.shape[1] != prototypes.shape[1]:
        raise AmbiguityComputationError(
            f"embedding dim {embeddings.shape[1]} != prototype dim {prototypes.shape[1]}"
        )
    num_classes = prototypes.shape[0]
    if num_classes < 2:
        raise AmbiguityComputationError(
            f"a competing class is undefined with K={num_classes} (need at least 2 classes)"
        )
    return num_classes


def _top2_similarity(embeddings: np.ndarray, prototypes: np.ndarray):
    """Shared core: raw [N, K] cosine similarities plus the top-1/top-2
    class indices and the raw margin between them. Used both by
    `compute_raw_margins` (fitting normalization, train_original.csv only)
    and by `compute_sample_ambiguity` (the full per-sample result), so the
    two never compute similarity with subtly different code paths."""
    _validate_embeddings_and_prototypes(embeddings, prototypes)
    embeddings_unit = _l2_normalize_rows(embeddings.astype(np.float64))
    prototypes_unit = _l2_normalize_rows(prototypes.astype(np.float64))
    similarity = embeddings_unit @ prototypes_unit.T  # [N, K]

    order = np.argsort(-similarity, axis=1, kind="stable")  # descending
    top_class = order[:, 0]
    competing_class = order[:, 1]
    rows = np.arange(similarity.shape[0])
    raw_margin = similarity[rows, top_class] - similarity[rows, competing_class]
    return similarity, top_class, competing_class, raw_margin


def compute_raw_margins(embeddings: np.ndarray, prototypes: np.ndarray) -> np.ndarray:
    """Raw top-2 cosine-similarity margins only (no normalization, no
    entropy) - used to FIT `MarginNormalization` from train_original.csv's
    own distribution (via `fit_margin_normalization`) before any
    ambiguity score is ever computed."""
    _, _, _, raw_margin = _top2_similarity(embeddings, prototypes)
    return raw_margin


def compute_sample_ambiguity(
    embeddings: np.ndarray,
    prototypes: np.ndarray,
    normalization: MarginNormalization,
    entropy_temperature: float = 0.1,
) -> SampleAmbiguityResult:
    """Per-sample ambiguity for every row of `embeddings` [N, D] against
    `prototypes` [K, D]. Requires K >= 2 (a "competing class" is undefined
    with fewer than 2 classes). `normalization` must have been fit on
    train_original.csv's own raw-margin distribution (see
    `fit_margin_normalization`) - this function does not fit it itself, so
    the same frozen normalization can be reused across many calls (e.g.
    once for train_original.csv itself, then again unchanged for
    val_original.csv during validation analysis) without ever re-fitting
    it on non-training data.

    `top_class`/`competing_class`/`raw_margin`/`ambiguity` depend ONLY on
    each sample's own embedding - never on a true or predicted label - so
    this identical computation applies unchanged whether or not labels are
    available (development-time analysis, or a future real inference use)."""
    num_classes = _validate_embeddings_and_prototypes(embeddings, prototypes)
    if entropy_temperature <= 0:
        raise AmbiguityComputationError(f"entropy_temperature must be > 0, got {entropy_temperature}")

    similarity, top_class, competing_class, raw_margin = _top2_similarity(embeddings, prototypes)
    ambiguity = _apply_margin_normalization(raw_margin, normalization)

    scaled = similarity / entropy_temperature
    scaled = scaled - scaled.max(axis=1, keepdims=True)  # log-sum-exp stability
    exp_scaled = np.exp(scaled)
    q = exp_scaled / exp_scaled.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_q = np.where(q > 0, np.log(q), 0.0)
    entropy_nats = -(q * log_q).sum(axis=1)
    entropy = entropy_nats / math.log(num_classes)

    return SampleAmbiguityResult(
        top_class=top_class,
        competing_class=competing_class,
        raw_margin=raw_margin,
        ambiguity=ambiguity,
        similarity=similarity,
        entropy=entropy,
    )

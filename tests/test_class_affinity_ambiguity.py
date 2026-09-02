"""Tests for src.losses.class_affinity_ambiguity (Phase 3: continuous
class-affinity ambiguity - pure math, no I/O). Phase 1's src.losses.ambiguity
and Phase 2's src.losses.neighborhood_ambiguity are untouched by this
module and are not re-tested here.
"""

import numpy as np
import pytest

from src.losses.class_affinity_ambiguity import (
    DEFAULT_M,
    DEFAULT_TEMPERATURE,
    ClassAffinityAmbiguityError,
    compute_class_affinities,
    compute_class_affinity_matrix,
    compute_entropy_diagnostic,
    compute_label_aware_ambiguity,
    compute_label_aware_boundary_gap,
    compute_primary_ambiguity,
    compute_sample_class_affinity_result,
    compute_top_affinities,
    fit_boundary_gap_scale,
    fit_margin_scale,
)

# 3 classes x 3 samples: classes 0/1 close together, class 2 far (near-opposite).
TOY_EMBEDDINGS = np.array(
    [
        [1.0, 0.0],  # 0 - class 0
        [0.95, 0.05],  # 1 - class 0
        [0.9, 0.1],  # 2 - class 0
        [0.0, 1.0],  # 3 - class 1
        [0.05, 0.95],  # 4 - class 1
        [0.1, 0.9],  # 5 - class 1
        [-1.0, 0.0],  # 6 - class 2
        [-0.95, -0.05],  # 7 - class 2
        [-0.9, -0.1],  # 8 - class 2
    ]
)
TOY_LABELS = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])


def test_default_m_and_temperature():
    assert DEFAULT_M == 5
    assert DEFAULT_TEMPERATURE == 0.1


# --- compute_class_affinities ---


def test_affinity_hand_computed_own_class_without_exclusion():
    # Query at class 0's exact centroid direction; affinity to class 0
    # using m=2 (top-2 of its 3 same-class neighbors, no exclusion).
    query = np.array([[1.0, 0.0]])
    affinities = compute_class_affinities(query, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2)
    assert affinities[0, 0] > affinities[0, 1] > affinities[0, 2]


def test_affinity_uses_fewer_than_m_when_class_has_fewer_candidates():
    query = np.array([[1.0, 0.0]])
    # m=10 but each class only has 3 samples -> must use all 3, not raise.
    affinities = compute_class_affinities(query, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=10)
    expected_class0 = (TOY_EMBEDDINGS[0:3] @ np.array([1.0, 0.0])).mean()
    # (already-unit-length column since query itself is unit) - compare via direct cosine mean
    unit_train = TOY_EMBEDDINGS[0:3] / np.linalg.norm(TOY_EMBEDDINGS[0:3], axis=1, keepdims=True)
    expected = unit_train.dot(np.array([1.0, 0.0])).mean()
    assert affinities[0, 0] == pytest.approx(expected, abs=1e-8)


def test_affinity_exclude_self_removes_own_entry():
    # Row 0 (class 0) affinity to its OWN class must exclude its perfect
    # self-similarity (1.0). With m=1, excluding self means class-0
    # affinity for row 0 is exactly its similarity to the single best
    # OTHER class-0 sample; without exclusion, it is exactly 1.0 (itself).
    affinities_excl = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=1, exclude_self=True)
    affinities_noexcl = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=1, exclude_self=False)
    assert affinities_noexcl[0, 0] == pytest.approx(1.0, abs=1e-8)  # top-1 (m=1) is itself
    assert affinities_excl[0, 0] < 1.0  # self excluded -> next-best same-class sample only
    assert affinities_noexcl[0, 0] > affinities_excl[0, 0]


def test_affinity_exclude_self_requires_matching_shapes():
    with pytest.raises(ClassAffinityAmbiguityError, match="same set"):
        compute_class_affinities(TOY_EMBEDDINGS[:5], TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2, exclude_self=True)


def test_affinity_rejects_mismatched_dims():
    with pytest.raises(ClassAffinityAmbiguityError, match="embedding dim"):
        compute_class_affinities(np.zeros((2, 3)), np.zeros((5, 4)), np.zeros(5, dtype=int), num_classes=3, m=2)


def test_affinity_rejects_non_2d():
    with pytest.raises(ClassAffinityAmbiguityError, match="2-D"):
        compute_class_affinities(np.zeros(3), TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2)


def test_affinity_rejects_non_positive_m():
    with pytest.raises(ClassAffinityAmbiguityError, match="m must be"):
        compute_class_affinities(TOY_EMBEDDINGS[:1], TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=0)


def test_affinity_deterministic():
    a = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2, exclude_self=True)
    b = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2, exclude_self=True)
    assert np.array_equal(a, b)


def test_affinity_scale_invariant_to_embedding_magnitude():
    query = np.array([[2.0, 0.0]])  # same direction as [1,0], different magnitude
    a1 = compute_class_affinities(query, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2)
    a2 = compute_class_affinities(np.array([[1.0, 0.0]]), TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2)
    assert np.allclose(a1, a2)


# --- compute_top_affinities ---


def test_top_affinities_hand_computed():
    affinities = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.8]])
    top = compute_top_affinities(affinities)
    assert top.top1_class[0] == 0 and top.top2_class[0] == 1
    assert top.raw_margin[0] == pytest.approx(0.4)
    # tie between class 1 and 2 for row 1 -> stable sort picks lower index first
    assert top.top1_class[1] == 1
    assert top.top2_class[1] == 2
    assert top.raw_margin[1] == pytest.approx(0.0)


def test_top_affinities_rejects_fewer_than_two_classes():
    with pytest.raises(ClassAffinityAmbiguityError, match="K >= 2"):
        compute_top_affinities(np.array([[0.5]]))


# --- fit_margin_scale / compute_primary_ambiguity ---


def test_fit_margin_scale_is_95th_percentile():
    margins = np.arange(1, 101) / 100.0  # 0.01..1.00
    scale = fit_margin_scale(margins)
    assert scale == pytest.approx(np.percentile(margins, 95.0))


def test_fit_margin_scale_rejects_zero():
    with pytest.raises(ClassAffinityAmbiguityError, match="numerically zero"):
        fit_margin_scale(np.zeros(10))


def test_fit_margin_scale_rejects_negative():
    with pytest.raises(ClassAffinityAmbiguityError, match="numerically zero"):
        fit_margin_scale(np.full(10, -0.5))


def test_compute_primary_ambiguity_endpoints():
    margin = np.array([0.0, 1.0, 2.0])
    scale = 1.0
    ambiguity = compute_primary_ambiguity(margin, scale)
    assert ambiguity[0] == pytest.approx(1.0)  # zero margin -> max ambiguity
    assert ambiguity[1] == pytest.approx(0.0)  # margin == scale -> min ambiguity
    assert ambiguity[2] == pytest.approx(0.0)  # margin > scale -> clipped, still min ambiguity


def test_compute_primary_ambiguity_bounded():
    rng = np.random.default_rng(0)
    margin = rng.uniform(-1, 3, size=50)
    ambiguity = compute_primary_ambiguity(margin, scale=1.5)
    assert np.all(ambiguity >= 0.0) and np.all(ambiguity <= 1.0)


def test_compute_primary_ambiguity_rejects_non_positive_scale():
    with pytest.raises(ClassAffinityAmbiguityError, match="scale must be"):
        compute_primary_ambiguity(np.array([0.5]), scale=0.0)


# --- compute_entropy_diagnostic ---


def test_entropy_low_for_dominant_class():
    affinities = np.array([[10.0, 0.0, 0.0]])
    entropy = compute_entropy_diagnostic(affinities, temperature=0.1)
    assert entropy[0] < 0.05


def test_entropy_bounded():
    rng = np.random.default_rng(1)
    affinities = rng.normal(size=(20, 5))
    entropy = compute_entropy_diagnostic(affinities, temperature=0.1)
    assert np.all(entropy >= 0.0) and np.all(entropy <= 1.0 + 1e-8)


def test_entropy_rejects_non_positive_temperature():
    with pytest.raises(ClassAffinityAmbiguityError, match="temperature"):
        compute_entropy_diagnostic(np.array([[0.1, 0.2]]), temperature=0.0)


# --- label-aware boundary gap (analysis only) ---


def test_boundary_gap_hand_computed():
    affinities = np.array([[0.9, 0.5, 0.1]])
    true_labels = np.array([0])
    gap = compute_label_aware_boundary_gap(affinities, true_labels)
    assert gap[0] == pytest.approx(0.9 - 0.5)


def test_boundary_gap_negative_when_wrong_class_dominates():
    affinities = np.array([[0.1, 0.9, 0.2]])
    true_labels = np.array([0])
    gap = compute_label_aware_boundary_gap(affinities, true_labels)
    assert gap[0] < 0


def test_fit_boundary_gap_scale_rejects_zero():
    with pytest.raises(ClassAffinityAmbiguityError, match="numerically zero"):
        fit_boundary_gap_scale(np.zeros(10))


def test_compute_label_aware_ambiguity_matches_primary_formula_shape():
    gap = np.array([0.0, 1.0, -1.0])
    ambiguity = compute_label_aware_ambiguity(gap, scale=1.0)
    assert ambiguity[0] == pytest.approx(1.0)
    assert ambiguity[1] == pytest.approx(0.0)
    assert ambiguity[2] == pytest.approx(1.0)  # negative gap clipped to 0 -> max ambiguity


# --- class affinity matrix ---


def test_class_matrix_symmetric_zero_diagonal_bounded():
    train_affinities = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2, exclude_self=True)
    matrix = compute_class_affinity_matrix(train_affinities, TOY_LABELS, num_classes=3)
    assert np.allclose(matrix, matrix.T)
    assert np.all(np.diag(matrix) == 0.0)
    assert np.all(matrix >= 0.0) and np.all(matrix <= 1.0)


def test_class_matrix_deterministic():
    train_affinities = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2, exclude_self=True)
    a = compute_class_affinity_matrix(train_affinities, TOY_LABELS, num_classes=3)
    b = compute_class_affinity_matrix(train_affinities, TOY_LABELS, num_classes=3)
    assert np.array_equal(a, b)


def test_class_matrix_uses_full_range():
    train_affinities = compute_class_affinities(TOY_EMBEDDINGS, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2, exclude_self=True)
    matrix = compute_class_affinity_matrix(train_affinities, TOY_LABELS, num_classes=3)
    off_diag = matrix[~np.eye(3, dtype=bool)]
    assert off_diag.max() == pytest.approx(1.0)
    assert off_diag.min() == pytest.approx(0.0)
    assert matrix[0, 1] == pytest.approx(1.0)  # classes 0/1 are the closest pair (see Phase 2's identical toy setup)


def test_class_matrix_degenerate_returns_zero():
    # 4 classes x 2 samples each, mutually orthogonal directions -> every
    # cross-class cosine similarity is exactly 0, so every off-diagonal
    # directed score is identical (degenerate) after rescaling.
    embeddings = np.repeat(np.eye(4), 2, axis=0)  # 8 rows, 2 per orthogonal direction
    labels = np.array([0, 0, 1, 1, 2, 2, 3, 3])
    train_affinities = compute_class_affinities(embeddings, embeddings, labels, num_classes=4, m=1, exclude_self=True)
    matrix = compute_class_affinity_matrix(train_affinities, labels, num_classes=4)
    assert np.all(matrix == 0.0)


def test_class_matrix_never_uses_validation_data():
    # Purely a documentation-consistency check: the function signature has
    # no validation-related parameter at all.
    import inspect

    params = set(inspect.signature(compute_class_affinity_matrix).parameters.keys())
    assert not any("val" in p.lower() or "test" in p.lower() for p in params)


# --- bundle ---


def test_sample_result_bundle_without_labels_has_no_label_aware_fields():
    affinities = compute_class_affinities(TOY_EMBEDDINGS[:2], TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2)
    result = compute_sample_class_affinity_result(affinities, margin_scale=0.5)
    assert result.boundary_gap is None
    assert result.label_aware_ambiguity is None
    assert result.ambiguity is not None


def test_sample_result_bundle_with_labels_populates_label_aware_fields():
    affinities = compute_class_affinities(TOY_EMBEDDINGS[:2], TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, m=2)
    result = compute_sample_class_affinity_result(
        affinities, margin_scale=0.5, true_labels=np.array([0, 0]), boundary_gap_scale=0.3
    )
    assert result.boundary_gap is not None
    assert result.label_aware_ambiguity is not None
    assert result.label_aware_ambiguity.shape == (2,)

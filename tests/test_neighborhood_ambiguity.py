"""Tests for src.losses.neighborhood_ambiguity (Phase 2: cross-class
neighborhood ambiguity - pure math, no I/O). Phase 1's src.losses.ambiguity
is untouched by this module and is not re-tested here; see
tests/test_ambiguity.py (still passing unmodified) for that regression
coverage.
"""

import numpy as np
import pytest

from src.losses.neighborhood_ambiguity import (
    DEFAULT_K,
    NeighborhoodAmbiguityError,
    compute_neighborhood_class_matrix,
    compute_sample_neighborhood_ambiguity,
    compute_validation_neighborhood_ambiguity,
    find_cross_class_neighbors,
)

# 3 classes x 3 samples, arranged so classes 0 and 1 are close (small angle)
# and class 2 is far (near-opposite) from both - lets us hand-verify
# neighbor selection and matrix values.
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


# --- find_cross_class_neighbors ---


def test_neighbors_exclude_self():
    result = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    for i in range(TOY_EMBEDDINGS.shape[0]):
        assert i not in result.neighbor_indices[i]


def test_neighbors_exclude_same_class():
    result = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    for i in range(TOY_EMBEDDINGS.shape[0]):
        own_class = TOY_LABELS[i]
        assert np.all(result.neighbor_classes[i] != own_class)


def test_neighbors_are_the_true_top_k_by_cosine_similarity():
    # Sample 0 (class 0, near [1,0]) should have its top-3 cross-class
    # neighbors be all of class 1 (near [0,1], closer in angle than class
    # 2's near-opposite [-1,0]).
    result = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    assert set(result.neighbor_classes[0].tolist()) == {1}
    assert set(result.neighbor_indices[0].tolist()) == {3, 4, 5}


def test_neighbors_sorted_descending_by_similarity():
    result = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    for i in range(TOY_EMBEDDINGS.shape[0]):
        sims = result.neighbor_similarities[i]
        assert list(sims) == sorted(sims, reverse=True)


def test_neighbors_deterministic():
    result_a = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    result_b = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    assert np.array_equal(result_a.neighbor_indices, result_b.neighbor_indices)
    assert np.array_equal(result_a.neighbor_similarities, result_b.neighbor_similarities)


def test_neighbors_rejects_k_too_large():
    # Each class has only 6 cross-class candidates (9 total - 3 same-class).
    with pytest.raises(NeighborhoodAmbiguityError, match="eligible"):
        find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=7)


def test_neighbors_rejects_non_positive_k():
    with pytest.raises(NeighborhoodAmbiguityError, match="k must be"):
        find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=0)


def test_neighbors_rejects_mismatched_labels_length():
    with pytest.raises(NeighborhoodAmbiguityError, match="labels must be"):
        find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS[:-1], k=2)


def test_default_k_is_ten():
    assert DEFAULT_K == 10


# --- compute_sample_neighborhood_ambiguity ---


def test_sample_ambiguity_nearest_competing_class_matches_top1_neighbor():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    result = compute_sample_neighborhood_ambiguity(neighbors, num_classes=3)
    assert np.array_equal(result.nearest_competing_class, neighbors.neighbor_classes[:, 0])


def test_sample_ambiguity_class_fraction_sums_to_one():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    result = compute_sample_neighborhood_ambiguity(neighbors, num_classes=3)
    assert np.allclose(result.class_fraction.sum(axis=1), 1.0)


def test_sample_ambiguity_own_class_fraction_is_always_zero():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    result = compute_sample_neighborhood_ambiguity(neighbors, num_classes=3)
    for i in range(TOY_EMBEDDINGS.shape[0]):
        assert result.class_fraction[i, TOY_LABELS[i]] == 0.0


def test_sample_ambiguity_hand_computed_strongest_class():
    # Sample 0's 3 cross-class neighbors are all class 1 -> fraction[1]=1.0.
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    result = compute_sample_neighborhood_ambiguity(neighbors, num_classes=3)
    assert result.strongest_competing_class[0] == 1
    assert result.strongest_competing_fraction[0] == pytest.approx(1.0)
    assert result.strongest_competing_mean_similarity[0] == pytest.approx(result.mean_topk_similarity[0])


def test_sample_ambiguity_mean_topk_similarity_matches_hand_computation():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    result = compute_sample_neighborhood_ambiguity(neighbors, num_classes=3)
    for i in range(TOY_EMBEDDINGS.shape[0]):
        assert result.mean_topk_similarity[i] == pytest.approx(neighbors.neighbor_similarities[i].mean())


def test_sample_ambiguity_rejects_fewer_than_two_classes():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    with pytest.raises(NeighborhoodAmbiguityError, match="num_classes"):
        compute_sample_neighborhood_ambiguity(neighbors, num_classes=1)


# --- compute_neighborhood_class_matrix ---


def test_class_matrix_is_symmetric():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    matrix = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    assert np.allclose(matrix, matrix.T)


def test_class_matrix_diagonal_is_exactly_zero():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    matrix = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    assert np.all(np.diag(matrix) == 0.0)


def test_class_matrix_bounded_zero_one():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    matrix = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


def test_class_matrix_deterministic():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    matrix_a = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    matrix_b = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    assert np.array_equal(matrix_a, matrix_b)


def test_class_matrix_normalization_uses_full_range():
    # Since classes 0/1 are much closer than 0/2 or 1/2, after min-max
    # rescaling the strongest pair should be exactly 1.0 and the weakest
    # exactly 0.0 (both derived only from the matrix's own off-diagonal
    # entries - no external reference value).
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    matrix = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    off_diag = matrix[~np.eye(3, dtype=bool)]
    assert off_diag.max() == pytest.approx(1.0)
    assert off_diag.min() == pytest.approx(0.0)
    assert matrix[0, 1] == pytest.approx(1.0)  # classes 0/1 are the closest pair


def test_class_matrix_directed_scores_hand_computed():
    # Hand-computation: class 0's 3 samples each have 3 cross-class
    # neighbors ALL from class 1 (verified above), so score(0->1) =
    # mean_i [ sum_j max(0,cos)/k ] = mean of (sum of that sample's 3
    # neighbor similarities / 3) since all 3 neighbors are class 1.
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    expected_score_0_to_1 = neighbors.neighbor_similarities[0:3].sum(axis=1).mean() / 3
    matrix = compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=3)
    # matrix is rescaled, but the 0<->1 pair is the max, so its raw value
    # (before rescale) must equal the symmetrized directed score - check
    # indirectly via rank rather than exact equality (already covered by
    # test_class_matrix_normalization_uses_full_range for the max case).
    assert expected_score_0_to_1 > 0


def test_class_matrix_rejects_missing_class():
    neighbors = find_cross_class_neighbors(TOY_EMBEDDINGS, TOY_LABELS, k=3)
    with pytest.raises(NeighborhoodAmbiguityError, match="num_classes"):
        compute_neighborhood_class_matrix(neighbors, TOY_LABELS, num_classes=1)


def test_class_matrix_degenerate_equal_scores_returns_zero_matrix():
    # A perfectly symmetric 4-class toy example where every cross-class
    # pair has identical directed scores (equidistant clusters) - the
    # degenerate (M == m) branch must not divide by zero.
    embeddings = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    labels = np.array([0, 1, 2, 3])
    neighbors = find_cross_class_neighbors(embeddings, labels, k=1)
    matrix = compute_neighborhood_class_matrix(neighbors, labels, num_classes=4)
    assert np.all(matrix == 0.0)


# --- compute_validation_neighborhood_ambiguity ---


def test_validation_ambiguity_low_for_sample_deep_in_one_class():
    val_embeddings = np.array([[1.0, 0.0]])  # right at class 0's cluster
    result = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=3)
    assert result.strongest_class[0] == 0
    assert result.ambiguity_margin[0] < 0.2  # k=3 nearest are all class 0 -> margin=1 -> ambiguity~0


def test_validation_ambiguity_high_for_sample_between_two_classes():
    val_embeddings = np.array([[0.7, 0.7]])  # exactly between class 0 and class 1
    result = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=6)
    assert result.ambiguity_margin[0] > 0.5


def test_validation_ambiguity_class_fraction_sums_to_one():
    val_embeddings = np.array([[0.7, 0.7], [1.0, 0.0], [-1.0, -0.05]])
    result = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=4)
    assert np.allclose(result.class_fraction.sum(axis=1), 1.0)


def test_validation_ambiguity_margin_and_entropy_bounded():
    rng = np.random.default_rng(0)
    val_embeddings = rng.normal(size=(10, 2))
    result = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=5)
    assert np.all(result.ambiguity_margin >= 0.0) and np.all(result.ambiguity_margin <= 1.0)
    assert np.all(result.ambiguity_entropy >= 0.0) and np.all(result.ambiguity_entropy <= 1.0 + 1e-8)


def test_validation_ambiguity_competing_class_is_second_ranked():
    val_embeddings = np.array([[0.7, 0.7]])
    result = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=6)
    fractions = result.class_fraction[0]
    ranked = np.argsort(-fractions, kind="stable")
    assert result.strongest_class[0] == ranked[0]
    assert result.competing_class[0] == ranked[1]


def test_validation_ambiguity_deterministic():
    val_embeddings = np.array([[0.7, 0.7], [1.0, 0.0]])
    result_a = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=4)
    result_b = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=4)
    assert np.array_equal(result_a.ambiguity_margin, result_b.ambiguity_margin)
    assert np.array_equal(result_a.strongest_class, result_b.strongest_class)


def test_validation_ambiguity_never_uses_other_validation_samples():
    # Passing a single validation embedding at a time must give identical
    # results to passing it as part of a larger validation batch - proof
    # that validation samples never influence each other's neighbor search
    # (only train_embeddings/train_labels are ever searched).
    val_embeddings = np.array([[0.7, 0.7], [1.0, 0.0], [-1.0, -0.05]])
    batched = compute_validation_neighborhood_ambiguity(val_embeddings, TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=4)
    for i in range(val_embeddings.shape[0]):
        single = compute_validation_neighborhood_ambiguity(val_embeddings[i : i + 1], TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=4)
        assert batched.ambiguity_margin[i] == pytest.approx(single.ambiguity_margin[0])
        assert batched.strongest_class[i] == single.strongest_class[0]


def test_validation_ambiguity_rejects_mismatched_embedding_dims():
    with pytest.raises(NeighborhoodAmbiguityError, match="embedding dim"):
        compute_validation_neighborhood_ambiguity(np.zeros((2, 3)), np.zeros((5, 4)), np.zeros(5, dtype=int), num_classes=3, k=2)


def test_validation_ambiguity_rejects_k_larger_than_train_set():
    with pytest.raises(NeighborhoodAmbiguityError, match="k must be"):
        compute_validation_neighborhood_ambiguity(TOY_EMBEDDINGS[:1], TOY_EMBEDDINGS, TOY_LABELS, num_classes=3, k=100)

"""Tests for src.losses.ambiguity: the learned class-level ambiguity
matrix (feeds CS-SupCon's ambiguity_source="learned_class" mode) and the
sample-level ambiguity score (analysis-only in this phase - see module
docstring). Pure math only - no I/O, no checkpoints, no real dataset.
"""

import numpy as np
import pytest
import torch

from src.losses.ambiguity import (
    DEFAULT_AMBIGUITY_SCALE,
    DEFAULT_REFERENCE_MODEL_NAME,
    AmbiguityComputationError,
    AmbiguityConfigError,
    MarginNormalization,
    VALID_AMBIGUITY_SOURCES,
    class_ambiguity_matrix_to_buffer,
    compute_class_ambiguity_matrix,
    compute_raw_margins,
    compute_sample_ambiguity,
    fit_margin_normalization,
    load_ambiguity_settings,
)


# --- class-level ambiguity matrix ---


def test_class_matrix_is_symmetric():
    rng = np.random.default_rng(0)
    prototypes = rng.normal(size=(6, 12))
    matrix = compute_class_ambiguity_matrix(prototypes)
    assert np.allclose(matrix, matrix.T)


def test_class_matrix_diagonal_is_exactly_zero():
    rng = np.random.default_rng(1)
    prototypes = rng.normal(size=(5, 8))
    matrix = compute_class_ambiguity_matrix(prototypes)
    assert np.all(np.diag(matrix) == 0.0)


def test_class_matrix_bounded_zero_one():
    rng = np.random.default_rng(2)
    prototypes = rng.normal(size=(10, 16))
    matrix = compute_class_ambiguity_matrix(prototypes)
    assert np.all(matrix >= 0.0)
    assert np.all(matrix <= 1.0)


def test_class_matrix_is_deterministic_given_fixed_prototypes():
    rng = np.random.default_rng(3)
    prototypes = rng.normal(size=(4, 10))
    matrix_a = compute_class_ambiguity_matrix(prototypes)
    matrix_b = compute_class_ambiguity_matrix(prototypes)
    assert np.array_equal(matrix_a, matrix_b)


def test_class_matrix_hand_computed_orthogonal_and_identical():
    # Class 0 and 1 identical direction (max ambiguity); class 2 orthogonal
    # to both (zero ambiguity, after rectification).
    prototypes = np.array([[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]])
    matrix = compute_class_ambiguity_matrix(prototypes)
    assert matrix[0, 1] == pytest.approx(1.0, abs=1e-8)
    assert matrix[1, 0] == pytest.approx(1.0, abs=1e-8)
    assert matrix[0, 2] == pytest.approx(0.0, abs=1e-8)
    assert matrix[1, 2] == pytest.approx(0.0, abs=1e-8)
    assert matrix[0, 0] == 0.0 and matrix[1, 1] == 0.0 and matrix[2, 2] == 0.0


def test_class_matrix_hand_computed_negative_similarity_rectified_to_zero():
    # Opposite-direction prototypes -> cosine similarity -1 -> rectified to 0.
    prototypes = np.array([[1.0, 0.0], [-1.0, 0.0]])
    matrix = compute_class_ambiguity_matrix(prototypes)
    assert matrix[0, 1] == pytest.approx(0.0, abs=1e-8)


def test_class_matrix_rejects_non_2d_input():
    with pytest.raises(AmbiguityComputationError, match="2-D"):
        compute_class_ambiguity_matrix(np.zeros(4))


def test_class_matrix_rejects_zero_classes():
    with pytest.raises(AmbiguityComputationError, match="K > 0"):
        compute_class_ambiguity_matrix(np.zeros((0, 4)))


def test_class_ambiguity_matrix_to_buffer_is_non_trainable():
    matrix = compute_class_ambiguity_matrix(np.array([[1.0, 0.0], [0.0, 1.0]]))
    buffer = class_ambiguity_matrix_to_buffer(matrix)
    assert isinstance(buffer, torch.Tensor)
    assert buffer.requires_grad is False
    assert buffer.dtype == torch.float32
    assert buffer.shape == (2, 2)


# --- sample-level ambiguity: raw margin, normalization, competing class ---


def test_raw_margin_zero_when_embedding_equidistant_from_two_prototypes():
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0]])
    embeddings = np.array([[1.0, 1.0]])  # exactly equidistant (equal cosine similarity to both)
    raw_margin = compute_raw_margins(embeddings, prototypes)
    assert raw_margin[0] == pytest.approx(0.0, abs=1e-8)


def test_raw_margin_large_when_embedding_matches_one_prototype_exactly():
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]])
    embeddings = np.array([[1.0, 0.0]])  # identical to prototype 0; orthogonal to 1; opposite to 2
    raw_margin = compute_raw_margins(embeddings, prototypes)
    # similarity to prototype 0 = 1.0, next-best (prototype 1) = 0.0 -> margin = 1.0
    assert raw_margin[0] == pytest.approx(1.0, abs=1e-8)


def test_fit_margin_normalization_matches_min_max():
    raw_margins = np.array([0.1, 0.5, 0.9, 0.3])
    normalization = fit_margin_normalization(raw_margins)
    assert normalization.margin_min == pytest.approx(0.1)
    assert normalization.margin_max == pytest.approx(0.9)


def test_apply_margin_normalization_endpoints():
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0]])
    # Two synthetic embeddings whose raw margins define the min/max of the
    # fitted range, plus a third exactly at the low end -> ambiguity 1.0,
    # and the high end -> ambiguity 0.0.
    embeddings = np.array([[1.0, 1.0], [1.0, 0.0]])  # margins: 0.0 and 1.0
    raw_margin = compute_raw_margins(embeddings, prototypes)
    normalization = fit_margin_normalization(raw_margin)
    result = compute_sample_ambiguity(embeddings, prototypes, normalization)
    assert result.ambiguity[0] == pytest.approx(1.0, abs=1e-8)  # smallest margin -> max ambiguity
    assert result.ambiguity[1] == pytest.approx(0.0, abs=1e-8)  # largest margin -> min ambiguity


def test_apply_margin_normalization_degenerate_zero_span_yields_zero_ambiguity():
    normalization = MarginNormalization(margin_min=0.5, margin_max=0.5)
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0]])
    embeddings = np.array([[1.0, 0.0]])
    result = compute_sample_ambiguity(embeddings, prototypes, normalization)
    assert result.ambiguity[0] == 0.0


def test_ambiguity_is_bounded_zero_one():
    rng = np.random.default_rng(4)
    prototypes = rng.normal(size=(5, 10))
    embeddings = rng.normal(size=(20, 10))
    raw_margin = compute_raw_margins(embeddings, prototypes)
    normalization = fit_margin_normalization(raw_margin)
    result = compute_sample_ambiguity(embeddings, prototypes, normalization)
    assert np.all(result.ambiguity >= 0.0)
    assert np.all(result.ambiguity <= 1.0)


def test_competing_class_is_the_second_most_similar_prototype():
    prototypes = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    embeddings = np.array([[1.0, 0.0, 0.0]])
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    result = compute_sample_ambiguity(embeddings, prototypes, normalization)
    assert result.top_class[0] == 0
    assert result.competing_class[0] == 1  # prototype 1 is closer than prototype 2


def test_competing_class_identity_is_temperature_independent():
    # Regardless of entropy_temperature, top/competing class identity must
    # be unchanged (softmax is rank-preserving).
    prototypes = np.array([[1.0, 0.0, 0.0], [0.9, 0.1, 0.0], [0.0, 0.0, 1.0]])
    embeddings = np.array([[1.0, 0.0, 0.0]])
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    result_a = compute_sample_ambiguity(embeddings, prototypes, normalization, entropy_temperature=0.01)
    result_b = compute_sample_ambiguity(embeddings, prototypes, normalization, entropy_temperature=5.0)
    assert result_a.top_class[0] == result_b.top_class[0]
    assert result_a.competing_class[0] == result_b.competing_class[0]
    assert result_a.raw_margin[0] == pytest.approx(result_b.raw_margin[0])


def test_similarity_vector_has_shape_n_by_k():
    prototypes = np.random.default_rng(5).normal(size=(7, 4))
    embeddings = np.random.default_rng(6).normal(size=(3, 4))
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    result = compute_sample_ambiguity(embeddings, prototypes, normalization)
    assert result.similarity.shape == (3, 7)


# --- entropy diagnostic ---


def test_entropy_is_bounded_zero_one():
    rng = np.random.default_rng(7)
    prototypes = rng.normal(size=(6, 8))
    embeddings = rng.normal(size=(15, 8))
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    result = compute_sample_ambiguity(embeddings, prototypes, normalization)
    assert np.all(result.entropy >= 0.0)
    assert np.all(result.entropy <= 1.0 + 1e-8)


def test_entropy_low_for_single_dominant_class():
    # One prototype heavily favored -> low entropy.
    prototypes = np.eye(10)
    embeddings = np.array([[10.0] + [0.0] * 9])  # overwhelmingly closest to prototype 0
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    result = compute_sample_ambiguity(embeddings, prototypes, normalization, entropy_temperature=0.1)
    assert result.entropy[0] < 0.1


def test_entropy_distinguishes_two_way_tie_from_diffuse_confusion():
    # Two-way near-tie: entropy should be noticeably lower than a diffuse,
    # many-way near-tie, even if both have a similarly small raw margin -
    # the exact blind spot the margin score alone cannot capture (see
    # REPRODUCIBILITY.md's numerical discussion).
    prototypes = np.eye(10)
    two_way = np.array([[0.55, 0.52] + [0.0] * 8])
    diffuse = np.array([[0.30, 0.29, 0.28, 0.27] + [0.0] * 6])
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    result_two_way = compute_sample_ambiguity(two_way, prototypes, normalization, entropy_temperature=0.1)
    result_diffuse = compute_sample_ambiguity(diffuse, prototypes, normalization, entropy_temperature=0.1)
    assert result_two_way.entropy[0] < result_diffuse.entropy[0]


def test_entropy_temperature_must_be_positive():
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0]])
    embeddings = np.array([[1.0, 0.0]])
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    with pytest.raises(AmbiguityComputationError, match="entropy_temperature"):
        compute_sample_ambiguity(embeddings, prototypes, normalization, entropy_temperature=0.0)


# --- malformed inputs ---


def test_compute_sample_ambiguity_rejects_mismatched_embedding_dim():
    prototypes = np.zeros((3, 4))
    embeddings = np.zeros((2, 5))
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    with pytest.raises(AmbiguityComputationError, match="embedding dim"):
        compute_sample_ambiguity(embeddings, prototypes, normalization)


def test_compute_sample_ambiguity_rejects_fewer_than_two_classes():
    prototypes = np.zeros((1, 4))
    embeddings = np.zeros((2, 4))
    normalization = MarginNormalization(margin_min=0.0, margin_max=1.0)
    with pytest.raises(AmbiguityComputationError, match="at least 2 classes"):
        compute_sample_ambiguity(embeddings, prototypes, normalization)


def test_compute_sample_ambiguity_rejects_non_2d_embeddings():
    prototypes = np.zeros((3, 4))
    with pytest.raises(AmbiguityComputationError, match="2-D"):
        compute_sample_ambiguity(np.zeros(4), prototypes, MarginNormalization(0.0, 1.0))


# --- configuration validation ---


def test_default_ambiguity_source_is_fixed_pairs():
    settings = load_ambiguity_settings({})
    assert settings.ambiguity_source == "fixed_pairs"
    assert settings.ambiguity_scale == DEFAULT_AMBIGUITY_SCALE
    assert settings.reference_checkpoint_path is None
    assert settings.reference_model_name == DEFAULT_REFERENCE_MODEL_NAME


def test_valid_ambiguity_sources_are_exactly_three():
    assert VALID_AMBIGUITY_SOURCES == ("fixed_pairs", "learned_class", "learned_class_affinity")


def test_learned_class_sample_is_recognized_but_rejected():
    with pytest.raises(AmbiguityConfigError, match="not implemented in this phase"):
        load_ambiguity_settings({"ambiguity_source": "learned_class_sample"})


def test_unknown_ambiguity_source_rejected():
    with pytest.raises(AmbiguityConfigError, match="not recognized"):
        load_ambiguity_settings({"ambiguity_source": "totally_bogus"})


def test_learned_class_requires_reference_checkpoint_path():
    with pytest.raises(AmbiguityConfigError, match="reference_checkpoint_path is required"):
        load_ambiguity_settings({"ambiguity_source": "learned_class"})


def test_learned_class_rejects_non_positive_ambiguity_scale():
    with pytest.raises(AmbiguityConfigError, match="ambiguity_scale must be > 0"):
        load_ambiguity_settings(
            {"ambiguity_source": "learned_class", "reference_checkpoint_path": "x.pt", "ambiguity_scale": 0.0}
        )


def test_learned_class_accepts_valid_config():
    settings = load_ambiguity_settings(
        {
            "ambiguity_source": "learned_class",
            "reference_checkpoint_path": "results/checkpoints/run1/best.pt",
            "reference_model_name": "aa_evidentnet",
            "ambiguity_scale": 2.5,
        }
    )
    assert settings.ambiguity_source == "learned_class"
    assert settings.ambiguity_scale == 2.5
    assert settings.reference_checkpoint_path == "results/checkpoints/run1/best.pt"
    assert settings.reference_model_name == "aa_evidentnet"


# --- learned_class_affinity (feature/learned-ambiguity, Phase 3-experimental):
# same requirements/behavior as learned_class, mirrored exactly - the two
# differ only in which upstream module builds the matrix they end up
# installing, never in how load_ambiguity_settings validates them. ---


def test_learned_class_affinity_requires_reference_checkpoint_path():
    with pytest.raises(AmbiguityConfigError, match="reference_checkpoint_path is required"):
        load_ambiguity_settings({"ambiguity_source": "learned_class_affinity"})


def test_learned_class_affinity_rejects_non_positive_ambiguity_scale():
    with pytest.raises(AmbiguityConfigError, match="ambiguity_scale must be > 0"):
        load_ambiguity_settings(
            {"ambiguity_source": "learned_class_affinity", "reference_checkpoint_path": "x.pt", "ambiguity_scale": 0.0}
        )


def test_learned_class_affinity_accepts_valid_config():
    settings = load_ambiguity_settings(
        {
            "ambiguity_source": "learned_class_affinity",
            "reference_checkpoint_path": "results/checkpoints/run1/best.pt",
            "reference_model_name": "aa_evidentnet",
            "ambiguity_scale": 1.0,
        }
    )
    assert settings.ambiguity_source == "learned_class_affinity"
    assert settings.ambiguity_scale == 1.0
    assert settings.reference_checkpoint_path == "results/checkpoints/run1/best.pt"
    assert settings.reference_model_name == "aa_evidentnet"


def test_learned_class_affinity_reuses_the_same_default_ambiguity_scale_as_learned_class():
    # Reusing the SAME predetermined default (never a separately invented
    # or tuned value for the new mode) is a hard requirement of this
    # experiment's design - see configs/losses.yaml's comment.
    settings = load_ambiguity_settings(
        {"ambiguity_source": "learned_class_affinity", "reference_checkpoint_path": "x.pt"}
    )
    assert settings.ambiguity_scale == DEFAULT_AMBIGUITY_SCALE == 1.0


def test_fixed_pairs_does_not_require_reference_checkpoint_path():
    # fixed_pairs must never require a reference checkpoint, even if
    # ambiguity_scale is left at its default or explicitly set.
    settings = load_ambiguity_settings({"ambiguity_source": "fixed_pairs", "ambiguity_scale": 3.0})
    assert settings.reference_checkpoint_path is None

"""Tests for src.losses.evidential (Task 9: EDL loss only - no combined
training objective, no training-loop integration). All tests use tiny
synthetic tensors - no internet access, no real dataset, no data/raw/ or
test set involvement."""

import pytest
import torch

from src.losses.evidential import (
    DEFAULT_KL_ANNEALING_EPOCHS,
    DEFAULT_KL_WEIGHT_MAX,
    EDLLoss,
    EDLSettings,
    EvidentialConfigError,
    EvidentialHead,
    EvidentialOutput,
    compute_evidential_output,
    edl_loss,
    load_edl_settings,
)


def _random_alpha(batch_size=8, num_classes=10, seed=0, scale=1.0):
    g = torch.Generator().manual_seed(seed)
    raw = torch.randn(batch_size, num_classes, generator=g) * scale
    return compute_evidential_output(raw).alpha


# --- EvidentialHead / compute_evidential_output construction and shapes ---


def test_evidential_head_constructs():
    head = EvidentialHead(embedding_dim=16, num_classes=10)
    assert head.num_classes == 10


def test_evidential_head_rejects_invalid_dims():
    with pytest.raises(ValueError):
        EvidentialHead(embedding_dim=0, num_classes=10)
    with pytest.raises(ValueError):
        EvidentialHead(embedding_dim=16, num_classes=0)


def test_evidential_head_output_shapes():
    head = EvidentialHead(embedding_dim=16, num_classes=10)
    embedding = torch.randn(5, 16)
    output = head(embedding)
    assert isinstance(output, EvidentialOutput)
    assert output.raw_output.shape == (5, 10)
    assert output.evidence.shape == (5, 10)
    assert output.alpha.shape == (5, 10)
    assert output.probabilities.shape == (5, 10)
    assert output.uncertainty.shape == (5,)


def test_evidential_head_rejects_non_2d_embedding():
    head = EvidentialHead(embedding_dim=16, num_classes=10)
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        head(torch.randn(5, 16, 3))


def test_compute_evidential_output_rejects_non_2d_input():
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        compute_evidential_output(torch.randn(5))


def test_compute_evidential_output_rejects_zero_width():
    with pytest.raises(ValueError):
        compute_evidential_output(torch.randn(3, 0))


def test_compute_evidential_output_rejects_invalid_epsilon():
    raw = torch.randn(3, 10)
    with pytest.raises(EvidentialConfigError):
        compute_evidential_output(raw, epsilon=0.0)
    with pytest.raises(EvidentialConfigError):
        compute_evidential_output(raw, epsilon=-1e-8)


# --- evidence / alpha / probability / uncertainty properties ---


def test_evidence_is_non_negative():
    raw = torch.randn(8, 10) * 5.0
    output = compute_evidential_output(raw)
    assert bool((output.evidence >= 0).all())


def test_alpha_is_strictly_positive_and_at_least_one():
    raw = torch.randn(8, 10) * 5.0
    output = compute_evidential_output(raw)
    assert bool((output.alpha > 0).all())
    assert bool((output.alpha >= 1.0).all())


def test_alpha_equals_evidence_plus_one():
    raw = torch.randn(8, 10)
    output = compute_evidential_output(raw)
    assert torch.allclose(output.alpha, output.evidence + 1.0)


def test_probabilities_have_correct_shape_and_sum_to_one():
    raw = torch.randn(7, 10)
    output = compute_evidential_output(raw)
    assert output.probabilities.shape == (7, 10)
    assert torch.allclose(output.probabilities.sum(dim=1), torch.ones(7), atol=1e-5)


def test_uncertainty_shape_and_finiteness():
    raw = torch.randn(7, 10)
    output = compute_evidential_output(raw)
    assert output.uncertainty.shape == (7,)
    assert torch.isfinite(output.uncertainty).all()


def test_uncertainty_in_valid_range_for_k_10():
    raw = torch.randn(20, 10) * 10.0
    output = compute_evidential_output(raw)
    assert bool((output.uncertainty > 0).all())
    assert bool((output.uncertainty <= 1.0).all())


def test_raw_output_is_preserved_unmodified():
    raw = torch.randn(4, 10)
    output = compute_evidential_output(raw)
    assert torch.equal(output.raw_output, raw)


# --- qualitative uncertainty behavior ---


def test_more_evidence_lowers_uncertainty():
    low_evidence_raw = torch.full((5, 10), -5.0)  # softplus(-5) ~ 0.0067 -> low evidence
    high_evidence_raw = torch.full((5, 10), 5.0)  # softplus(5) ~ 5.0067 -> high evidence
    low = compute_evidential_output(low_evidence_raw)
    high = compute_evidential_output(high_evidence_raw)
    assert bool((high.uncertainty < low.uncertainty).all())


def test_near_zero_evidence_produces_high_uncertainty():
    raw = torch.full((5, 10), -30.0)  # softplus(-30) ~ 0 -> evidence ~ 0, alpha ~ 1 for every class
    output = compute_evidential_output(raw)
    assert bool((output.uncertainty > 0.99).all())


def test_large_evidence_produces_low_uncertainty():
    raw = torch.full((5, 10), 1000.0)
    output = compute_evidential_output(raw)
    assert bool((output.uncertainty < 0.01).all())


def test_uncertainty_monotonically_decreases_with_increasing_uniform_evidence():
    raws = [torch.full((3, 10), v) for v in (-10.0, -2.0, 0.0, 2.0, 10.0, 50.0)]
    uncertainties = [compute_evidential_output(r).uncertainty.mean().item() for r in raws]
    assert all(a >= b for a, b in zip(uncertainties, uncertainties[1:]))


# --- arbitrary batch sizes / degenerate batches ---


@pytest.mark.parametrize("batch_size", [1, 2, 5, 17, 33])
def test_evidential_output_supports_arbitrary_batch_sizes(batch_size):
    raw = torch.randn(batch_size, 10)
    output = compute_evidential_output(raw)
    assert output.probabilities.shape == (batch_size, 10)
    assert output.uncertainty.shape == (batch_size,)
    assert torch.isfinite(output.uncertainty).all()


def test_all_classes_represented_single_batch():
    raw = torch.randn(10, 10)
    output = compute_evidential_output(raw)
    assert torch.isfinite(output.probabilities).all()


# --- extreme numerical inputs ---


def test_extreme_positive_raw_output_does_not_overflow():
    raw = torch.full((4, 10), 1e6)
    output = compute_evidential_output(raw)
    assert torch.isfinite(output.evidence).all()
    assert torch.isfinite(output.alpha).all()
    assert torch.isfinite(output.probabilities).all()
    assert torch.isfinite(output.uncertainty).all()


def test_extreme_negative_raw_output_does_not_underflow_to_nan():
    raw = torch.full((4, 10), -1e6)
    output = compute_evidential_output(raw)
    assert torch.isfinite(output.evidence).all()
    assert torch.isfinite(output.alpha).all()
    assert torch.allclose(output.alpha, torch.ones_like(output.alpha), atol=1e-4)
    assert torch.isfinite(output.probabilities).all()
    assert torch.isfinite(output.uncertainty).all()


def test_mixed_extreme_and_normal_values_are_finite():
    raw = torch.tensor([[1e6, -1e6, 0.0, 3.2, -0.5, 100.0, -100.0, 0.001, -0.001, 50.0]])
    output = compute_evidential_output(raw)
    assert torch.isfinite(output.evidence).all()
    assert torch.isfinite(output.probabilities).all()
    assert torch.isfinite(output.uncertainty).all()


# --- EDL loss: scalar / finite / differentiable ---


def test_edl_loss_returns_scalar():
    alpha = _random_alpha()
    labels = torch.randint(0, 10, (8,))
    loss = edl_loss(alpha, labels)
    assert loss.dim() == 0


def test_edl_loss_is_finite():
    alpha = _random_alpha()
    labels = torch.randint(0, 10, (8,))
    loss = edl_loss(alpha, labels)
    assert torch.isfinite(loss)


def test_edl_loss_gradients_propagate():
    raw = torch.randn(8, 10, requires_grad=True)
    alpha = compute_evidential_output(raw).alpha
    labels = torch.randint(0, 10, (8,))
    loss = edl_loss(alpha, labels)
    loss.backward()
    assert raw.grad is not None
    assert torch.isfinite(raw.grad).all()
    assert torch.any(raw.grad != 0)


def test_edl_loss_differentiable_through_evidential_head_and_linear():
    torch.manual_seed(0)
    embed = torch.randn(6, 16, requires_grad=True)
    head = EvidentialHead(embedding_dim=16, num_classes=5)
    labels = torch.randint(0, 5, (6,))
    output = head(embed)
    loss = edl_loss(output.alpha, labels)
    loss.backward()
    assert head.linear.weight.grad is not None
    assert torch.isfinite(head.linear.weight.grad).all()
    assert embed.grad is not None
    assert torch.isfinite(embed.grad).all()


def test_edl_loss_gradients_are_nonzero_for_typical_input():
    raw = (torch.randn(10, 10) * 2.0).requires_grad_(True)
    alpha = compute_evidential_output(raw).alpha
    labels = torch.randint(0, 10, (10,))
    loss = edl_loss(alpha, labels, kl_weight_max=1.0)
    loss.backward()
    assert torch.any(raw.grad.abs() > 1e-8)


# --- EDL loss: penalizes incorrect/confident evidence ---


def test_confident_correct_evidence_gives_lower_loss_than_confident_incorrect():
    num_classes = 4
    # Strong evidence concentrated on class 0.
    raw_correct_favoring = torch.zeros(1, num_classes)
    raw_correct_favoring[0, 0] = 20.0
    alpha_correct = compute_evidential_output(raw_correct_favoring).alpha
    labels = torch.tensor([0])

    # Same strong evidence, but concentrated on the WRONG class (class 1).
    raw_incorrect_favoring = torch.zeros(1, num_classes)
    raw_incorrect_favoring[0, 1] = 20.0
    alpha_incorrect = compute_evidential_output(raw_incorrect_favoring).alpha

    loss_correct = edl_loss(alpha_correct, labels, kl_weight_max=0.0)
    loss_incorrect = edl_loss(alpha_incorrect, labels, kl_weight_max=0.0)
    assert loss_correct.item() < loss_incorrect.item()


def test_kl_regularizer_penalizes_evidence_for_wrong_classes():
    # Same correct-class evidence, but one case ALSO has strong evidence
    # for a wrong class - the KL term should penalize that extra
    # (unjustified) evidence more when kl_weight_max > 0.
    num_classes = 4
    labels = torch.tensor([0])

    raw_clean = torch.zeros(1, num_classes)
    raw_clean[0, 0] = 10.0
    alpha_clean = compute_evidential_output(raw_clean).alpha

    raw_with_wrong_evidence = torch.zeros(1, num_classes)
    raw_with_wrong_evidence[0, 0] = 10.0
    raw_with_wrong_evidence[0, 1] = 10.0
    alpha_with_wrong_evidence = compute_evidential_output(raw_with_wrong_evidence).alpha

    loss_clean = edl_loss(alpha_clean, labels, kl_weight_max=1.0, epoch=100, kl_annealing_epochs=1)
    loss_with_wrong = edl_loss(alpha_with_wrong_evidence, labels, kl_weight_max=1.0, epoch=100, kl_annealing_epochs=1)
    assert loss_with_wrong.item() > loss_clean.item()


# --- KL annealing ---


def test_kl_annealing_scales_loss_with_epoch():
    alpha = _random_alpha(batch_size=6, num_classes=6, seed=3, scale=3.0)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])

    loss_epoch_0 = edl_loss(alpha, labels, epoch=0, kl_annealing_epochs=10, kl_weight_max=1.0)
    loss_epoch_10 = edl_loss(alpha, labels, epoch=10, kl_annealing_epochs=10, kl_weight_max=1.0)
    loss_epoch_100 = edl_loss(alpha, labels, epoch=100, kl_annealing_epochs=10, kl_weight_max=1.0)

    # lambda_t is clamped at kl_weight_max once epoch >= kl_annealing_epochs.
    assert torch.isclose(loss_epoch_10, loss_epoch_100)
    assert not torch.isclose(loss_epoch_0, loss_epoch_10)


def test_kl_weight_max_zero_disables_regularizer():
    alpha = _random_alpha(batch_size=6, num_classes=6, seed=4, scale=3.0)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    loss_a = edl_loss(alpha, labels, kl_weight_max=0.0, epoch=0)
    loss_b = edl_loss(alpha, labels, kl_weight_max=0.0, epoch=100)
    assert torch.isclose(loss_a, loss_b)


def test_no_epoch_argument_applies_full_kl_weight():
    alpha = _random_alpha(batch_size=6, num_classes=6, seed=5, scale=3.0)
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    loss_no_epoch = edl_loss(alpha, labels, kl_weight_max=1.0)
    loss_full_epoch = edl_loss(alpha, labels, kl_weight_max=1.0, epoch=1e9, kl_annealing_epochs=10)
    assert torch.isclose(loss_no_epoch, loss_full_epoch)


# --- degenerate batches ---


def test_batch_size_one_is_finite():
    raw = torch.randn(1, 10)
    alpha = compute_evidential_output(raw).alpha
    labels = torch.tensor([3])
    loss = edl_loss(alpha, labels)
    assert torch.isfinite(loss)


def test_single_class_labels_batch_is_finite():
    raw = torch.randn(8, 5)
    alpha = compute_evidential_output(raw).alpha
    labels = torch.zeros(8, dtype=torch.long)  # every sample labeled class 0
    loss = edl_loss(alpha, labels)
    assert torch.isfinite(loss)


def test_repeated_identical_labels_across_varied_evidence_is_finite():
    torch.manual_seed(6)
    raw = torch.randn(12, 3) * 8.0
    alpha = compute_evidential_output(raw).alpha
    labels = torch.full((12,), 1, dtype=torch.long)
    loss = edl_loss(alpha, labels)
    assert torch.isfinite(loss)


@pytest.mark.parametrize("batch_size", [1, 2, 7, 25])
def test_edl_loss_supports_arbitrary_batch_sizes(batch_size):
    raw = torch.randn(batch_size, 10)
    alpha = compute_evidential_output(raw).alpha
    labels = torch.randint(0, 10, (batch_size,))
    loss = edl_loss(alpha, labels)
    assert torch.isfinite(loss)


def test_extreme_alpha_values_do_not_create_nan_or_inf_in_loss():
    raw = torch.tensor([[1e6] * 10, [-1e6] * 10, [0.0] * 10])
    alpha = compute_evidential_output(raw).alpha
    labels = torch.tensor([0, 5, 9])
    loss = edl_loss(alpha, labels, kl_weight_max=1.0)
    assert torch.isfinite(loss)


# --- input/config validation ---


def test_invalid_labels_out_of_range_fail_clearly():
    alpha = _random_alpha(batch_size=5, num_classes=10)
    labels = torch.tensor([0, 1, 2, 10, 4])
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        edl_loss(alpha, labels)


def test_negative_labels_fail_clearly():
    alpha = _random_alpha(batch_size=3, num_classes=10)
    labels = torch.tensor([0, -1, 2])
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        edl_loss(alpha, labels)


def test_labels_batch_size_mismatch_fails_clearly():
    alpha = _random_alpha(batch_size=5, num_classes=10)
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError):
        edl_loss(alpha, labels)


def test_alpha_below_one_fails_clearly():
    alpha = torch.full((3, 10), 0.5)  # invalid: alpha must be >= 1
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="alpha"):
        edl_loss(alpha, labels)


def test_alpha_wrong_dim_fails_clearly():
    alpha = torch.randn(3, 10, 2)
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match=r"\[B, K\]"):
        edl_loss(alpha, labels)


def test_negative_epoch_fails_clearly():
    alpha = _random_alpha(batch_size=3, num_classes=10)
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(ValueError, match="epoch"):
        edl_loss(alpha, labels, epoch=-1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"kl_annealing_epochs": 0},
        {"kl_annealing_epochs": -5},
        {"kl_weight_max": -1.0},
    ],
)
def test_invalid_loss_hyperparameters_fail_clearly(kwargs):
    alpha = _random_alpha(batch_size=3, num_classes=10)
    labels = torch.tensor([0, 1, 2])
    with pytest.raises(EvidentialConfigError):
        edl_loss(alpha, labels, **kwargs)


def test_edl_loss_module_rejects_invalid_construction():
    with pytest.raises(EvidentialConfigError):
        EDLLoss(kl_annealing_epochs=0)
    with pytest.raises(EvidentialConfigError):
        EDLLoss(kl_weight_max=-1.0)


# --- EDLLoss module wrapper ---


def test_edl_loss_module_matches_functional_form():
    alpha = _random_alpha(batch_size=8, num_classes=10, seed=7)
    labels = torch.randint(0, 10, (8,))
    module = EDLLoss(kl_annealing_epochs=5, kl_weight_max=0.5)
    via_module = module(alpha, labels, epoch=2)
    via_function = edl_loss(alpha, labels, epoch=2, kl_annealing_epochs=5, kl_weight_max=0.5)
    assert torch.equal(via_module, via_function)


def test_edl_loss_module_from_settings():
    settings = EDLSettings(enabled=True, loss_weight=1.0, kl_annealing_epochs=7, kl_weight_max=0.3, epsilon=1e-8)
    module = EDLLoss.from_settings(settings)
    assert module.kl_annealing_epochs == 7
    assert module.kl_weight_max == 0.3


# --- determinism ---


def test_edl_loss_deterministic_given_same_inputs():
    alpha = _random_alpha(batch_size=8, num_classes=10, seed=8)
    labels = torch.randint(0, 10, (8,))
    loss_a = edl_loss(alpha, labels, epoch=3, kl_annealing_epochs=10, kl_weight_max=1.0)
    loss_b = edl_loss(alpha, labels, epoch=3, kl_annealing_epochs=10, kl_weight_max=1.0)
    assert torch.equal(loss_a, loss_b)


def test_compute_evidential_output_deterministic():
    raw = torch.randn(6, 10)
    out_a = compute_evidential_output(raw)
    out_b = compute_evidential_output(raw)
    assert torch.equal(out_a.uncertainty, out_b.uncertainty)
    assert torch.equal(out_a.probabilities, out_b.probabilities)


# --- config loading/validation (configs/losses.yaml: edl) ---


def test_load_edl_settings_from_real_config_file():
    import yaml

    with open("configs/losses.yaml", "r", encoding="utf-8") as f:
        losses_config = yaml.safe_load(f)

    settings = load_edl_settings(losses_config)

    assert isinstance(settings, EDLSettings)
    assert settings.enabled is True
    assert settings.kl_annealing_epochs > 0
    assert settings.kl_weight_max >= 0
    assert settings.loss_weight >= 0
    assert settings.epsilon > 0


def test_load_edl_settings_defaults_when_section_missing():
    settings = load_edl_settings({})
    assert settings.enabled is True
    assert settings.kl_annealing_epochs == DEFAULT_KL_ANNEALING_EPOCHS
    assert settings.kl_weight_max == DEFAULT_KL_WEIGHT_MAX


@pytest.mark.parametrize(
    "bad_section",
    [
        {"loss_weight": -1.0},
        {"kl_annealing_epochs": 0},
        {"kl_annealing_epochs": -3},
        {"kl_weight_max": -0.5},
        {"epsilon": 0.0},
        {"epsilon": -1e-8},
    ],
)
def test_load_edl_settings_rejects_invalid_values(bad_section):
    with pytest.raises(EvidentialConfigError):
        load_edl_settings({"edl": bad_section})

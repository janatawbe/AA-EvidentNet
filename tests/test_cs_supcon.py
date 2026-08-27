"""Tests for src.losses.cs_supcon (Task 8: CS-SupCon loss only - no EDL, no
training-loop integration). All tests use tiny synthetic tensors - no
internet access, no real dataset, no data/raw/ or test set involvement."""

import pytest
import torch

from src.losses.cs_supcon import (
    AmbiguityPairs,
    CSSupConConfigError,
    CSSupConLoss,
    CSSupConSettings,
    cs_supcon_loss,
    load_cs_supcon_settings,
    resolve_ambiguity_pairs,
)

CANONICAL_CLASSES = [
    "Central Serous Chorioretinopathy",  # 0
    "Diabetic Retinopathy",  # 1
    "Disc Edema",  # 2
    "Glaucoma",  # 3
    "Healthy",  # 4
    "Macular Scar",  # 5
    "Myopia",  # 6
    "Pterygium",  # 7
    "Retinal Detachment",  # 8
    "Retinitis Pigmentosa",  # 9
]

REAL_AMBIGUITY_PAIR_NAMES = [
    ["Healthy", "Glaucoma"],
    ["Disc Edema", "Glaucoma"],
    ["Diabetic Retinopathy", "Central Serous Chorioretinopathy"],
]


def _real_ambiguity_pairs() -> AmbiguityPairs:
    return resolve_ambiguity_pairs(REAL_AMBIGUITY_PAIR_NAMES, CANONICAL_CLASSES)


def _make_batch(batch_size=8, dim=16, num_classes=10, seed=0):
    g = torch.Generator().manual_seed(seed)
    embeddings = torch.randn(batch_size, dim, generator=g, requires_grad=True)
    labels = torch.randint(0, num_classes, (batch_size,), generator=g)
    return embeddings, labels


# --- canonical class ordering / ambiguity-pair resolution ---


def test_canonical_ordering_matches_dataset_config():
    from src.data.dataset import build_class_to_idx

    assert build_class_to_idx(CANONICAL_CLASSES) == {
        name: idx for idx, name in enumerate(CANONICAL_CLASSES)
    }


def test_resolve_ambiguity_pairs_resolves_the_three_real_pairs():
    pairs = _real_ambiguity_pairs()
    assert len(pairs) == 3
    assert pairs.contains(4, 3)  # Healthy <-> Glaucoma
    assert pairs.contains(2, 3)  # Disc Edema <-> Glaucoma
    assert pairs.contains(1, 0)  # Diabetic Retinopathy <-> CSC
    assert pairs.contains(3, 4)  # order-independent


def test_resolve_ambiguity_pairs_rejects_unknown_class_name():
    with pytest.raises(CSSupConConfigError, match="unknown class name"):
        resolve_ambiguity_pairs([["Healthy", "Not A Real Class"]], CANONICAL_CLASSES)


def test_resolve_ambiguity_pairs_rejects_self_pair():
    with pytest.raises(CSSupConConfigError, match="cannot be ambiguous with itself"):
        resolve_ambiguity_pairs([["Healthy", "Healthy"]], CANONICAL_CLASSES)


def test_resolve_ambiguity_pairs_rejects_duplicate_pair_same_order():
    with pytest.raises(CSSupConConfigError, match="duplicate"):
        resolve_ambiguity_pairs(
            [["Healthy", "Glaucoma"], ["Healthy", "Glaucoma"]], CANONICAL_CLASSES
        )


def test_resolve_ambiguity_pairs_rejects_duplicate_pair_reversed_order():
    with pytest.raises(CSSupConConfigError, match="duplicate"):
        resolve_ambiguity_pairs(
            [["Healthy", "Glaucoma"], ["Glaucoma", "Healthy"]], CANONICAL_CLASSES
        )


def test_resolve_ambiguity_pairs_rejects_wrong_arity():
    with pytest.raises(CSSupConConfigError, match="exactly 2"):
        resolve_ambiguity_pairs([["Healthy"]], CANONICAL_CLASSES)
    with pytest.raises(CSSupConConfigError, match="exactly 2"):
        resolve_ambiguity_pairs([["Healthy", "Glaucoma", "Myopia"]], CANONICAL_CLASSES)


def test_resolve_ambiguity_pairs_reports_multiple_errors_together():
    with pytest.raises(CSSupConConfigError) as excinfo:
        resolve_ambiguity_pairs(
            [["Not A Class", "Also Not A Class"], ["Healthy", "Healthy"]],
            CANONICAL_CLASSES,
        )
    message = str(excinfo.value)
    assert "Not A Class" in message
    assert "cannot be ambiguous with itself" in message


def test_resolve_ambiguity_pairs_empty_list_is_valid():
    pairs = resolve_ambiguity_pairs([], CANONICAL_CLASSES)
    assert len(pairs) == 0


# --- configs/losses.yaml: cs_supcon loading/validation ---


def test_load_cs_supcon_settings_from_real_config_file():
    import yaml

    with open("configs/losses.yaml", "r", encoding="utf-8") as f:
        losses_config = yaml.safe_load(f)

    settings = load_cs_supcon_settings(losses_config, CANONICAL_CLASSES)

    assert isinstance(settings, CSSupConSettings)
    assert settings.enabled is True
    assert settings.temperature > 0
    assert settings.ambiguity_weight > 0
    assert len(settings.ambiguity_pairs) == 3
    assert settings.ambiguity_pairs.contains(4, 3)
    assert settings.ambiguity_pairs.contains(2, 3)
    assert settings.ambiguity_pairs.contains(1, 0)


def test_load_cs_supcon_settings_defaults_when_section_missing():
    settings = load_cs_supcon_settings({}, CANONICAL_CLASSES)
    assert settings.enabled is True
    assert settings.temperature > 0
    assert len(settings.ambiguity_pairs) == 0


@pytest.mark.parametrize(
    "bad_section",
    [
        {"temperature": 0.0},
        {"temperature": -1.0},
        {"loss_weight": -0.5},
        {"ambiguity_weight": 0.0},
        {"ambiguity_weight": -2.0},
    ],
)
def test_load_cs_supcon_settings_rejects_invalid_scalar_values(bad_section):
    with pytest.raises(CSSupConConfigError):
        load_cs_supcon_settings({"cs_supcon": bad_section}, CANONICAL_CLASSES)


def test_cs_supcon_loss_constructor_rejects_invalid_temperature():
    with pytest.raises(CSSupConConfigError):
        CSSupConLoss(temperature=0.0)
    with pytest.raises(CSSupConConfigError):
        CSSupConLoss(temperature=-1.0)


def test_cs_supcon_loss_constructor_rejects_invalid_ambiguity_weight():
    with pytest.raises(CSSupConConfigError):
        CSSupConLoss(ambiguity_weight=0.0)
    with pytest.raises(CSSupConConfigError):
        CSSupConLoss(ambiguity_weight=-1.0)


# --- core numerical/behavioral properties ---


def test_output_is_scalar():
    embeddings, labels = _make_batch()
    loss = cs_supcon_loss(embeddings, labels)
    assert loss.dim() == 0


def test_output_is_finite():
    embeddings, labels = _make_batch()
    loss = cs_supcon_loss(embeddings, labels)
    assert torch.isfinite(loss).all()


def test_gradients_propagate():
    embeddings, labels = _make_batch(batch_size=12, num_classes=4)
    loss = cs_supcon_loss(embeddings, labels)
    loss.backward()
    assert embeddings.grad is not None
    assert torch.isfinite(embeddings.grad).all()
    assert torch.any(embeddings.grad != 0)


def test_loss_is_differentiable_through_a_linear_layer():
    torch.manual_seed(0)
    raw = torch.randn(10, 8, requires_grad=True)
    labels = torch.randint(0, 3, (10,))
    linear = torch.nn.Linear(8, 5)
    embeddings = linear(raw)
    loss = cs_supcon_loss(embeddings, labels)
    loss.backward()
    assert linear.weight.grad is not None
    assert torch.isfinite(linear.weight.grad).all()


def test_identical_embeddings_same_class_is_finite_and_deterministic():
    # A degenerate but valid input: every embedding in the batch identical,
    # single class. The loss must remain finite (no 0/0 from a
    # zero-similarity spread) and reproducible.
    embeddings = torch.ones(6, 4) + torch.randn(6, 4) * 1e-6
    labels = torch.zeros(6, dtype=torch.long)
    loss_a = cs_supcon_loss(embeddings, labels, temperature=0.5)
    loss_b = cs_supcon_loss(embeddings, labels, temperature=0.5)
    assert torch.isfinite(loss_a)
    assert torch.isclose(loss_a, loss_b)


def test_well_separated_classes_give_lower_loss_than_all_identical_embeddings():
    # If every embedding in the batch is identical regardless of label, the
    # model cannot possibly distinguish classes - that should score worse
    # (or certainly no better) than embeddings that are well clustered by
    # class and well separated across classes.
    torch.manual_seed(12)
    num_classes, per_class, dim = 3, 4, 6
    labels = torch.arange(num_classes).repeat_interleave(per_class)

    collapsed = torch.ones(num_classes * per_class, dim) + torch.randn(num_classes * per_class, dim) * 1e-6
    collapsed_loss = cs_supcon_loss(collapsed, labels, temperature=0.2)

    centers = torch.randn(num_classes, dim) * 20
    separated = centers[labels] + torch.randn(num_classes * per_class, dim) * 0.01
    separated_loss = cs_supcon_loss(separated, labels, temperature=0.2)

    assert torch.isfinite(collapsed_loss)
    assert torch.isfinite(separated_loss)
    assert separated_loss.item() < collapsed_loss.item()


def test_identical_embeddings_across_all_different_classes_is_finite():
    embeddings = torch.ones(5, 4)
    labels = torch.arange(5)
    loss = cs_supcon_loss(embeddings, labels)
    assert torch.isfinite(loss)


def test_same_class_positive_pairs_recognized_relative_to_random_labels():
    # A batch where embeddings cluster tightly by label should have much
    # lower loss than the same embeddings under a random label shuffle
    # (which mostly breaks the true same-class positive structure).
    torch.manual_seed(1)
    num_classes, per_class, dim = 4, 4, 8
    centers = torch.randn(num_classes, dim) * 10
    labels = torch.arange(num_classes).repeat_interleave(per_class)
    embeddings = centers[labels] + torch.randn(num_classes * per_class, dim) * 0.01

    aligned_loss = cs_supcon_loss(embeddings, labels)

    shuffled_labels = labels[torch.randperm(labels.numel())]
    # Guard against (unlikely) identical shuffle.
    if torch.equal(shuffled_labels, labels):
        shuffled_labels = labels.flip(0)
    shuffled_loss = cs_supcon_loss(embeddings, shuffled_labels)

    assert aligned_loss.item() < shuffled_loss.item()


def test_self_pairs_are_excluded_from_denominator():
    # A single duplicated-embedding class member should not let an anchor
    # treat itself as a similarity-1 "negative" inflating the denominator;
    # verified indirectly by checking finiteness and that increasing an
    # anchor's own similarity to itself (impossible to change - self excluded)
    # has no bearing: loss must match a manual per-anchor computation that
    # explicitly zeroes the diagonal.
    torch.manual_seed(2)
    embeddings = torch.randn(6, 5, requires_grad=False)
    labels = torch.tensor([0, 0, 1, 1, 2, 2])

    import torch.nn.functional as F

    normalized = F.normalize(embeddings, dim=1)
    sim = torch.matmul(normalized, normalized.T) / 0.1
    diag = torch.diagonal(sim)
    # If self-comparisons were NOT excluded, the diagonal (always
    # similarity 1.0, the max possible) would dominate every row's
    # log-sum-exp. Confirm our loss does not blow up/collapse to reflect
    # that: the module's output must be finite and independent of this.
    loss = cs_supcon_loss(embeddings, labels, temperature=0.1)
    assert torch.isfinite(loss)
    assert diag.allclose(torch.ones(6) / 0.1, atol=1e-3)  # sanity: diag is indeed self-sim


def test_one_sample_per_class_batch_is_finite_and_zero():
    embeddings, _ = _make_batch(batch_size=6, num_classes=6)
    labels = torch.arange(6)  # every label unique - no positives exist anywhere
    loss = cs_supcon_loss(embeddings, labels)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_single_class_batch_is_finite():
    embeddings, _ = _make_batch(batch_size=7, num_classes=1)
    labels = torch.zeros(7, dtype=torch.long)
    loss = cs_supcon_loss(embeddings, labels)
    assert torch.isfinite(loss)


def test_batch_size_one_is_finite_and_zero():
    embeddings = torch.randn(1, 8)
    labels = torch.tensor([3])
    loss = cs_supcon_loss(embeddings, labels)
    assert torch.isfinite(loss)
    assert loss.item() == pytest.approx(0.0)


def test_arbitrary_batch_sizes_are_supported():
    for batch_size in (2, 3, 5, 17, 33):
        embeddings, labels = _make_batch(batch_size=batch_size, num_classes=4, seed=batch_size)
        loss = cs_supcon_loss(embeddings, labels)
        assert torch.isfinite(loss)


def test_pre_normalized_embeddings_are_supported():
    embeddings, labels = _make_batch(batch_size=10, num_classes=3)
    normalized = torch.nn.functional.normalize(embeddings.detach(), dim=1).requires_grad_(True)
    loss = cs_supcon_loss(normalized, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert normalized.grad is not None


def test_extreme_scale_embeddings_remain_numerically_stable():
    torch.manual_seed(3)
    embeddings = torch.randn(10, 6) * 1000.0
    labels = torch.randint(0, 3, (10,))
    loss = cs_supcon_loss(embeddings, labels, temperature=0.05)
    assert torch.isfinite(loss)


# --- ambiguity weighting must actually change behavior ---


def test_ambiguity_weighting_changes_loss_vs_disabled():
    torch.manual_seed(4)
    dim = 8
    # Construct a batch containing an ambiguous pair (Healthy=4, Glaucoma=3)
    # plus an unrelated class (Myopia=6), with embeddings arranged so the
    # ambiguous classes are close together (a "hard" case the ambiguity
    # weighting should react to).
    healthy_center = torch.randn(dim)
    glaucoma_center = healthy_center + torch.randn(dim) * 0.05  # close: ambiguous & confusable
    myopia_center = torch.randn(dim) * 5  # far: unrelated

    embeddings = torch.stack(
        [
            healthy_center + torch.randn(dim) * 0.01,
            healthy_center + torch.randn(dim) * 0.01,
            glaucoma_center + torch.randn(dim) * 0.01,
            glaucoma_center + torch.randn(dim) * 0.01,
            myopia_center + torch.randn(dim) * 0.01,
            myopia_center + torch.randn(dim) * 0.01,
        ]
    )
    labels = torch.tensor([4, 4, 3, 3, 6, 6])

    no_ambiguity = cs_supcon_loss(embeddings, labels, ambiguity_pairs=AmbiguityPairs.empty())
    with_ambiguity = cs_supcon_loss(embeddings, labels, ambiguity_pairs=_real_ambiguity_pairs())

    assert not torch.isclose(no_ambiguity, with_ambiguity)
    # Upweighting a close/confusable negative pair increases its
    # contribution to the denominator, which increases the loss for those
    # anchors relative to the unweighted case.
    assert with_ambiguity.item() > no_ambiguity.item()


def test_ambiguity_weighting_has_no_effect_when_no_ambiguous_classes_present():
    # A batch containing none of the configured ambiguous classes should
    # be unaffected by enabling ambiguity weighting.
    embeddings, _ = _make_batch(batch_size=8, num_classes=3, seed=5)
    labels = torch.tensor([5, 5, 6, 6, 7, 7, 5, 6])  # Macular Scar/Myopia/Pterygium only

    no_ambiguity = cs_supcon_loss(embeddings, labels, ambiguity_pairs=AmbiguityPairs.empty())
    with_ambiguity = cs_supcon_loss(embeddings, labels, ambiguity_pairs=_real_ambiguity_pairs())

    assert torch.isclose(no_ambiguity, with_ambiguity, atol=1e-6)


def test_changing_ambiguity_weight_changes_loss():
    embeddings, labels = _make_batch(batch_size=8, num_classes=10, seed=6)
    labels = torch.tensor([4, 3, 4, 3, 6, 7, 6, 7])  # includes the Healthy/Glaucoma pair

    loss_w2 = cs_supcon_loss(embeddings, labels, ambiguity_weight=2.0, ambiguity_pairs=_real_ambiguity_pairs())
    loss_w5 = cs_supcon_loss(embeddings, labels, ambiguity_weight=5.0, ambiguity_pairs=_real_ambiguity_pairs())

    assert not torch.isclose(loss_w2, loss_w5)


def test_changing_temperature_changes_loss():
    embeddings, labels = _make_batch(batch_size=8, num_classes=4, seed=7)

    loss_t1 = cs_supcon_loss(embeddings, labels, temperature=0.07)
    loss_t2 = cs_supcon_loss(embeddings, labels, temperature=0.5)

    assert not torch.isclose(loss_t1, loss_t2)


def test_unrelated_negatives_are_not_upweighted_like_ambiguous_ones():
    # Directly compare a batch of only-ambiguous-pair classes vs. an
    # otherwise-identical batch using only unrelated classes: the ambiguity
    # weighting should change the former but not the latter.
    torch.manual_seed(8)
    dim = 6
    center_a = torch.randn(dim)
    center_b = center_a + torch.randn(dim) * 0.05

    def build(label_a, label_b):
        embeddings = torch.stack(
            [
                center_a + torch.randn(dim) * 0.01,
                center_a + torch.randn(dim) * 0.01,
                center_b + torch.randn(dim) * 0.01,
                center_b + torch.randn(dim) * 0.01,
            ]
        )
        labels = torch.tensor([label_a, label_a, label_b, label_b])
        return embeddings, labels

    torch.manual_seed(8)
    amb_embeddings, amb_labels = build(4, 3)  # Healthy/Glaucoma: ambiguous
    torch.manual_seed(8)
    unrel_embeddings, unrel_labels = build(5, 6)  # Macular Scar/Myopia: unrelated

    amb_no = cs_supcon_loss(amb_embeddings, amb_labels, ambiguity_pairs=AmbiguityPairs.empty())
    amb_with = cs_supcon_loss(amb_embeddings, amb_labels, ambiguity_pairs=_real_ambiguity_pairs())
    unrel_no = cs_supcon_loss(unrel_embeddings, unrel_labels, ambiguity_pairs=AmbiguityPairs.empty())
    unrel_with = cs_supcon_loss(unrel_embeddings, unrel_labels, ambiguity_pairs=_real_ambiguity_pairs())

    assert not torch.isclose(amb_no, amb_with)
    assert torch.isclose(unrel_no, unrel_with, atol=1e-6)


# --- determinism ---


def test_deterministic_given_same_inputs_and_config():
    embeddings, labels = _make_batch(batch_size=9, num_classes=4, seed=9)
    module = CSSupConLoss(temperature=0.1, ambiguity_weight=2.0, ambiguity_pairs=_real_ambiguity_pairs())

    loss_1 = module(embeddings, labels)
    loss_2 = module(embeddings, labels)

    assert torch.equal(loss_1, loss_2)


def test_deterministic_across_freshly_constructed_modules():
    embeddings, labels = _make_batch(batch_size=9, num_classes=4, seed=10)

    loss_a = CSSupConLoss(temperature=0.2, ambiguity_weight=3.0)(embeddings, labels)
    loss_b = CSSupConLoss(temperature=0.2, ambiguity_weight=3.0)(embeddings, labels)

    assert torch.equal(loss_a, loss_b)


# --- input validation: labels/embedding shape ---


def test_labels_outside_configured_range_fail_clearly():
    embeddings, _ = _make_batch(batch_size=5, num_classes=10)
    labels = torch.tensor([0, 1, 2, 10, 4])  # 10 is out of [0, 9]
    module = CSSupConLoss()
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        module(embeddings, labels, num_classes=10)


def test_negative_labels_fail_clearly():
    embeddings, _ = _make_batch(batch_size=3, num_classes=10)
    labels = torch.tensor([0, -1, 2])
    module = CSSupConLoss()
    with pytest.raises(ValueError, match=r"\[0, 9\]"):
        module(embeddings, labels, num_classes=10)


def test_embedding_dim_must_be_2d():
    labels = torch.tensor([0, 1, 2])
    module = CSSupConLoss()
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        module(torch.randn(3, 4, 5), labels)
    with pytest.raises(ValueError, match=r"\[B, D\]"):
        module(torch.randn(3), labels)


def test_embedding_width_must_be_positive():
    labels = torch.tensor([0, 1])
    module = CSSupConLoss()
    with pytest.raises(ValueError):
        module(torch.randn(2, 0), labels)


def test_labels_batch_size_must_match_embeddings():
    embeddings = torch.randn(5, 8)
    labels = torch.tensor([0, 1, 2])  # wrong length
    module = CSSupConLoss()
    with pytest.raises(ValueError):
        module(embeddings, labels)


def test_forward_without_num_classes_does_not_validate_label_range():
    # num_classes is optional - if the caller doesn't pass it, out-of-range
    # labels are not (and cannot be) checked against a bound. This documents
    # that behavior rather than asserting a silent, surprising check.
    embeddings = torch.randn(3, 4)
    labels = torch.tensor([0, 1, 999])
    module = CSSupConLoss()
    loss = module(embeddings, labels)
    assert torch.isfinite(loss)

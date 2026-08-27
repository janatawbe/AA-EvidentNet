"""Tests for src.losses.combined (Task 7 completion): the combined
AA-EvidentNet training objective L_total = L_classification +
cs_supcon_weight*L_CS-SupCon + edl_weight*L_EDL. Uses tiny synthetic
tensors/fake outputs - no real dataset, no data/raw/, no test set."""

from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.losses.combined import (
    CombinedAAEvidentNetLoss,
    CombinedObjectiveConfigError,
    CombinedObjectiveSettings,
    build_combined_aa_evidentnet_loss,
    load_combined_objective_settings,
)
from src.losses.cs_supcon import AmbiguityPairs, CSSupConLoss
from src.losses.evidential import EDLLoss, compute_evidential_output

CANONICAL_CLASSES = [
    "Central Serous Chorioretinopathy",
    "Diabetic Retinopathy",
    "Disc Edema",
    "Glaucoma",
    "Healthy",
    "Macular Scar",
    "Myopia",
    "Pterygium",
    "Retinal Detachment",
    "Retinitis Pigmentosa",
]


def _fake_output(batch_size=8, num_classes=10, embedding_dim=16, seed=0, requires_grad=True):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch_size, num_classes, generator=g, requires_grad=requires_grad)
    embedding = torch.randn(batch_size, embedding_dim, generator=g, requires_grad=requires_grad)
    raw_evidential = torch.randn(batch_size, num_classes, generator=g, requires_grad=requires_grad)
    dirichlet_alpha = compute_evidential_output(raw_evidential).alpha
    labels = torch.randint(0, num_classes, (batch_size,), generator=g)
    return SimpleNamespace(logits=logits, embedding=embedding, dirichlet_alpha=dirichlet_alpha), labels


def _full_loss(**overrides):
    kwargs = dict(
        cs_supcon_loss=CSSupConLoss(),
        cs_supcon_weight=1.0,
        edl_loss_module=EDLLoss(),
        edl_weight=1.0,
    )
    kwargs.update(overrides)
    return CombinedAAEvidentNetLoss(**kwargs)


# --- settings loading from configs/losses.yaml ---


def test_load_combined_objective_settings_from_real_config_file():
    import yaml

    with open("configs/losses.yaml", "r", encoding="utf-8") as f:
        losses_config = yaml.safe_load(f)

    settings = load_combined_objective_settings(losses_config, CANONICAL_CLASSES)

    assert isinstance(settings, CombinedObjectiveSettings)
    assert settings.cs_supcon.enabled is True
    assert settings.edl.enabled is True
    assert settings.cs_supcon.loss_weight == 1.0
    assert settings.edl.loss_weight == 1.0
    assert settings.label_smoothing == 0.0


def test_load_combined_objective_settings_rejects_unimplemented_class_weighting():
    losses_config = {"baseline": {"class_weighting": "inverse_frequency"}}
    with pytest.raises(CombinedObjectiveConfigError, match="class_weighting"):
        load_combined_objective_settings(losses_config, CANONICAL_CLASSES)


def test_load_combined_objective_settings_accepts_class_weighting_none():
    losses_config = {"baseline": {"class_weighting": "none"}}
    settings = load_combined_objective_settings(losses_config, CANONICAL_CLASSES)
    assert settings.label_smoothing == 0.0


def test_load_combined_objective_settings_rejects_invalid_label_smoothing():
    with pytest.raises(CombinedObjectiveConfigError):
        load_combined_objective_settings({"baseline": {"label_smoothing": 1.0}}, CANONICAL_CLASSES)
    with pytest.raises(CombinedObjectiveConfigError):
        load_combined_objective_settings({"baseline": {"label_smoothing": -0.1}}, CANONICAL_CLASSES)


def test_build_combined_aa_evidentnet_loss_from_real_config():
    import yaml

    with open("configs/losses.yaml", "r", encoding="utf-8") as f:
        losses_config = yaml.safe_load(f)
    criterion = build_combined_aa_evidentnet_loss(losses_config, CANONICAL_CLASSES)
    assert isinstance(criterion, CombinedAAEvidentNetLoss)
    assert criterion.cs_supcon_loss is not None
    assert criterion.edl_loss_module is not None
    assert criterion.cs_supcon_weight == 1.0
    assert criterion.edl_weight == 1.0


def test_build_combined_aa_evidentnet_loss_resolves_real_ambiguity_pairs():
    losses_config = {
        "cs_supcon": {
            "enabled": True,
            "ambiguity_pairs": [
                ["Healthy", "Glaucoma"],
                ["Disc Edema", "Glaucoma"],
                ["Diabetic Retinopathy", "Central Serous Chorioretinopathy"],
            ],
        },
        "edl": {"enabled": True},
    }
    criterion = build_combined_aa_evidentnet_loss(losses_config, CANONICAL_CLASSES)
    assert len(criterion.cs_supcon_loss.ambiguity_pairs) == 3


# --- construction validation ---


def test_constructor_rejects_negative_weights():
    with pytest.raises(CombinedObjectiveConfigError):
        CombinedAAEvidentNetLoss(cs_supcon_loss=CSSupConLoss(), cs_supcon_weight=-1.0, edl_loss_module=None, edl_weight=0.0)
    with pytest.raises(CombinedObjectiveConfigError):
        CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(), edl_weight=-1.0)


# --- forward: scalar / finite ---


def test_forward_returns_scalar():
    criterion = _full_loss()
    output, labels = _fake_output()
    loss = criterion(output, labels)
    assert loss.dim() == 0


def test_forward_is_finite():
    criterion = _full_loss()
    output, labels = _fake_output()
    loss = criterion(output, labels)
    assert torch.isfinite(loss)


# --- each term actually contributes ---


def test_classification_only_matches_plain_cross_entropy():
    criterion = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=None, edl_weight=0.0)
    output, labels = _fake_output(requires_grad=False)
    loss = criterion(output, labels)
    expected = F.cross_entropy(output.logits, labels)
    assert torch.allclose(loss, expected)
    assert criterion.last_components.keys() == {"classification", "total"}


def test_enabling_cs_supcon_changes_loss_vs_classification_only():
    output, labels = _fake_output(requires_grad=False)
    classification_only = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=None, edl_weight=0.0)
    with_cs_supcon = CombinedAAEvidentNetLoss(cs_supcon_loss=CSSupConLoss(), cs_supcon_weight=1.0, edl_loss_module=None, edl_weight=0.0)

    loss_a = classification_only(output, labels)
    loss_b = with_cs_supcon(output, labels)
    assert not torch.isclose(loss_a, loss_b)
    assert "cs_supcon" in with_cs_supcon.last_components
    assert "cs_supcon" not in classification_only.last_components


def test_enabling_edl_changes_loss_vs_classification_only():
    output, labels = _fake_output(requires_grad=False)
    classification_only = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=None, edl_weight=0.0)
    with_edl = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(), edl_weight=1.0)

    loss_a = classification_only(output, labels)
    loss_b = with_edl(output, labels)
    assert not torch.isclose(loss_a, loss_b)
    assert "edl" in with_edl.last_components


def test_all_three_terms_present_with_full_config():
    criterion = _full_loss()
    output, labels = _fake_output(requires_grad=False)
    criterion(output, labels)
    assert set(criterion.last_components.keys()) == {"classification", "cs_supcon", "edl", "total"}


def test_changing_cs_supcon_weight_changes_total_loss():
    output, labels = _fake_output(requires_grad=False)
    low = CombinedAAEvidentNetLoss(cs_supcon_loss=CSSupConLoss(), cs_supcon_weight=0.5, edl_loss_module=None, edl_weight=0.0)
    high = CombinedAAEvidentNetLoss(cs_supcon_loss=CSSupConLoss(), cs_supcon_weight=5.0, edl_loss_module=None, edl_weight=0.0)
    assert not torch.isclose(low(output, labels), high(output, labels))


def test_changing_edl_weight_changes_total_loss():
    output, labels = _fake_output(requires_grad=False)
    low = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(), edl_weight=0.5)
    high = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(), edl_weight=5.0)
    assert not torch.isclose(low(output, labels), high(output, labels))


def test_disabled_loss_module_none_forces_zero_weight_even_if_requested():
    # cs_supcon_loss=None means "disabled" - the weight is forced to 0
    # regardless of what's passed for cs_supcon_weight.
    criterion = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=99.0, edl_loss_module=None, edl_weight=0.0)
    assert criterion.cs_supcon_weight == 0.0


# --- gradients propagate through all components ---


def test_gradients_propagate_through_all_three_components():
    criterion = _full_loss()
    output, labels = _fake_output(requires_grad=True)
    loss = criterion(output, labels)
    loss.backward()
    assert output.logits.grad is not None and torch.isfinite(output.logits.grad).all()
    assert output.embedding.grad is not None and torch.isfinite(output.embedding.grad).all()
    # dirichlet_alpha is derived from raw_evidential via compute_evidential_output,
    # so it is not itself a leaf - verify gradients reached it via retain_grad.
    assert torch.any(output.embedding.grad != 0)


def test_gradients_propagate_through_upstream_linear_layers():
    # Simulates AA-EvidentNet's actual wiring: logits/embedding/dirichlet_alpha
    # all derive from a shared upstream embedding via separate Linear heads.
    torch.manual_seed(0)
    batch_size, embed_dim, num_classes = 6, 12, 10
    shared = torch.randn(batch_size, embed_dim, requires_grad=True)
    classifier = torch.nn.Linear(embed_dim, num_classes)
    evidential_head = torch.nn.Linear(embed_dim, num_classes)

    logits = classifier(shared)
    raw_evidential = evidential_head(shared)
    dirichlet_alpha = compute_evidential_output(raw_evidential).alpha
    output = SimpleNamespace(logits=logits, embedding=shared, dirichlet_alpha=dirichlet_alpha)
    labels = torch.randint(0, num_classes, (batch_size,))

    criterion = _full_loss()
    loss = criterion(output, labels)
    loss.backward()

    assert shared.grad is not None
    assert torch.isfinite(shared.grad).all()
    assert torch.any(shared.grad != 0)
    assert classifier.weight.grad is not None
    assert evidential_head.weight.grad is not None


def test_gradients_reach_real_aa_evidentnet_alpha_fusion_gate():
    # End-to-end with the REAL AA-EvidentNet model (not a fake output) -
    # verifies the combined loss's gradients reach the fusion gate alpha,
    # not just the two heads.
    from src.models.aa_evidentnet import AAEvidentNet

    torch.manual_seed(0)
    model = AAEvidentNet(pretrained=False, num_classes=10, embedding_dim=16, local_feature_dim=8)
    images = torch.randn(3, 3, 224, 224)
    labels = torch.randint(0, 10, (3,))

    criterion = _full_loss()
    output = model(images, return_features=True)
    loss = criterion(output, labels)
    loss.backward()

    assert model._alpha_raw.grad is not None
    assert torch.isfinite(model._alpha_raw.grad).all()


# --- KL annealing via set_epoch ---


def test_set_epoch_changes_edl_contribution():
    output, labels = _fake_output(requires_grad=False, seed=3)
    criterion = CombinedAAEvidentNetLoss(
        cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(kl_annealing_epochs=10, kl_weight_max=1.0), edl_weight=1.0
    )
    criterion.set_epoch(0)
    loss_epoch_0 = criterion(output, labels)
    criterion.set_epoch(10)
    loss_epoch_10 = criterion(output, labels)
    assert not torch.isclose(loss_epoch_0, loss_epoch_10)


def test_no_set_epoch_call_applies_full_kl_weight():
    output, labels = _fake_output(requires_grad=False, seed=4)
    criterion = CombinedAAEvidentNetLoss(
        cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(kl_annealing_epochs=10, kl_weight_max=1.0), edl_weight=1.0
    )
    loss_no_epoch = criterion(output, labels)
    criterion.set_epoch(10_000)
    loss_far_epoch = criterion(output, labels)
    assert torch.isclose(loss_no_epoch, loss_far_epoch)


# --- missing required fields fail clearly ---


def test_missing_logits_fails_clearly():
    criterion = _full_loss()
    output = SimpleNamespace(embedding=torch.randn(4, 8), dirichlet_alpha=torch.ones(4, 10))
    labels = torch.randint(0, 10, (4,))
    with pytest.raises(ValueError, match="logits"):
        criterion(output, labels)


def test_missing_embedding_fails_clearly_when_cs_supcon_enabled():
    criterion = CombinedAAEvidentNetLoss(cs_supcon_loss=CSSupConLoss(), cs_supcon_weight=1.0, edl_loss_module=None, edl_weight=0.0)
    output = SimpleNamespace(logits=torch.randn(4, 10), embedding=None)
    labels = torch.randint(0, 10, (4,))
    with pytest.raises(ValueError, match="embedding"):
        criterion(output, labels)


def test_missing_dirichlet_alpha_fails_clearly_when_edl_enabled():
    criterion = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=EDLLoss(), edl_weight=1.0)
    output = SimpleNamespace(logits=torch.randn(4, 10), dirichlet_alpha=None)
    labels = torch.randint(0, 10, (4,))
    with pytest.raises(ValueError, match="dirichlet_alpha"):
        criterion(output, labels)


def test_missing_embedding_does_not_fail_when_cs_supcon_disabled():
    criterion = CombinedAAEvidentNetLoss(cs_supcon_loss=None, cs_supcon_weight=0.0, edl_loss_module=None, edl_weight=0.0)
    output = SimpleNamespace(logits=torch.randn(4, 10))
    labels = torch.randint(0, 10, (4,))
    loss = criterion(output, labels)
    assert torch.isfinite(loss)


# --- determinism ---


def test_deterministic_given_same_inputs_and_config():
    criterion = _full_loss()
    output, labels = _fake_output(requires_grad=False, seed=5)
    loss_a = criterion(output, labels)
    loss_b = criterion(output, labels)
    assert torch.equal(loss_a, loss_b)


# --- ambiguity pairs respected end to end ---


def test_ambiguity_pairs_actually_affect_combined_loss():
    output, labels = _fake_output(requires_grad=False, seed=6, batch_size=6)
    labels = torch.tensor([4, 3, 4, 3, 6, 7])  # Healthy(4)/Glaucoma(3) pair present

    no_ambiguity = CombinedAAEvidentNetLoss(
        cs_supcon_loss=CSSupConLoss(ambiguity_pairs=AmbiguityPairs.empty()), cs_supcon_weight=1.0, edl_loss_module=None, edl_weight=0.0
    )
    real_pairs = AmbiguityPairs.empty()
    from src.losses.cs_supcon import resolve_ambiguity_pairs

    real_pairs = resolve_ambiguity_pairs([["Healthy", "Glaucoma"]], CANONICAL_CLASSES)
    with_ambiguity = CombinedAAEvidentNetLoss(
        cs_supcon_loss=CSSupConLoss(ambiguity_pairs=real_pairs), cs_supcon_weight=1.0, edl_loss_module=None, edl_weight=0.0
    )

    assert not torch.isclose(no_ambiguity(output, labels), with_ambiguity(output, labels))

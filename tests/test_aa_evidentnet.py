"""Tests for src.models.aa_evidentnet.AAEvidentNet (Task 7: architecture;
Task 9: evidential/Dirichlet uncertainty head integration - CS-SupCon/EDL
LOSSES themselves are tested separately in test_cs_supcon.py/
test_evidential.py, not here). All tests use pretrained=False and
tiny/synthetic tensors - no internet access or pretrained-weight download
required."""

import copy

import pytest
import torch

from src.models.aa_evidentnet import AAEvidentNet, AAEvidentNetOutput, LocalBranch
from src.models.base import ModelOutput, UnknownModelError
from src.models.factory import ModelConfigError, PROPOSED_MODEL_NAMES, create_model

MINI_CONFIG = {
    "num_classes": 10,
    "proposed": {
        "aa_evidentnet": {
            "global_backbone": "maxvit_tiny_tf_224",
            "pretrained": False,
            "num_classes": 10,
            "embedding_dim": 256,
            "local_feature_dim": 128,
            "dropout": 0.0,
        }
    },
}


def _make_model(**overrides):
    kwargs = dict(global_backbone="maxvit_tiny_tf_224", pretrained=False, num_classes=10, embedding_dim=256)
    kwargs.update(overrides)
    return AAEvidentNet(**kwargs)


# --- construction ---


def test_model_constructs_with_defaults():
    model = _make_model()
    assert model.num_classes == 10
    assert model.embedding_dim == 256
    assert model.global_feature_dim > 0
    assert model.local_feature_dim == 128
    assert model.feature_dim == 256  # interface parity with TimmBackboneModel


def test_model_constructs_via_factory():
    model = create_model("aa_evidentnet", MINI_CONFIG)
    assert isinstance(model, AAEvidentNet)
    assert model.num_classes == 10
    assert model.embedding_dim == 256


def test_factory_registers_aa_evidentnet_in_proposed_names():
    assert "aa_evidentnet" in PROPOSED_MODEL_NAMES


def test_unknown_proposed_model_name_fails_clearly():
    # Only names in PROPOSED_MODEL_NAMES ("aa_evidentnet") are dispatched
    # to the proposed-model path; any other unrecognized name falls
    # through to the baseline lookup (the factory's default case), which
    # still fails clearly - this preserves the exact pre-existing baseline
    # error message from Task 5 (tests/test_models.py) unchanged.
    with pytest.raises(ModelConfigError, match="Unknown"):
        create_model("not_a_real_proposed_model", MINI_CONFIG)


def test_missing_proposed_config_entry_fails_clearly():
    config = {"num_classes": 10, "proposed": {}}
    with pytest.raises(ModelConfigError, match="Unknown proposed model name"):
        create_model("aa_evidentnet", config)


def test_unknown_global_backbone_fails_clearly():
    with pytest.raises(UnknownModelError):
        AAEvidentNet(global_backbone="this_is_not_a_real_timm_model_xyz", pretrained=False, num_classes=10)


def test_existing_baseline_error_message_unaffected():
    # Regression guard: extending create_model() for proposed models must
    # not change the existing baseline error message/path.
    with pytest.raises(ModelConfigError, match="Unknown baseline model name"):
        create_model("not_a_real_model", {"num_classes": 10, "baselines": {}})


# --- output shapes ---


def test_forward_default_returns_logits_only():
    model = _make_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert isinstance(logits, torch.Tensor)
    assert logits.shape == (2, 10)


@pytest.mark.parametrize("batch_size", [1, 2, 5, 8])
def test_forward_pass_with_different_batch_sizes(batch_size):
    model = _make_model()
    model.eval()
    x = torch.randn(batch_size, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (batch_size, 10)


def test_return_features_shape_and_type():
    model = _make_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert isinstance(output, AAEvidentNetOutput)
    assert isinstance(output, ModelOutput)  # interface parity with baselines
    assert output.logits.shape == (4, 10)
    assert output.embedding.shape == (4, 256)
    assert output.features.shape == (4, 256)  # features aliases embedding
    assert torch.equal(output.features, output.embedding)


def test_global_and_local_features_available_and_correct_shape():
    model = _make_model(embedding_dim=256)
    model.eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert output.global_feature is not None
    assert output.local_feature is not None
    assert output.global_feature.shape == (3, 256)
    assert output.local_feature.shape == (3, 256)
    # Global and local branches are architecturally distinct - their
    # projected features should not be identical.
    assert not torch.allclose(output.global_feature, output.local_feature)


def test_embedding_dimension_matches_config_default_256():
    model = _make_model(embedding_dim=256)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert output.embedding.shape[1] == 256


def test_embedding_dimension_is_configurable():
    model = _make_model(embedding_dim=64)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert output.embedding.shape[1] == 64
    assert output.global_feature.shape[1] == 64
    assert output.local_feature.shape[1] == 64


def test_num_classes_is_configurable():
    model = _make_model(num_classes=4)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_global_feature_dim_read_from_backbone_not_hardcoded():
    model = _make_model()
    assert model.global_feature_dim == model.global_backbone.num_features


# --- finiteness ---


def test_no_nan_or_inf_in_forward_pass():
    model = _make_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    for tensor in (output.logits, output.embedding, output.global_feature, output.local_feature):
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


def test_invalid_input_dimensions_fail_clearly():
    model = _make_model()
    bad_input = torch.randn(3, 224, 224)  # missing batch dim
    with pytest.raises(ValueError, match="4D"):
        model(bad_input)


# --- adaptive fusion / alpha ---


def test_alpha_initializes_at_one_half():
    model = _make_model()
    assert model.alpha.item() == pytest.approx(0.5)


def test_alpha_is_always_in_unit_interval():
    model = _make_model()
    # +-15 (not +-100): sigmoid(100) rounds to exactly 1.0 in float32
    # precision, which would make this test spuriously check a precision
    # artifact rather than the actual (0, 1) constraint.
    with torch.no_grad():
        model._alpha_raw.fill_(15.0)
    assert 0.0 < model.alpha.item() < 1.0
    assert model.alpha.item() > 0.999
    with torch.no_grad():
        model._alpha_raw.fill_(-15.0)
    assert 0.0 < model.alpha.item() < 1.0
    assert model.alpha.item() < 0.001


def test_alpha_is_a_learnable_parameter():
    model = _make_model()
    param_names = [name for name, _ in model.named_parameters()]
    assert any("alpha" in name for name in param_names)
    assert model._alpha_raw.requires_grad


def test_fusion_formula_matches_alpha_global_plus_one_minus_alpha_local():
    model = _make_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
        expected = model.alpha * output.global_feature + (1 - model.alpha) * output.local_feature
    assert torch.allclose(output.embedding, expected, atol=1e-5)


def test_alpha_near_one_makes_fused_approach_global_feature():
    model = _make_model()
    with torch.no_grad():
        model._alpha_raw.fill_(50.0)  # sigmoid(50) ~= 1.0
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert torch.allclose(output.embedding, output.global_feature, atol=1e-3)


def test_alpha_near_zero_makes_fused_approach_local_feature():
    model = _make_model()
    with torch.no_grad():
        model._alpha_raw.fill_(-50.0)  # sigmoid(-50) ~= 0.0
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert torch.allclose(output.embedding, output.local_feature, atol=1e-3)


def test_alpha_updates_during_training():
    """Verifies alpha is actually part of the computation graph and
    receives gradient updates - not a frozen/detached constant."""
    model = AAEvidentNet(pretrained=False, num_classes=10, embedding_dim=32, local_feature_dim=16)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    x = torch.randn(4, 3, 224, 224)
    labels = torch.randint(0, 10, (4,))

    before = model._alpha_raw.item()
    for _ in range(3):
        optimizer.zero_grad()
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
    after = model._alpha_raw.item()
    assert before != after


# --- LocalBranch in isolation ---


def test_local_branch_output_shape():
    branch = LocalBranch(in_channels=3, out_dim=64)
    x = torch.randn(3, 3, 224, 224)
    out = branch(x)
    assert out.shape == (3, 64)


def test_local_branch_works_on_small_synthetic_images():
    branch = LocalBranch(in_channels=3, out_dim=32)
    x = torch.randn(2, 3, 32, 32)
    out = branch(x)
    assert out.shape == (2, 32)


def test_local_branch_parameter_count_is_reasonable():
    # "Keep this branch computationally reasonable" - it should be a small
    # fraction of the ~30M-parameter MaxViT global backbone.
    branch = LocalBranch(in_channels=3, out_dim=128)
    total = sum(p.numel() for p in branch.parameters())
    assert 0 < total < 500_000


# --- parameter counts / trainability ---


def test_all_parameters_trainable_by_default():
    model = _make_model()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 0
    assert trainable == total


def test_parameter_count_dominated_by_global_backbone():
    model = _make_model()
    global_params = sum(p.numel() for p in model.global_backbone.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    assert global_params / total_params > 0.9  # global MaxViT backbone should dominate


# --- deterministic / reproducible behavior ---


def test_same_seed_produces_identical_outputs():
    torch.manual_seed(123)
    model_a = _make_model()
    torch.manual_seed(123)
    model_b = _make_model()

    x = torch.randn(2, 3, 224, 224)
    model_a.eval()
    model_b.eval()
    with torch.no_grad():
        out_a = model_a(x)
        out_b = model_b(x)
    assert torch.allclose(out_a, out_b, atol=1e-6)


def test_eval_mode_is_deterministic_across_calls():
    model = _make_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        out_1 = model(x)
        out_2 = model(x)
    assert torch.equal(out_1, out_2)


# --- checkpoint/state_dict compatibility (generic nn.Module interface) ---


def test_state_dict_round_trip(tmp_path):
    model = _make_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = model(x)

    path = tmp_path / "aa_evidentnet.pt"
    torch.save(model.state_dict(), path)

    fresh_model = _make_model()
    fresh_model.load_state_dict(torch.load(path, weights_only=True))
    fresh_model.eval()
    with torch.no_grad():
        actual = fresh_model(x)
    assert torch.allclose(expected, actual, atol=1e-6)


def test_alpha_is_included_in_state_dict():
    model = _make_model()
    state = model.state_dict()
    assert any("alpha" in key for key in state.keys())


# --- compatibility with the existing Trainer (Task 6 infrastructure) ---


class _TinyImageLabelDataset(torch.utils.data.Dataset):
    def __init__(self, n=8, num_classes=10, seed=0):
        gen = torch.Generator().manual_seed(seed)
        self.images = torch.randn(n, 3, 224, 224, generator=gen)
        self.labels = torch.randint(0, num_classes, (n,), generator=gen)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return {"image": self.images[idx], "label": int(self.labels[idx])}


def test_aa_evidentnet_trains_via_existing_trainer_without_modification():
    """AA-EvidentNet must be usable by src.training.trainer.Trainer exactly
    like a baseline model - no special-casing required anywhere in the
    training engine."""
    from torch.utils.data import DataLoader

    from src.training.trainer import Trainer, TrainingConfig, build_optimizer, build_scheduler

    model = AAEvidentNet(global_backbone="maxvit_tiny_tf_224", pretrained=False, num_classes=10, embedding_dim=32, local_feature_dim=16)
    train_loader = DataLoader(_TinyImageLabelDataset(8, seed=1), batch_size=4, shuffle=True, drop_last=True)
    val_loader = DataLoader(_TinyImageLabelDataset(4, seed=2), batch_size=4, shuffle=False)

    config = TrainingConfig(epochs=1, mixed_precision=False)
    optimizer = build_optimizer(model, config)
    scheduler = build_scheduler(optimizer, config)

    before_state = copy.deepcopy(model.state_dict())
    trainer = Trainer(model, optimizer, scheduler, train_loader, val_loader, config, device=torch.device("cpu"))
    result = trainer.fit()
    after_state = model.state_dict()

    assert len(result.history) == 1
    assert set(result.history[0].train_metrics.keys()) == {"loss", "accuracy", "macro_f1"}
    assert set(result.history[0].val_metrics.keys()) == {"loss", "accuracy", "macro_f1", "balanced_accuracy"}
    # A real forward/backward/optimizer-step epoch must change weights,
    # including the fusion gate.
    changed = any(not torch.equal(before_state[k], after_state[k]) for k in before_state)
    assert changed
    assert before_state["_alpha_raw"].item() != after_state["_alpha_raw"].item()


def test_aa_evidentnet_checkpoint_via_existing_checkpointing_module(tmp_path):
    from src.training.checkpointing import (
        assert_checkpoint_compatible,
        build_checkpoint,
        load_checkpoint,
        restore_training_state,
        save_checkpoint,
    )

    model = _make_model(embedding_dim=32, local_feature_dim=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, scheduler=scheduler, epoch=0, best_metric=0.1,
        monitor_metric="val_macro_f1", training_config={}, seed=42, model_name="aa_evidentnet",
        architecture=model.architecture, num_classes=10, dataset_manifest_hash="h", git_commit="c",
    )
    path = tmp_path / "aa_evidentnet_checkpoint.pt"
    save_checkpoint(checkpoint, path)

    loaded = load_checkpoint(path)
    assert_checkpoint_compatible(loaded, "aa_evidentnet", 10)  # must not raise

    fresh_model = _make_model(embedding_dim=32, local_feature_dim=16)
    fresh_optimizer = torch.optim.AdamW(fresh_model.parameters(), lr=1e-3)
    state = restore_training_state(loaded, fresh_model, fresh_optimizer)
    assert state["epoch"] == 0
    assert state["best_metric"] == 0.1


# --- evidential/Dirichlet uncertainty head integration (Task 9) ---


def test_return_features_exposes_evidential_outputs():
    model = _make_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert output.evidence is not None
    assert output.dirichlet_alpha is not None
    assert output.probabilities is not None
    assert output.uncertainty is not None
    assert output.evidential_raw is not None


def test_evidential_output_shapes():
    model = _make_model(num_classes=10)
    model.eval()
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert output.logits.shape == (5, 10)
    assert output.embedding.shape == (5, 256)
    assert output.evidential_raw.shape == (5, 10)
    assert output.evidence.shape == (5, 10)
    assert output.dirichlet_alpha.shape == (5, 10)
    assert output.probabilities.shape == (5, 10)
    assert output.uncertainty.shape == (5,)


def test_evidential_head_uses_a_separate_linear_layer_from_the_classifier():
    model = _make_model()
    assert model.evidential_head.linear is not model.classifier
    assert not torch.equal(model.evidential_head.linear.weight, model.classifier.weight)


def test_ordinary_logits_unaffected_by_evidential_head():
    # The plain model(images) path (used by the existing Trainer) must
    # return exactly the same logits with or without the evidential head
    # having ever been invoked.
    model = _make_model()
    model.eval()
    x = torch.randn(3, 3, 224, 224)
    with torch.no_grad():
        logits_only = model(x)
        output = model(x, return_features=True)
    assert torch.equal(logits_only, output.logits)


def test_evidence_is_non_negative():
    model = _make_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert bool((output.evidence >= 0).all())


def test_dirichlet_alpha_is_evidence_plus_one():
    model = _make_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert torch.allclose(output.dirichlet_alpha, output.evidence + 1.0)


def test_probabilities_sum_to_one():
    model = _make_model()
    model.eval()
    x = torch.randn(6, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    sums = output.probabilities.sum(dim=1)
    assert torch.allclose(sums, torch.ones(6), atol=1e-5)


def test_uncertainty_is_finite_and_in_valid_range():
    model = _make_model(num_classes=10)
    model.eval()
    x = torch.randn(6, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert torch.isfinite(output.uncertainty).all()
    assert bool((output.uncertainty > 0).all())
    assert bool((output.uncertainty <= 1.0).all())


def test_no_nan_or_inf_in_evidential_outputs():
    model = _make_model()
    model.eval()
    x = torch.randn(4, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    for tensor in (output.evidential_raw, output.evidence, output.dirichlet_alpha, output.probabilities, output.uncertainty):
        assert not torch.isnan(tensor).any()
        assert not torch.isinf(tensor).any()


def test_evidential_head_is_included_in_model_parameters():
    model = _make_model()
    param_names = [name for name, _ in model.named_parameters()]
    assert any("evidential_head" in name for name in param_names)


def test_evidential_head_gradients_propagate_through_full_model():
    model = AAEvidentNet(pretrained=False, num_classes=10, embedding_dim=32, local_feature_dim=16)
    x = torch.randn(3, 3, 224, 224)
    output = model(x, return_features=True)
    loss = output.uncertainty.sum() + output.evidence.sum()
    loss.backward()
    assert model.evidential_head.linear.weight.grad is not None
    assert torch.isfinite(model.evidential_head.linear.weight.grad).all()


def test_state_dict_round_trip_includes_evidential_head(tmp_path):
    model = _make_model()
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        expected = model(x, return_features=True)

    path = tmp_path / "aa_evidentnet_edl.pt"
    torch.save(model.state_dict(), path)

    fresh_model = _make_model()
    fresh_model.load_state_dict(torch.load(path, weights_only=True))
    fresh_model.eval()
    with torch.no_grad():
        actual = fresh_model(x, return_features=True)
    assert torch.allclose(expected.uncertainty, actual.uncertainty, atol=1e-6)
    assert torch.allclose(expected.dirichlet_alpha, actual.dirichlet_alpha, atol=1e-6)

"""Tests for src.models.base / src.models.factory. All tests use
pretrained=False and tiny synthetic tensors — no internet access or
pretrained-weight download required."""

import pytest
import torch

from src.models.base import ModelOutput, TimmBackboneModel, UnknownModelError
from src.models.factory import MODEL_NAMES, ModelConfigError, create_model

MINI_CONFIG = {
    "num_classes": 10,
    "baselines": {
        "resnet50": {"architecture": "resnet50", "pretrained": False, "num_classes": 10, "dropout": 0.0},
        "efficientnetb0": {"architecture": "efficientnet_b0", "pretrained": False, "num_classes": 10, "dropout": 0.0},
        "maxvit": {"architecture": "maxvit_tiny_tf_224", "pretrained": False, "num_classes": 10, "dropout": 0.0},
    },
}


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_create_model_initializes(name):
    model = create_model(name, MINI_CONFIG)
    assert isinstance(model, TimmBackboneModel)
    assert model.num_classes == 10
    assert model.feature_dim > 0


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_forward_pass_output_shape(name):
    model = create_model(name, MINI_CONFIG)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (2, 10)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_batch_input_works_with_larger_batch(name):
    model = create_model(name, MINI_CONFIG)
    model.eval()
    x = torch.randn(5, 3, 224, 224)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (5, 10)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_return_features_shape_nonzero(name):
    model = create_model(name, MINI_CONFIG)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert isinstance(output, ModelOutput)
    assert output.logits.shape == (2, 10)
    assert output.features is not None
    assert output.features.shape[0] == 2
    assert output.features.shape[1] > 0
    assert output.features.shape[1] == model.feature_dim


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_no_nan_or_inf_in_forward_pass(name):
    model = create_model(name, MINI_CONFIG)
    model.eval()
    x = torch.randn(2, 3, 224, 224)
    with torch.no_grad():
        output = model(x, return_features=True)
    assert not torch.isnan(output.logits).any()
    assert not torch.isinf(output.logits).any()
    assert not torch.isnan(output.features).any()
    assert not torch.isinf(output.features).any()


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_invalid_input_dimensions_fail_clearly(name):
    model = create_model(name, MINI_CONFIG)
    bad_input = torch.randn(3, 224, 224)  # missing batch dimension (3D, not 4D)
    with pytest.raises(ValueError, match="4D"):
        model(bad_input)


def test_invalid_model_name_fails_clearly():
    with pytest.raises(ModelConfigError, match="Unknown baseline model name"):
        create_model("not_a_real_model", MINI_CONFIG)


def test_missing_architecture_key_fails_clearly():
    config = {"num_classes": 10, "baselines": {"broken": {"pretrained": False}}}
    with pytest.raises(ModelConfigError, match="architecture"):
        create_model("broken", config)


def test_unknown_timm_architecture_fails_clearly():
    with pytest.raises(UnknownModelError):
        TimmBackboneModel(architecture="this_is_not_a_real_timm_model_xyz", pretrained=False, num_classes=10)


def test_all_baselines_expose_consistent_interface():
    for name in MODEL_NAMES:
        model = create_model(name, MINI_CONFIG)
        assert hasattr(model, "forward")
        assert hasattr(model, "feature_dim")
        assert hasattr(model, "num_classes")
        assert hasattr(model, "architecture")


def test_num_classes_is_configurable():
    config = {
        "num_classes": 4,
        "baselines": {"resnet50": {"architecture": "resnet50", "pretrained": False, "num_classes": 4}},
    }
    model = create_model("resnet50", config)
    x = torch.randn(2, 3, 224, 224)
    logits = model(x)
    assert logits.shape == (2, 4)


def test_num_classes_falls_back_to_top_level_config():
    config = {
        "num_classes": 7,
        "baselines": {"resnet50": {"architecture": "resnet50", "pretrained": False}},
    }
    model = create_model("resnet50", config)
    assert model.num_classes == 7


def test_parameter_counts_are_positive_and_trainable():
    model = create_model("efficientnetb0", MINI_CONFIG)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert total > 0
    assert trainable == total  # nothing frozen by default

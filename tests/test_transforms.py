"""Tests for src.data.transforms: runtime preprocessing (Task 5), distinct
from the offline augmentation pipeline (Task 4)."""

import torch
from PIL import Image

from src.data.transforms import (
    build_eval_transform,
    build_train_transform,
    build_transforms_from_config,
)


def _make_pil_image(size=(300, 200), color=(120, 60, 200)):
    return Image.new("RGB", size, color)


def test_eval_transform_output_shape():
    transform = build_eval_transform(image_size=224, resize_size=256)
    img = _make_pil_image()
    tensor = transform(img)
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)


def test_train_transform_output_shape():
    transform = build_train_transform(image_size=224, resize_size=256)
    tensor = transform(_make_pil_image())
    assert tensor.shape == (3, 224, 224)


def test_transform_works_on_small_images_too():
    transform = build_eval_transform(image_size=224, resize_size=256)
    tensor = transform(_make_pil_image(size=(20, 20)))
    assert tensor.shape == (3, 224, 224)


def test_transform_works_on_non_square_images():
    transform = build_eval_transform(image_size=224, resize_size=256)
    tensor = transform(_make_pil_image(size=(2004, 1690)))  # real DS2 aspect ratio
    assert tensor.shape == (3, 224, 224)


def test_eval_transform_is_deterministic():
    transform = build_eval_transform(image_size=224, resize_size=256)
    img = _make_pil_image()
    a = transform(img)
    b = transform(img)
    assert torch.equal(a, b)


def test_train_transform_is_deterministic():
    # No randomness should be introduced at the runtime-preprocessing layer
    # (Task 4's offline augmentation already ran once, ahead of time).
    transform = build_train_transform(image_size=224, resize_size=256)
    img = _make_pil_image()
    a = transform(img)
    b = transform(img)
    assert torch.equal(a, b)


def test_normalization_is_applied():
    # A mid-gray image's ToTensor value (0.5) should be shifted away from
    # 0.5 by ImageNet normalization (mean/std != 0.5/1.0).
    transform = build_eval_transform(image_size=32, resize_size=32)
    gray_img = Image.new("RGB", (32, 32), (128, 128, 128))
    tensor = transform(gray_img)
    assert not torch.allclose(tensor, torch.full_like(tensor, 0.5019608), atol=1e-3)


def test_custom_normalization_stats_are_honored():
    transform_default = build_eval_transform(image_size=32, resize_size=32)
    transform_custom = build_eval_transform(
        image_size=32, resize_size=32, normalize_mean=[0.0, 0.0, 0.0], normalize_std=[1.0, 1.0, 1.0]
    )
    img = Image.new("RGB", (32, 32), (128, 128, 128))
    default_tensor = transform_default(img)
    custom_tensor = transform_custom(img)
    assert not torch.allclose(default_tensor, custom_tensor)
    # With mean=0, std=1, normalization is a no-op on top of ToTensor.
    assert torch.allclose(custom_tensor, torch.full_like(custom_tensor, 128 / 255), atol=1e-4)


def test_build_transforms_from_config_uses_config_values():
    config = {
        "image": {
            "size": 64,
            "resize_size": 72,
            "normalize_mean": [0.5, 0.5, 0.5],
            "normalize_std": [0.5, 0.5, 0.5],
        }
    }
    train_t, eval_t = build_transforms_from_config(config)
    img = _make_pil_image()
    assert train_t(img).shape == (3, 64, 64)
    assert eval_t(img).shape == (3, 64, 64)


def test_build_transforms_from_config_defaults_when_missing():
    train_t, eval_t = build_transforms_from_config({})
    img = _make_pil_image()
    assert train_t(img).shape == (3, 224, 224)
    assert eval_t(img).shape == (3, 224, 224)

"""Runtime image preprocessing (Task 5) — distinct from the offline
training-only augmentation pipeline (Task 4, src/data/generate_balanced_dataset.py).

  - Offline augmentation (Task 4): ran ONCE, ahead of time, only against
    train_original.csv, to produce the physical image files referenced by
    train_balanced.csv (flips, rotations, brightness/contrast, affine,
    color jitter). Those pixels are already baked into the generated files.
  - Runtime preprocessing (this module): applied every time ANY sample is
    loaded (train, validation, or test) — resize, center-crop, tensor
    conversion, and normalization. It is deterministic and carries no
    additional randomness, since diversity for training was already
    introduced offline. Validation and test use the exact same
    deterministic function; nothing here is ever random.

Default normalization uses ImageNet statistics because all three baseline
backbones (ResNet50, EfficientNetB0, MaxViT) use ImageNet-pretrained
weights.
"""

from typing import Any, Dict, Optional, Sequence, Tuple

import torchvision.transforms as T

DEFAULT_IMAGE_SIZE = 224
DEFAULT_RESIZE_SIZE = 256
DEFAULT_NORMALIZE_MEAN = [0.485, 0.456, 0.406]
DEFAULT_NORMALIZE_STD = [0.229, 0.224, 0.225]


def _build_transform(image_size: int, resize_size: int, normalize_mean: Sequence[float], normalize_std: Sequence[float]) -> T.Compose:
    return T.Compose(
        [
            T.Resize(resize_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(mean=list(normalize_mean), std=list(normalize_std)),
        ]
    )


def build_train_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
) -> T.Compose:
    """Runtime preprocessing for training samples (train_original.csv or
    train_balanced.csv). Deterministic - no additional random augmentation
    is layered on top of Task 4's offline pipeline here."""
    return _build_transform(
        image_size, resize_size, normalize_mean or DEFAULT_NORMALIZE_MEAN, normalize_std or DEFAULT_NORMALIZE_STD
    )


def build_eval_transform(
    image_size: int = DEFAULT_IMAGE_SIZE,
    resize_size: int = DEFAULT_RESIZE_SIZE,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
) -> T.Compose:
    """Deterministic preprocessing for validation/test. Never apply random
    augmentation here — kept as a separate function (even though it
    currently performs the same operations as build_train_transform) so a
    future change to training-time preprocessing can never accidentally
    leak into evaluation."""
    return _build_transform(
        image_size, resize_size, normalize_mean or DEFAULT_NORMALIZE_MEAN, normalize_std or DEFAULT_NORMALIZE_STD
    )


def build_transforms_from_config(dataset_config: Dict[str, Any]) -> Tuple[T.Compose, T.Compose]:
    """Build (train_transform, eval_transform) from a loaded
    configs/dataset.yaml dict's `image` section."""
    image_cfg = dataset_config.get("image", {}) or {}
    image_size = image_cfg.get("size", DEFAULT_IMAGE_SIZE)
    resize_size = image_cfg.get("resize_size", DEFAULT_RESIZE_SIZE)
    mean = image_cfg.get("normalize_mean", DEFAULT_NORMALIZE_MEAN)
    std = image_cfg.get("normalize_std", DEFAULT_NORMALIZE_STD)
    train_transform = build_train_transform(image_size, resize_size, mean, std)
    eval_transform = build_eval_transform(image_size, resize_size, mean, std)
    return train_transform, eval_transform

"""Shared fixtures/helpers for dataset audit tests.

All fixtures build tiny, self-contained image sets under tmp_path — no test
in this suite touches the real 1.7GB DS2 dataset under data/raw/.
"""

import hashlib

import yaml
from PIL import Image


def make_image(path, size=(20, 20), color=None):
    """Write a tiny valid JPEG to `path`, creating parent dirs as needed.

    When `color` is not given, it is derived deterministically from `path`
    so that two unrelated make_image() calls never accidentally produce
    byte-identical files (which would register as an unintended exact
    duplicate). Pass an explicit `color` (and matching `size`) when a test
    *wants* two files to be byte-identical.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    if color is None:
        digest = hashlib.md5(str(path).encode("utf-8")).digest()
        color = (digest[0], digest[1], digest[2])
    Image.new("RGB", size, color).save(path, format="JPEG")


def make_invalid_image(path, content=b"this is not a real jpeg file"):
    """Write a file with a .jpg-looking name that is not a decodable image."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


DEFAULT_TEST_AUGMENTATION_CONFIG = {
    "enabled": True,
    # Deliberately tiny (vs. the real 2000) so tests exercise the full
    # audit -> split -> balance pipeline in milliseconds, not minutes.
    "target_samples_per_class": 10,
    "horizontal_flip": {"enabled": True, "probability": 0.5},
    "rotation": {"enabled": True, "degrees": 10},
    "brightness": {"enabled": True, "factor": 0.15},
    "contrast": {"enabled": True, "factor": 0.15},
    "affine": {"enabled": True, "translate": 0.05, "scale": {"min": 0.95, "max": 1.05}},
    "color_jitter": {"enabled": True, "factor": 0.10},
}


def write_min_dataset_config(
    tmp_path,
    class_directory_mapping,
    raw_dir,
    audit_dir,
    policies=None,
    keywords=None,
    supported_extensions=None,
    augmentation=None,
    config_name="dataset.yaml",
):
    """Write a minimal but schema-valid dataset.yaml for run_dataset_audit().

    `augmentation` overrides src/data/generate_balanced_dataset.py's config
    (defaults to DEFAULT_TEST_AUGMENTATION_CONFIG, a tiny target for fast
    tests) - pass an explicit dict (e.g. {"target_samples_per_class": 2000})
    to test closer to real settings.
    """
    config = {
        "seed": 42,
        "paths": {
            "raw_dir": str(raw_dir),
            "processed_dir": str(tmp_path / "processed"),
            "manifests_dir": str(tmp_path / "manifests"),
            "audit_dir": str(audit_dir),
        },
        "class_names": sorted(class_directory_mapping.keys()),
        "class_directory_mapping": class_directory_mapping,
        "num_classes": len(class_directory_mapping),
        "split": {
            "train_fraction": 0.70,
            "val_fraction": 0.20,
            "test_fraction": 0.10,
            "stratified": True,
        },
        "target_train_samples_per_class": 2000,
        "augmentation": {**DEFAULT_TEST_AUGMENTATION_CONFIG, **(augmentation or {})},
        "audit": {
            "supported_extensions": supported_extensions or [".jpg", ".jpeg"],
            "policies": policies or {},
            "augmentation_keywords": keywords
            if keywords is not None
            else [
                "augmented",
                "augmentation",
                "flip",
                "flipped",
                "rotate",
                "rotated",
                "rotation",
                "brightness",
                "contrast",
                "crop",
                "cropped",
                "zoom",
                "noise",
                "blur",
                "blurred",
                "copy",
                "variant",
                "synthetic",
                "generated",
            ],
        },
    }
    config_path = tmp_path / config_name
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def write_min_training_config(tmp_path, overrides=None, config_name="training.yaml"):
    """Write a minimal, fast training.yaml for src.training tests -
    results/logs and results/checkpoints both point under tmp_path so
    tests never touch the real project's results/ or experiments/
    directories."""
    config = {
        "seed": 42,
        "device": "cpu",
        "smoke_test": False,
        "batch_size": 4,
        "epochs": 2,
        "num_workers": 0,
        "optimizer": {"name": "adamw", "lr": 3.0e-4, "weight_decay": 1.0e-4, "betas": [0.9, 0.999], "eps": 1.0e-8},
        "scheduler": {"name": "reduce_on_plateau", "factor": 0.5, "patience": 5, "min_lr": 1.0e-6},
        "early_stopping": {"enabled": True, "patience": 10},
        "monitor_metric": "val_macro_f1",
        "mode": "max",
        "gradient_clip_norm": 1.0,
        "gradient_accumulation_steps": 1,
        "mixed_precision": True,
        "checkpoint_frequency": 1,
        "checkpointing": {"save_dir": str(tmp_path / "checkpoints")},
        "logging": {"log_dir": str(tmp_path / "logs")},
    }
    config.update(overrides or {})
    config_path = tmp_path / config_name
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path


def write_min_models_config(tmp_path, num_classes=10, config_name="models.yaml"):
    """Minimal models.yaml with all three real baselines, pretrained=False
    (fast, offline). Uses the real timm architecture names since there is
    no lightweight fake substitute for "a real create_model() call"."""
    config = {
        "seed": 42,
        "num_classes": num_classes,
        "image_size": 224,
        "baselines": {
            "resnet50": {"architecture": "resnet50", "pretrained": False, "num_classes": num_classes, "dropout": 0.0},
            "efficientnetb0": {
                "architecture": "efficientnet_b0",
                "pretrained": False,
                "num_classes": num_classes,
                "dropout": 0.0,
            },
            "maxvit": {
                "architecture": "maxvit_tiny_tf_224",
                "pretrained": False,
                "num_classes": num_classes,
                "dropout": 0.0,
            },
        },
    }
    config_path = tmp_path / config_name
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)
    return config_path

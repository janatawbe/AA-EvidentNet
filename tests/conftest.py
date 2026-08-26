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


def write_min_dataset_config(
    tmp_path,
    class_directory_mapping,
    raw_dir,
    audit_dir,
    policies=None,
    keywords=None,
    supported_extensions=None,
    config_name="dataset.yaml",
):
    """Write a minimal but schema-valid dataset.yaml for run_dataset_audit()."""
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

"""Tests for src.utils.config: YAML loading and deterministic hashing."""

from pathlib import Path

import pytest
import yaml

from src.utils.config import hash_config, load_config

REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.mark.parametrize(
    "config_name",
    [
        "dataset.yaml",
        "training.yaml",
        "models.yaml",
        "losses.yaml",
        "evaluation.yaml",
        "experiments.yaml",
    ],
)
def test_real_configs_load_as_dicts(config_name):
    config = load_config(REPO_ROOT / "configs" / config_name)
    assert isinstance(config, dict)
    assert "seed" in config


def test_load_config_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path / "does_not_exist.yaml")


def test_load_config_roundtrip(tmp_path):
    data = {"seed": 42, "batch_size": 16, "nested": {"a": 1, "b": [1, 2, 3]}}
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f)

    loaded = load_config(config_path)
    assert loaded == data


def test_hash_config_is_deterministic():
    config = {"seed": 42, "batch_size": 16, "image_size": 224}
    assert hash_config(config) == hash_config(config)


def test_hash_config_independent_of_key_order():
    config_a = {"seed": 42, "batch_size": 16}
    config_b = {"batch_size": 16, "seed": 42}
    assert hash_config(config_a) == hash_config(config_b)


def test_hash_config_sensitive_to_value_changes():
    config_a = {"seed": 42, "batch_size": 16}
    config_b = {"seed": 42, "batch_size": 32}
    assert hash_config(config_a) != hash_config(config_b)


def test_hash_config_sensitive_to_nested_changes():
    config_a = {"seed": 42, "nested": {"lr": 0.001}}
    config_b = {"seed": 42, "nested": {"lr": 0.002}}
    assert hash_config(config_a) != hash_config(config_b)


def test_hash_config_returns_hex_sha256():
    digest = hash_config({"seed": 42})
    assert isinstance(digest, str)
    assert len(digest) == 64
    int(digest, 16)  # raises ValueError if not valid hex

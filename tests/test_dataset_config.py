"""Tests for configs/dataset.yaml: fixed split ratios and the canonical
class name <-> raw directory mapping.

These values are fixed research-methodology decisions (see the comments in
configs/dataset.yaml), not provisional defaults, so these tests pin them
down exactly rather than just sanity-checking types.
"""

import math
from pathlib import Path

import pytest

from src.utils.config import load_config

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_CONFIG_PATH = REPO_ROOT / "configs" / "dataset.yaml"


@pytest.fixture(scope="module")
def dataset_config():
    return load_config(DATASET_CONFIG_PATH)


def test_split_ratios_are_exactly_70_20_10(dataset_config):
    split = dataset_config["split"]
    assert split["train_fraction"] == pytest.approx(0.70, abs=0.0)
    assert split["val_fraction"] == pytest.approx(0.20, abs=0.0)
    assert split["test_fraction"] == pytest.approx(0.10, abs=0.0)


def test_split_ratios_sum_to_one(dataset_config):
    split = dataset_config["split"]
    total = split["train_fraction"] + split["val_fraction"] + split["test_fraction"]
    assert math.isclose(total, 1.0, rel_tol=0, abs_tol=1e-9)


def test_split_is_stratified(dataset_config):
    assert dataset_config["split"]["stratified"] is True


def test_target_train_samples_per_class_is_fixed_at_2000(dataset_config):
    # Fixed research methodology (not provisional) — see configs/dataset.yaml.
    # This test guards against accidental changes; it is not asserting that
    # 2000 is "correct" in any tunable sense.
    assert dataset_config["target_train_samples_per_class"] == 2000


def test_num_classes_matches_class_names_length(dataset_config):
    assert dataset_config["num_classes"] == len(dataset_config["class_names"])
    assert dataset_config["num_classes"] == 10


def test_class_directory_mapping_covers_every_canonical_class_exactly_once(dataset_config):
    class_names = dataset_config["class_names"]
    mapping = dataset_config["class_directory_mapping"]

    assert set(mapping.keys()) == set(class_names)
    assert len(mapping) == len(class_names)  # no duplicate canonical keys


def test_class_directory_mapping_values_are_unique(dataset_config):
    mapping = dataset_config["class_directory_mapping"]
    directory_names = list(mapping.values())
    assert len(directory_names) == len(set(directory_names))


def test_class_directory_mapping_is_not_trivial_substring_match(dataset_config):
    # At least one canonical name must differ from its mapped raw directory
    # name (e.g. "Central Serous Chorioretinopathy" vs the raw directory
    # "...[Color Fundus]"), proving this is an explicit lookup table rather
    # than an assumption that names match 1:1.
    mapping = dataset_config["class_directory_mapping"]
    assert any(canonical != raw_dir for canonical, raw_dir in mapping.items())


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw").is_dir(),
    reason="data/raw/ not present in this environment",
)
def test_class_directory_mapping_targets_exist_on_disk(dataset_config):
    raw_dir = REPO_ROOT / "data" / "raw"
    mapping = dataset_config["class_directory_mapping"]
    for canonical_name, directory_name in mapping.items():
        assert (raw_dir / directory_name).is_dir(), (
            f"Mapped directory for class '{canonical_name}' not found: "
            f"data/raw/{directory_name}"
        )


@pytest.mark.skipif(
    not (REPO_ROOT / "data" / "raw").is_dir(),
    reason="data/raw/ not present in this environment",
)
def test_raw_directory_names_are_all_covered_by_mapping(dataset_config):
    raw_dir = REPO_ROOT / "data" / "raw"
    mapping = dataset_config["class_directory_mapping"]
    mapped_dirs = set(mapping.values())
    actual_dirs = {p.name for p in raw_dir.iterdir() if p.is_dir()}
    assert actual_dirs == mapped_dirs

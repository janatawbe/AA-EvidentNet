"""Tests for src.data.dataset: manifest loading/validation and RetinalDataset.

Synthetic fixtures only — no test requires the real 1.7GB DS2 dataset.
"""

import csv

import pytest
import torch

from src.data.build_split import MANIFEST_COLUMNS
from src.data.dataset import (
    DatasetManifestError,
    RetinalDataset,
    assert_no_cross_manifest_overlap,
    assert_paths_exist,
    build_class_to_idx,
    build_idx_to_class,
    load_manifest_rows,
    resolve_image_path,
    validate_dataset_manifests,
    validate_manifest_rows,
)
from tests.conftest import make_image

CANONICAL_CLASSES = ["Alpha", "Beta", "Gamma"]


def _row(path, canonical_class, split="train", original_id=None, parent_original_id=None, is_original="true", augmentation_type="original"):
    original_id = original_id or f"id_{path}"
    return {
        "path": path,
        "class": canonical_class,
        "split": split,
        "original_id": original_id,
        "parent_original_id": parent_original_id or original_id,
        "is_original": is_original,
        "augmentation_type": augmentation_type,
    }


def _write_manifest(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


# --- class mapping determinism ---


def test_build_class_to_idx_is_alphabetical():
    mapping = build_class_to_idx(["Zebra", "Alpha", "Middle"])
    assert mapping == {"Alpha": 0, "Middle": 1, "Zebra": 2}


def test_build_class_to_idx_independent_of_input_order():
    a = build_class_to_idx(["Beta", "Alpha", "Gamma"])
    b = build_class_to_idx(["Gamma", "Beta", "Alpha"])
    assert a == b


def test_build_class_to_idx_deterministic_across_calls():
    assert build_class_to_idx(CANONICAL_CLASSES) == build_class_to_idx(CANONICAL_CLASSES)


def test_build_idx_to_class_is_inverse():
    to_idx = build_class_to_idx(CANONICAL_CLASSES)
    to_class = build_idx_to_class(CANONICAL_CLASSES)
    for name, idx in to_idx.items():
        assert to_class[idx] == name


# --- manifest loading ---


def test_load_manifest_rows_missing_file_raises(tmp_path):
    with pytest.raises(DatasetManifestError):
        load_manifest_rows(tmp_path / "does_not_exist.csv")


def test_load_manifest_rows_missing_columns_raises(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("path,class\nAlpha/a1.jpg,Alpha\n", encoding="utf-8")
    with pytest.raises(DatasetManifestError, match="missing required column"):
        load_manifest_rows(path)


def test_load_manifest_rows_empty_raises(tmp_path):
    path = tmp_path / "empty.csv"
    _write_manifest(path, [])
    with pytest.raises(DatasetManifestError, match="zero rows"):
        load_manifest_rows(path)


def test_load_manifest_rows_valid(tmp_path):
    path = tmp_path / "ok.csv"
    rows = [_row("Alpha/a1.jpg", "Alpha")]
    _write_manifest(path, rows)
    loaded = load_manifest_rows(path)
    assert len(loaded) == 1
    assert loaded[0]["path"] == "Alpha/a1.jpg"


# --- row-value validation ---


def test_validate_manifest_rows_accepts_well_formed():
    rows = [_row("Alpha/a1.jpg", "Alpha", split="train")]
    validate_manifest_rows(rows, CANONICAL_CLASSES, expected_split="train")  # must not raise


def test_validate_manifest_rows_rejects_invalid_class():
    rows = [_row("Alpha/a1.jpg", "NotARealClass")]
    with pytest.raises(DatasetManifestError, match="invalid class"):
        validate_manifest_rows(rows, CANONICAL_CLASSES)


def test_validate_manifest_rows_rejects_invalid_split():
    rows = [_row("Alpha/a1.jpg", "Alpha", split="bogus")]
    with pytest.raises(DatasetManifestError, match="invalid split"):
        validate_manifest_rows(rows, CANONICAL_CLASSES)


def test_validate_manifest_rows_rejects_split_mismatch():
    rows = [_row("Alpha/a1.jpg", "Alpha", split="val")]
    with pytest.raises(DatasetManifestError, match="does not match expected"):
        validate_manifest_rows(rows, CANONICAL_CLASSES, expected_split="train")


def test_validate_manifest_rows_rejects_is_original_augmentation_mismatch():
    rows = [_row("Alpha/a1.jpg", "Alpha", is_original="true", augmentation_type="rotation")]
    with pytest.raises(DatasetManifestError, match="inconsistent"):
        validate_manifest_rows(rows, CANONICAL_CLASSES)


def test_validate_manifest_rows_require_all_original_rejects_generated_row():
    rows = [_row("Alpha/gen.jpg", "Alpha", is_original="false", augmentation_type="rotation")]
    with pytest.raises(DatasetManifestError, match="must contain only original"):
        validate_manifest_rows(rows, CANONICAL_CLASSES, manifest_name="val_original.csv", require_all_original=True)


def test_validate_manifest_rows_reports_multiple_errors():
    rows = [
        _row("Alpha/a1.jpg", "BadClass"),
        _row("Alpha/a2.jpg", "Alpha", split="bogus"),
    ]
    with pytest.raises(DatasetManifestError) as exc_info:
        validate_manifest_rows(rows, CANONICAL_CLASSES)
    message = str(exc_info.value)
    assert "invalid class" in message
    assert "invalid split" in message


# --- path resolution / existence ---


def test_resolve_image_path_original_uses_raw_dir(tmp_path):
    row = _row("Alpha/a1.jpg", "Alpha", is_original="true")
    path = resolve_image_path(row, tmp_path / "raw", tmp_path / "processed" / "train")
    assert path == tmp_path / "raw" / "Alpha" / "a1.jpg"


def test_resolve_image_path_generated_uses_processed_train_dir(tmp_path):
    row = _row("Alpha/gen1.jpg", "Alpha", is_original="false", augmentation_type="rotation")
    path = resolve_image_path(row, tmp_path / "raw", tmp_path / "processed" / "train")
    assert path == tmp_path / "processed" / "train" / "Alpha" / "gen1.jpg"


def test_assert_paths_exist_detects_missing_file(tmp_path):
    rows = [_row("Alpha/missing.jpg", "Alpha")]
    with pytest.raises(DatasetManifestError, match="do not exist"):
        assert_paths_exist(rows, tmp_path / "raw", tmp_path / "processed" / "train")


def test_assert_paths_exist_passes_when_present(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    rows = [_row("Alpha/a1.jpg", "Alpha")]
    assert_paths_exist(rows, raw_dir, tmp_path / "processed" / "train")  # must not raise


# --- cross-manifest overlap ---


def test_assert_no_cross_manifest_overlap_passes_for_disjoint():
    manifests = {
        "train": [_row("Alpha/a1.jpg", "Alpha", original_id="id1")],
        "val": [_row("Alpha/a2.jpg", "Alpha", split="val", original_id="id2")],
    }
    assert_no_cross_manifest_overlap(manifests)  # must not raise


def test_assert_no_cross_manifest_overlap_detects_overlap():
    manifests = {
        "train": [_row("Alpha/a1.jpg", "Alpha", original_id="id1")],
        "val": [_row("Alpha/a1.jpg", "Alpha", split="val", original_id="id1")],
    }
    with pytest.raises(DatasetManifestError, match="overlap"):
        assert_no_cross_manifest_overlap(manifests)


# --- full manifest-set consistency ---


def test_validate_dataset_manifests_accepts_clean_set():
    train_balanced = [
        _row("Alpha/a1.jpg", "Alpha", original_id="id1"),
        _row("Alpha/gen1.jpg", "Alpha", original_id="gen1", parent_original_id="id1", is_original="false", augmentation_type="rotation"),
    ]
    val = [_row("Alpha/a2.jpg", "Alpha", split="val", original_id="id2")]
    test = [_row("Alpha/a3.jpg", "Alpha", split="test", original_id="id3")]
    validate_dataset_manifests(train_balanced, val, test, CANONICAL_CLASSES)  # must not raise


def test_validate_dataset_manifests_rejects_generated_row_in_val():
    train_balanced = [_row("Alpha/a1.jpg", "Alpha", original_id="id1")]
    val = [_row("Alpha/gen.jpg", "Alpha", split="val", is_original="false", augmentation_type="rotation", original_id="gen1")]
    test = [_row("Alpha/a3.jpg", "Alpha", split="test", original_id="id3")]
    with pytest.raises(DatasetManifestError, match="must contain only original"):
        validate_dataset_manifests(train_balanced, val, test, CANONICAL_CLASSES)


def test_validate_dataset_manifests_rejects_generated_row_in_test():
    train_balanced = [_row("Alpha/a1.jpg", "Alpha", original_id="id1")]
    val = [_row("Alpha/a2.jpg", "Alpha", split="val", original_id="id2")]
    test = [_row("Alpha/gen.jpg", "Alpha", split="test", is_original="false", augmentation_type="rotation", original_id="gen2")]
    with pytest.raises(DatasetManifestError, match="must contain only original"):
        validate_dataset_manifests(train_balanced, val, test, CANONICAL_CLASSES)


def test_validate_dataset_manifests_rejects_non_train_split_in_balanced():
    train_balanced = [_row("Alpha/a1.jpg", "Alpha", split="val", original_id="id1")]
    val = [_row("Alpha/a2.jpg", "Alpha", split="val", original_id="id2")]
    test = [_row("Alpha/a3.jpg", "Alpha", split="test", original_id="id3")]
    with pytest.raises(DatasetManifestError):
        validate_dataset_manifests(train_balanced, val, test, CANONICAL_CLASSES)


def test_validate_dataset_manifests_rejects_train_val_overlap():
    train_balanced = [_row("Alpha/a1.jpg", "Alpha", original_id="shared")]
    val = [_row("Alpha/a1.jpg", "Alpha", split="val", original_id="shared")]
    test = [_row("Alpha/a3.jpg", "Alpha", split="test", original_id="id3")]
    with pytest.raises(DatasetManifestError, match="overlap"):
        validate_dataset_manifests(train_balanced, val, test, CANONICAL_CLASSES)


# --- RetinalDataset ---


def test_retinal_dataset_len_and_getitem_schema(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(40, 40))
    rows = [_row("Alpha/a1.jpg", "Alpha", original_id="id1")]
    dataset = RetinalDataset(rows, CANONICAL_CLASSES, raw_dir, tmp_path / "processed" / "train")

    assert len(dataset) == 1
    sample = dataset[0]
    assert set(sample.keys()) == {
        "image", "label", "class_name", "image_path", "original_id", "parent_original_id", "is_original",
    }
    assert sample["label"] == build_class_to_idx(CANONICAL_CLASSES)["Alpha"]
    assert sample["class_name"] == "Alpha"
    assert sample["original_id"] == "id1"
    assert sample["parent_original_id"] == "id1"
    assert sample["is_original"] is True
    assert isinstance(sample["is_original"], bool)


def test_retinal_dataset_applies_transform(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(40, 40))
    rows = [_row("Alpha/a1.jpg", "Alpha")]

    def to_tensor_transform(img):
        return torch.from_numpy(__import__("numpy").array(img))

    dataset = RetinalDataset(rows, CANONICAL_CLASSES, raw_dir, tmp_path / "processed" / "train", transform=to_tensor_transform)
    sample = dataset[0]
    assert isinstance(sample["image"], torch.Tensor)


def test_retinal_dataset_reads_generated_image_from_processed_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(processed_train_dir / "Alpha" / "gen1.jpg", size=(30, 30))
    rows = [_row("Alpha/gen1.jpg", "Alpha", original_id="gen1", parent_original_id="id1", is_original="false", augmentation_type="rotation")]
    dataset = RetinalDataset(rows, CANONICAL_CLASSES, raw_dir, processed_train_dir)
    sample = dataset[0]
    assert sample["is_original"] is False
    assert sample["parent_original_id"] == "id1"
    assert "processed" in sample["image_path"]


def test_retinal_dataset_from_manifest_balanced_compatibility(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(processed_train_dir / "Alpha" / "gen1.jpg")
    rows = [
        _row("Alpha/a1.jpg", "Alpha", original_id="id1"),
        _row("Alpha/gen1.jpg", "Alpha", original_id="gen1", parent_original_id="id1", is_original="false", augmentation_type="rotation"),
    ]
    manifest_path = tmp_path / "train_balanced.csv"
    _write_manifest(manifest_path, rows)

    dataset = RetinalDataset.from_manifest(
        manifest_path, CANONICAL_CLASSES, raw_dir, processed_train_dir, expected_split="train"
    )
    assert len(dataset) == 2
    is_original_flags = {dataset[i]["is_original"] for i in range(2)}
    assert is_original_flags == {True, False}


def test_retinal_dataset_from_manifest_rejects_generated_row_for_val(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(processed_train_dir / "Alpha" / "gen1.jpg")
    rows = [_row("Alpha/gen1.jpg", "Alpha", split="val", is_original="false", augmentation_type="rotation", original_id="gen1")]
    manifest_path = tmp_path / "val_original.csv"
    _write_manifest(manifest_path, rows)

    with pytest.raises(DatasetManifestError, match="must contain only original"):
        RetinalDataset.from_manifest(
            manifest_path, CANONICAL_CLASSES, raw_dir, processed_train_dir,
            expected_split="val", require_all_original=True,
        )


def test_retinal_dataset_from_manifest_missing_file_raises(tmp_path):
    raw_dir = tmp_path / "raw"
    rows = [_row("Alpha/missing.jpg", "Alpha")]
    manifest_path = tmp_path / "train_original.csv"
    _write_manifest(manifest_path, rows)
    with pytest.raises(DatasetManifestError, match="do not exist"):
        RetinalDataset.from_manifest(manifest_path, CANONICAL_CLASSES, raw_dir, tmp_path / "processed" / "train")

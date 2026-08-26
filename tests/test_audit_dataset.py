"""Tests for src.data.audit_dataset: inventory, integrity, class structure."""

import pytest

from src.data.audit_dataset import (
    AuditConfigError,
    build_inventory,
    validate_class_directories,
    validate_dataset_config,
    write_corrupted_images_csv,
)
from src.data.records import INVENTORY_COLUMNS
from tests.conftest import make_image, make_invalid_image

MAPPING = {
    "Alpha": "Alpha",
    "Beta Disease": "Beta Disease [Raw]",
}


def _build_two_class_raw_dir(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Alpha" / "a2.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")
    return raw_dir


# --- Inventory: discovery, extensions, ordering, class mapping ---


def test_build_inventory_discovers_all_images(tmp_path):
    raw_dir = _build_two_class_raw_dir(tmp_path)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    assert len(records) == 3


def test_build_inventory_maps_canonical_class_and_directory(tmp_path):
    raw_dir = _build_two_class_raw_dir(tmp_path)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    by_filename = {r.filename: r for r in records}
    assert by_filename["b1.jpg"].canonical_class == "Beta Disease"
    assert by_filename["b1.jpg"].class_directory == "Beta Disease [Raw]"
    assert by_filename["a1.jpg"].canonical_class == "Alpha"


def test_build_inventory_only_supported_extensions(tmp_path):
    raw_dir = _build_two_class_raw_dir(tmp_path)
    (raw_dir / "Alpha" / "notes.txt").write_text("not an image")
    (raw_dir / "Alpha" / "a3.png").write_bytes(b"\x89PNG fake")
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    filenames = {r.filename for r in records}
    assert "notes.txt" not in filenames
    assert "a3.png" not in filenames


def test_build_inventory_extensions_are_configurable(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    (raw_dir / "Alpha" / "a2.bmp").write_bytes(b"BM fake bitmap")
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)

    records_jpg_only = build_inventory(raw_dir, MAPPING, [".jpg"])
    assert {r.filename for r in records_jpg_only} == {"a1.jpg"}

    records_with_bmp = build_inventory(raw_dir, MAPPING, [".jpg", ".bmp"])
    assert {r.filename for r in records_with_bmp} == {"a1.jpg", "a2.bmp"}


def test_build_inventory_deterministic_ordering(tmp_path):
    raw_dir = tmp_path / "raw"
    for name in ["z9.jpg", "a1.jpg", "M5.jpg", "b2.jpg"]:
        make_image(raw_dir / "Alpha" / name)
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)

    records_a = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    records_b = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])

    paths_a = [r.path for r in records_a]
    paths_b = [r.path for r in records_b]
    assert paths_a == paths_b  # stable across repeated runs
    assert paths_a == sorted(paths_a, key=str.lower)  # case-insensitive filename order


def test_build_inventory_missing_directory_is_skipped_not_crashed(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    # "Beta Disease [Raw]" directory intentionally not created.
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    assert len(records) == 1
    assert records[0].canonical_class == "Alpha"


def test_inventory_record_has_required_fields(tmp_path):
    raw_dir = _build_two_class_raw_dir(tmp_path)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    required = {
        "path",
        "filename",
        "extension",
        "class_directory",
        "canonical_class",
        "file_size_bytes",
        "width",
        "height",
        "mode",
        "sha256",
        "is_readable",
    }
    assert required.issubset(set(INVENTORY_COLUMNS))
    record = records[0]
    assert record.width == 20
    assert record.height == 20
    assert record.mode == "RGB"
    assert record.is_readable is True
    assert len(record.sha256) == 64
    assert record.file_size_bytes > 0


def test_inventory_csv_schema(tmp_path):
    import csv

    from src.data.records import write_csv

    raw_dir = _build_two_class_raw_dir(tmp_path)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    output = tmp_path / "audit" / "dataset_inventory.csv"
    write_csv(records, INVENTORY_COLUMNS, output)

    with open(output, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == list(INVENTORY_COLUMNS)
        rows = list(reader)
    assert len(rows) == 3


# --- Corruption / integrity ---


def test_readable_image_is_marked_readable(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "good.jpg")
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    assert records[0].is_readable is True
    assert records[0].error_message == ""


def test_corrupted_image_is_marked_unreadable(tmp_path):
    raw_dir = tmp_path / "raw"
    make_invalid_image(raw_dir / "Alpha" / "broken.jpg")
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])
    assert records[0].is_readable is False
    assert records[0].error_message != ""
    assert records[0].width is None
    assert records[0].height is None


def test_corrupted_images_csv_includes_only_unreadable(tmp_path):
    import csv

    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "good.jpg")
    make_invalid_image(raw_dir / "Alpha" / "broken.jpg")
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])

    output = tmp_path / "audit" / "corrupted_images.csv"
    write_corrupted_images_csv(records, output)

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["filename"] == "broken.jpg"


def test_corrupted_images_csv_valid_empty_schema_when_none_corrupted(tmp_path):
    import csv

    raw_dir = _build_two_class_raw_dir(tmp_path)
    records = build_inventory(raw_dir, MAPPING, [".jpg", ".jpeg"])

    output = tmp_path / "audit" / "corrupted_images.csv"
    write_corrupted_images_csv(records, output)

    with open(output, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)
    assert header is not None and len(header) > 0
    assert rows == []


# --- Class structure validation ---


def test_validate_dataset_config_accepts_matching_mapping():
    config = {
        "class_names": ["Alpha", "Beta"],
        "class_directory_mapping": {"Alpha": "Alpha", "Beta": "Beta"},
        "num_classes": 2,
    }
    validate_dataset_config(config)  # must not raise


def test_validate_dataset_config_rejects_mismatched_mapping():
    config = {
        "class_names": ["Alpha", "Beta"],
        "class_directory_mapping": {"Alpha": "Alpha"},  # missing "Beta"
        "num_classes": 2,
    }
    with pytest.raises(AuditConfigError):
        validate_dataset_config(config)


def test_validate_dataset_config_rejects_num_classes_mismatch():
    config = {
        "class_names": ["Alpha", "Beta"],
        "class_directory_mapping": {"Alpha": "Alpha", "Beta": "Beta"},
        "num_classes": 3,
    }
    with pytest.raises(AuditConfigError):
        validate_dataset_config(config)


def test_validate_class_directories_all_present(tmp_path):
    raw_dir = _build_two_class_raw_dir(tmp_path)
    result = validate_class_directories(raw_dir, MAPPING)
    assert result.ok
    assert result.missing_mapped_directories == {}
    assert result.unexpected_directories == []


def test_validate_class_directories_detects_missing(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    # "Beta Disease [Raw]" missing entirely.
    result = validate_class_directories(raw_dir, MAPPING)
    assert not result.ok
    assert "Beta Disease" in result.missing_mapped_directories


def test_validate_class_directories_detects_unexpected(tmp_path):
    raw_dir = _build_two_class_raw_dir(tmp_path)
    make_image(raw_dir / "Mystery Class" / "x1.jpg")
    result = validate_class_directories(raw_dir, MAPPING)
    assert not result.ok
    assert "Mystery Class" in result.unexpected_directories


def test_validate_class_directories_raises_if_raw_dir_missing(tmp_path):
    with pytest.raises(AuditConfigError):
        validate_class_directories(tmp_path / "does_not_exist", MAPPING)


def test_class_directory_mapping_not_substring_based(tmp_path):
    # A directory that is a substring/superstring of a mapped name must NOT
    # be silently matched — it must show up as unexpected.
    raw_dir = _build_two_class_raw_dir(tmp_path)
    make_image(raw_dir / "Beta Disease" / "wrong.jpg")  # missing "[Raw]" suffix
    result = validate_class_directories(raw_dir, MAPPING)
    assert "Beta Disease" in result.unexpected_directories

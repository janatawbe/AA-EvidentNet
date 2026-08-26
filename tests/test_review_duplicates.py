"""Tests for src.data.review_duplicates: visual contact-sheet generation.

Uses tiny synthetic fixture images only. Verifies raw source files are
never modified by contact-sheet rendering.
"""

from PIL import Image

from src.data.review_duplicates import render_contact_sheet_for_group
from src.utils.hashing import hash_file
from tests.conftest import make_image


def _two_member_row(raw_dir):
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(40, 40), color=(10, 20, 30))
    import shutil

    (raw_dir / "Beta").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Beta" / "b1.jpg")

    return {
        "duplicate_group_id": "DUPGROUP_0001",
        "sha256": "deadbeef" * 8,
        "num_files": 2,
        "classes": "Alpha;Beta",
        "paths": "Alpha|Alpha/a1.jpg;Beta|Beta/b1.jpg",
        "resolution": "UNRESOLVED",
        "resolved_class": "",
        "reviewer": "",
        "notes": "",
    }


def test_contact_sheet_generated_for_two_member_group(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "audit" / "duplicate_review"
    row = _two_member_row(raw_dir)

    output_path = render_contact_sheet_for_group(row, raw_dir, output_dir)

    assert output_path.exists()
    assert output_path.parent == output_dir
    assert output_path.name == "DUPGROUP_0001.png"

    with Image.open(output_path) as sheet:
        assert sheet.format == "PNG"
        assert sheet.width > 0
        assert sheet.height > 0


def test_contact_sheet_does_not_modify_raw_files(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "audit" / "duplicate_review"
    row = _two_member_row(raw_dir)

    hash_before_a = hash_file(raw_dir / "Alpha" / "a1.jpg")
    hash_before_b = hash_file(raw_dir / "Beta" / "b1.jpg")

    render_contact_sheet_for_group(row, raw_dir, output_dir)

    assert hash_file(raw_dir / "Alpha" / "a1.jpg") == hash_before_a
    assert hash_file(raw_dir / "Beta" / "b1.jpg") == hash_before_b
    # And nothing new was written under raw_dir itself.
    raw_files = sorted(p.name for p in (raw_dir / "Alpha").iterdir()) + sorted(
        p.name for p in (raw_dir / "Beta").iterdir()
    )
    assert raw_files == ["a1.jpg", "b1.jpg"]


def test_contact_sheet_multi_file_group_wider_than_two_file_group(tmp_path):
    raw_dir = tmp_path / "raw"
    output_dir = tmp_path / "audit" / "duplicate_review"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(40, 40), color=(1, 1, 1))
    import shutil

    (raw_dir / "Beta").mkdir(parents=True, exist_ok=True)
    (raw_dir / "Gamma").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Beta" / "b1.jpg")
    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Gamma" / "g1.jpg")

    two_member_row = {
        "duplicate_group_id": "DUPGROUP_TWO",
        "sha256": "aaaa",
        "paths": "Alpha|Alpha/a1.jpg;Beta|Beta/b1.jpg",
        "resolution": "UNRESOLVED",
    }
    three_member_row = {
        "duplicate_group_id": "DUPGROUP_THREE",
        "sha256": "bbbb",
        "paths": "Alpha|Alpha/a1.jpg;Beta|Beta/b1.jpg;Gamma|Gamma/g1.jpg",
        "resolution": "UNRESOLVED",
    }

    two_path = render_contact_sheet_for_group(two_member_row, raw_dir, output_dir)
    three_path = render_contact_sheet_for_group(three_member_row, raw_dir, output_dir)

    with Image.open(two_path) as two_img, Image.open(three_path) as three_img:
        assert three_img.width > two_img.width


def test_contact_sheet_raises_clear_error_when_paths_empty(tmp_path):
    import pytest

    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    output_dir = tmp_path / "audit" / "duplicate_review"
    row = {"duplicate_group_id": "DUPGROUP_EMPTY", "sha256": "x", "paths": "", "resolution": "UNRESOLVED"}

    with pytest.raises(ValueError, match="no members"):
        render_contact_sheet_for_group(row, raw_dir, output_dir)


# --- CLI ---


def test_cli_group_id_not_found_returns_error(tmp_path, capsys):
    import yaml

    from src.data.review_duplicates import main

    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    row = _two_member_row(raw_dir)

    import csv

    review_csv = audit_dir / "cross_class_duplicate_review.csv"
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(review_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    config = {"paths": {"raw_dir": str(raw_dir), "audit_dir": str(audit_dir)}}
    config_path = tmp_path / "dataset.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    exit_code = main(["--group-id", "DOES_NOT_EXIST", "--config", str(config_path)])
    assert exit_code == 1
    assert "not found" in capsys.readouterr().err


def test_cli_renders_group_by_id(tmp_path, capsys):
    import csv

    import yaml

    from src.data.review_duplicates import main

    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    row = _two_member_row(raw_dir)

    review_csv = audit_dir / "cross_class_duplicate_review.csv"
    review_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(review_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)

    config = {"paths": {"raw_dir": str(raw_dir), "audit_dir": str(audit_dir)}}
    config_path = tmp_path / "dataset.yaml"
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(config, f)

    exit_code = main(["--group-id", "DUPGROUP_0001", "--config", str(config_path)])
    assert exit_code == 0
    assert (audit_dir / "duplicate_review" / "DUPGROUP_0001.png").exists()

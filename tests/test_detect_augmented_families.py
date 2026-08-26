"""Tests for src.data.detect_augmented_families: naming-pattern heuristics."""

import pytest

from src.data.detect_augmented_families import (
    HIGHLY_SUSPICIOUS,
    NORMAL,
    SUSPICIOUS,
    _build_keyword_set,
    classify_filename,
)

KEYWORDS = [
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
]


def _classify(filename, directory_name="SomeClass"):
    keyword_set = _build_keyword_set(KEYWORDS)
    return classify_filename(filename, directory_name, keyword_set)


@pytest.mark.parametrize(
    "filename",
    [
        "CSCR1.jpg",
        "CSCR142.jpg",
        "Glaucoma_045.jpg",
        "Diabetic123.jpg",
        "img_2024.jpg",
        "patient-001.jpg",
        "0001.jpg",
        "retina_scan_17.jpg",
    ],
)
def test_ordinary_filenames_are_not_flagged(filename):
    classification, matched = _classify(filename)
    assert classification == NORMAL
    assert matched == []


@pytest.mark.parametrize(
    "filename,expected_keyword",
    [
        ("CSCR1_flipped.jpg", "flipped"),
        ("image_augmented_03.jpg", "augmented"),
        ("scan-rotated-90.jpg", "rotated"),
        ("healthy_synthetic_001.jpg", "synthetic"),
        ("sample_generated.jpg", "generated"),
        ("copy_of_scan.jpg", "copy"),
    ],
)
def test_keyword_matches_are_highly_suspicious(filename, expected_keyword):
    classification, matched = _classify(filename)
    assert classification == HIGHLY_SUSPICIOUS
    assert any(expected_keyword in m for m in matched)


def test_keyword_substring_inside_unrelated_word_not_flagged():
    # "Copyright" contains "copy" as a substring but not as a whole word.
    classification, matched = _classify("Copyright_notice.jpg")
    assert classification == NORMAL
    assert matched == []


@pytest.mark.parametrize(
    "filename",
    [
        "scan (1).jpg",
        "scan(2).jpg",
        "image-v2.jpg",
        "image_v3.jpg",
        "photo_dup.jpg",
        "photo-dupe2.jpg",
    ],
)
def test_structural_patterns_are_suspicious_not_highly_suspicious(filename):
    classification, matched = _classify(filename)
    assert classification == SUSPICIOUS
    assert matched != []


def test_double_extension_is_suspicious():
    classification, matched = _classify("scan.jpg.jpg")
    assert classification == SUSPICIOUS
    assert any("double_extension" in m for m in matched)


def test_directory_name_keyword_match_flags_file():
    classification, matched = _classify("a1.jpg", directory_name="Diabetic Retinopathy Augmented")
    assert classification == HIGHLY_SUSPICIOUS
    assert any("augmented" in m for m in matched)


def test_analyze_augmentation_families_and_report_writer(tmp_path):
    import csv

    from src.data.detect_augmented_families import (
        analyze_augmentation_families,
        write_augmentation_report_csv,
    )
    from src.data.records import ImageRecord

    def rec(filename):
        return ImageRecord(
            path=f"Alpha/{filename}",
            filename=filename,
            extension=".jpg",
            class_directory="Alpha",
            canonical_class="Alpha",
            file_size_bytes=100,
            width=20,
            height=20,
            mode="RGB",
            sha256="deadbeef",
            is_readable=True,
            error_message="",
        )

    records = [rec("CSCR1.jpg"), rec("CSCR1_flipped.jpg"), rec("scan (1).jpg")]
    findings = analyze_augmentation_families(records, KEYWORDS)
    assert len(findings) == 3

    output = tmp_path / "augmentation_family_report.csv"
    write_augmentation_report_csv(findings, output)

    with open(output, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # Only flagged (non-normal) files are written by default.
    assert len(rows) == 2
    flagged_filenames = {row["filename"] for row in rows}
    assert flagged_filenames == {"CSCR1_flipped.jpg", "scan (1).jpg"}
    for row in rows:
        assert row["reason"] != ""

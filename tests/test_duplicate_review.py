"""Tests for src.data.duplicate_review: the human-review workflow for
cross-class exact duplicate conflicts. No function under test may ever
infer/assign a resolution automatically — these tests specifically check
that behavior is absent.
"""

import csv

import pytest

from src.data.detect_duplicates import find_exact_duplicate_groups, get_cross_class_duplicate_groups, get_same_class_duplicate_groups
from src.data.duplicate_review import (
    RESOLUTION_EXCLUDE_GROUP,
    RESOLUTION_KEEP_CLASS,
    RESOLUTION_UNRESOLVED,
    REVIEW_COLUMNS,
    ReviewValidationError,
    assert_valid_review_rows,
    build_cross_class_review_rows,
    build_same_class_report_rows,
    build_summary_rows,
    count_unresolved,
    get_unresolved_group_ids,
    load_review_csv,
    merge_review_rows,
    parse_members_field,
    validate_review_rows,
    write_cross_class_review_csv,
)
from src.data.records import ImageRecord

CANONICAL_CLASSES = ["Alpha", "Beta", "Gamma"]


def _record(path, sha256, canonical_class, class_directory=None, filename=None):
    filename = filename or path.split("/")[-1]
    return ImageRecord(
        path=path,
        filename=filename,
        extension=".jpg",
        class_directory=class_directory or canonical_class,
        canonical_class=canonical_class,
        file_size_bytes=100,
        width=20,
        height=20,
        mode="RGB",
        sha256=sha256,
        is_readable=True,
        error_message="",
    )


def _two_class_conflict_and_one_same_class_group():
    records = [
        _record("Alpha/a1.jpg", "hashX", "Alpha"),
        _record("Beta/b1.jpg", "hashX", "Beta"),
        _record("Gamma/g1.jpg", "hashY", "Gamma"),
        _record("Gamma/g2.jpg", "hashY", "Gamma"),
    ]
    groups = find_exact_duplicate_groups(records)
    cross = get_cross_class_duplicate_groups(groups)
    same = get_same_class_duplicate_groups(groups)
    return cross, same


# --- manifest generation ---


def test_build_cross_class_review_rows_one_row_per_group_all_unresolved():
    cross, _ = _two_class_conflict_and_one_same_class_group()
    rows = build_cross_class_review_rows(cross)
    assert len(rows) == 1
    row = rows[0]
    assert row["resolution"] == RESOLUTION_UNRESOLVED
    assert row["resolved_class"] == ""
    assert row["reviewer"] == ""
    assert set(row["classes"].split(";")) == {"Alpha", "Beta"}


def test_build_cross_class_review_rows_represents_every_member():
    cross, _ = _two_class_conflict_and_one_same_class_group()
    rows = build_cross_class_review_rows(cross)
    members = parse_members_field(rows[0]["paths"])
    assert len(members) == 2
    assert ("Alpha", "Alpha/a1.jpg") in members
    assert ("Beta", "Beta/b1.jpg") in members


def test_build_cross_class_review_rows_multi_file_group_represents_all_members():
    records = [
        _record("Alpha/a1.jpg", "hashZ", "Alpha"),
        _record("Beta/b1.jpg", "hashZ", "Beta"),
        _record("Gamma/g1.jpg", "hashZ", "Gamma"),
    ]
    groups = find_exact_duplicate_groups(records)
    cross = get_cross_class_duplicate_groups(groups)
    rows = build_cross_class_review_rows(cross)
    assert len(rows) == 1
    assert int(rows[0]["num_files"]) == 3
    members = parse_members_field(rows[0]["paths"])
    assert len(members) == 3
    assert {m[0] for m in members} == {"Alpha", "Beta", "Gamma"}


def test_deterministic_group_ids_across_reordered_input():
    records = [
        _record("Alpha/a1.jpg", "hashX", "Alpha"),
        _record("Beta/b1.jpg", "hashX", "Beta"),
    ]
    rows_a = build_cross_class_review_rows(
        get_cross_class_duplicate_groups(find_exact_duplicate_groups(records))
    )
    rows_b = build_cross_class_review_rows(
        get_cross_class_duplicate_groups(find_exact_duplicate_groups(list(reversed(records))))
    )
    assert rows_a[0]["duplicate_group_id"] == rows_b[0]["duplicate_group_id"]


def test_write_cross_class_review_csv_schema(tmp_path):
    cross, _ = _two_class_conflict_and_one_same_class_group()
    rows = build_cross_class_review_rows(cross)
    output = tmp_path / "cross_class_duplicate_review.csv"
    write_cross_class_review_csv(rows, output)

    with open(output, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REVIEW_COLUMNS
        loaded = list(reader)
    assert len(loaded) == 1
    assert loaded[0]["resolution"] == RESOLUTION_UNRESOLVED


def test_same_class_report_rows_distinct_from_cross_class():
    _, same = _two_class_conflict_and_one_same_class_group()
    rows = build_same_class_report_rows(same)
    assert len(rows) == 1
    assert rows[0]["canonical_class"] == "Gamma"
    assert "no label conflict" in rows[0]["note"]
    assert "resolution" not in rows[0]  # same-class rows carry no resolution schema


# --- validation ---


def test_validate_review_rows_accepts_all_unresolved():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""},
    ]
    assert validate_review_rows(rows, CANONICAL_CLASSES) == []


def test_validate_review_rows_accepts_keep_class_with_valid_class():
    rows = [
        {
            "duplicate_group_id": "G1",
            "sha256": "h1",
            "resolution": RESOLUTION_KEEP_CLASS,
            "resolved_class": "Alpha",
        }
    ]
    assert validate_review_rows(rows, CANONICAL_CLASSES) == []


def test_validate_review_rows_rejects_keep_class_with_invalid_class():
    rows = [
        {
            "duplicate_group_id": "G1",
            "sha256": "h1",
            "resolution": RESOLUTION_KEEP_CLASS,
            "resolved_class": "NotARealClass",
        }
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES)
    assert len(errors) == 1
    assert "NotARealClass" in errors[0]


def test_validate_review_rows_rejects_keep_class_with_missing_class():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_KEEP_CLASS, "resolved_class": ""}
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES)
    assert len(errors) == 1
    assert "requires a non-empty resolved_class" in errors[0]


def test_validate_review_rows_accepts_exclude_group_with_no_class():
    rows = [
        {
            "duplicate_group_id": "G1",
            "sha256": "h1",
            "resolution": RESOLUTION_EXCLUDE_GROUP,
            "resolved_class": "",
        }
    ]
    assert validate_review_rows(rows, CANONICAL_CLASSES) == []


def test_validate_review_rows_rejects_exclude_group_with_class_set():
    rows = [
        {
            "duplicate_group_id": "G1",
            "sha256": "h1",
            "resolution": RESOLUTION_EXCLUDE_GROUP,
            "resolved_class": "Alpha",
        }
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES)
    assert len(errors) == 1
    assert "must not set resolved_class" in errors[0]


def test_validate_review_rows_rejects_unknown_resolution():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": "MAYBE", "resolved_class": ""}
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES)
    assert len(errors) == 1
    assert "unknown resolution" in errors[0]


def test_validate_review_rows_rejects_duplicate_group():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""},
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""},
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES)
    assert any("more than one row" in e for e in errors)


def test_validate_review_rows_detects_missing_expected_group():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""},
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES, expected_sha256_set={"h1", "h2"})
    assert any("h2" in e for e in errors)


def test_validate_review_rows_detects_unexpected_extra_group():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""},
    ]
    errors = validate_review_rows(rows, CANONICAL_CLASSES, expected_sha256_set=set())
    assert any("h1" in e for e in errors)


def test_assert_valid_review_rows_raises_on_error():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": "GARBAGE", "resolved_class": ""}
    ]
    with pytest.raises(ReviewValidationError):
        assert_valid_review_rows(rows, CANONICAL_CLASSES)


def test_assert_valid_review_rows_passes_silently_when_valid():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""}
    ]
    assert_valid_review_rows(rows, CANONICAL_CLASSES)  # must not raise


# --- unresolved detection ---


def test_count_unresolved():
    rows = [
        {"duplicate_group_id": "G1", "resolution": RESOLUTION_UNRESOLVED},
        {"duplicate_group_id": "G2", "resolution": RESOLUTION_KEEP_CLASS},
        {"duplicate_group_id": "G3", "resolution": RESOLUTION_UNRESOLVED},
    ]
    assert count_unresolved(rows) == 2
    assert get_unresolved_group_ids(rows) == ["G1", "G3"]


# --- merge preserves human resolutions across re-runs ---


def test_merge_preserves_existing_resolution_by_sha256():
    existing = [
        {
            "duplicate_group_id": "DUPGROUP_0001",
            "sha256": "hashX",
            "num_files": 2,
            "classes": "Alpha;Beta",
            "paths": "Alpha|Alpha/a1.jpg;Beta|Beta/b1.jpg",
            "resolution": RESOLUTION_KEEP_CLASS,
            "resolved_class": "Alpha",
            "reviewer": "dr_jane",
            "notes": "confirmed via chart review",
        }
    ]
    fresh = [
        {
            "duplicate_group_id": "DUPGROUP_0001",
            "sha256": "hashX",
            "num_files": 2,
            "classes": "Alpha;Beta",
            "paths": "Alpha|Alpha/a1.jpg;Beta|Beta/b1.jpg",
            "resolution": RESOLUTION_UNRESOLVED,
            "resolved_class": "",
            "reviewer": "",
            "notes": "",
        }
    ]
    merged = merge_review_rows(existing, fresh)
    assert len(merged) == 1
    assert merged[0]["resolution"] == RESOLUTION_KEEP_CLASS
    assert merged[0]["resolved_class"] == "Alpha"
    assert merged[0]["reviewer"] == "dr_jane"


def test_merge_new_group_not_in_existing_starts_unresolved():
    existing = []
    fresh = [
        {
            "duplicate_group_id": "DUPGROUP_0002",
            "sha256": "hashY",
            "num_files": 2,
            "classes": "Alpha;Gamma",
            "paths": "Alpha|Alpha/a2.jpg;Gamma|Gamma/g1.jpg",
            "resolution": RESOLUTION_UNRESOLVED,
            "resolved_class": "",
            "reviewer": "",
            "notes": "",
        }
    ]
    merged = merge_review_rows(existing, fresh)
    assert merged[0]["resolution"] == RESOLUTION_UNRESOLVED


def test_merge_never_invents_a_resolution_for_new_groups():
    # A regression guard: whatever merge_review_rows does, a group with no
    # prior human review must always come out UNRESOLVED, never guessed.
    fresh = build_cross_class_review_rows(
        get_cross_class_duplicate_groups(
            find_exact_duplicate_groups(
                [
                    _record("Alpha/a1.jpg", "hashQ", "Alpha"),
                    _record("Beta/b1.jpg", "hashQ", "Beta"),
                ]
            )
        )
    )
    merged = merge_review_rows([], fresh)
    assert all(r["resolution"] == RESOLUTION_UNRESOLVED for r in merged)
    assert all(r["resolved_class"] == "" for r in merged)


def test_load_review_csv_returns_empty_list_when_missing(tmp_path):
    assert load_review_csv(tmp_path / "does_not_exist.csv") == []


# --- summary ---


def test_build_summary_rows_counts_totals_and_resolution_states():
    rows = [
        {"duplicate_group_id": "G1", "sha256": "h1", "num_files": 2, "classes": "Alpha;Beta", "resolution": RESOLUTION_UNRESOLVED, "resolved_class": ""},
        {"duplicate_group_id": "G2", "sha256": "h2", "num_files": 3, "classes": "Alpha;Gamma", "resolution": RESOLUTION_KEEP_CLASS, "resolved_class": "Alpha"},
        {"duplicate_group_id": "G3", "sha256": "h3", "num_files": 2, "classes": "Beta;Gamma", "resolution": RESOLUTION_EXCLUDE_GROUP, "resolved_class": ""},
    ]
    summary = build_summary_rows(rows, CANONICAL_CLASSES)
    as_dict = {(r["metric"], r["key"]): r["value"] for r in summary}

    assert as_dict[("total_conflicting_groups", "")] == 3
    assert as_dict[("total_affected_files", "")] == 7
    assert as_dict[("groups_unresolved", "")] == 1
    assert as_dict[("groups_excluded", "")] == 1
    assert as_dict[("groups_resolved_to_class", "Alpha")] == 1
    assert as_dict[("groups_resolved_to_class", "Beta")] == 0
    assert as_dict[("groups_by_class_pair", "Alpha;Beta")] == 1

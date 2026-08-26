"""Tests for src.data.eligibility: the eligibility layer derived from the
raw-data audit + the human-reviewed cross-class duplicate manifest.

No function under test may ever assign or infer a canonical class label,
or hard-code which files are excluded — these tests specifically probe
that the exclusion set is entirely DERIVED from duplicate groups + review
resolution state.
"""

import pytest

from src.data.detect_duplicates import find_exact_duplicate_groups
from src.data.duplicate_review import (
    RESOLUTION_EXCLUDE_GROUP,
    RESOLUTION_KEEP_CLASS,
    RESOLUTION_UNRESOLVED,
)
from src.data.eligibility import (
    ELIGIBILITY_COLUMNS,
    ELIGIBLE_FALSE,
    ELIGIBLE_TRUE,
    EXCLUSION_HUMAN_EXCLUDED_CROSS_CLASS,
    EXCLUSION_SAME_CLASS_POLICY,
    EXCLUSION_UNRESOLVED_CROSS_CLASS,
    EligibilityValidationError,
    SplitGuardError,
    assert_split_is_valid,
    assert_valid_eligibility_rows,
    build_eligibility_rows,
    build_eligibility_summary_rows,
    build_eligible_class_distribution_rows,
    validate_eligibility_rows,
)
from src.data.records import ImageRecord

CANONICAL_CLASSES = ["Alpha", "Beta", "Gamma"]


def _record(path, sha256, canonical_class):
    filename = path.split("/")[-1]
    return ImageRecord(
        path=path,
        filename=filename,
        extension=".jpg",
        class_directory=canonical_class,
        canonical_class=canonical_class,
        file_size_bytes=100,
        width=20,
        height=20,
        mode="RGB",
        sha256=sha256,
        is_readable=True,
        error_message="",
    )


def _fixture_records():
    return [
        _record("Alpha/a1.jpg", "hashUnique1", "Alpha"),  # not duplicated
        _record("Alpha/a2.jpg", "hashCross", "Alpha"),  # cross-class conflict member
        _record("Beta/b1.jpg", "hashCross", "Beta"),  # cross-class conflict member
        _record("Gamma/g1.jpg", "hashSame", "Gamma"),  # same-class duplicate
        _record("Gamma/g2.jpg", "hashSame", "Gamma"),  # same-class duplicate
    ]


def _review_row(sha256_group, group_id, resolution=RESOLUTION_UNRESOLVED, resolved_class=""):
    return {
        "duplicate_group_id": group_id,
        "sha256": sha256_group,
        "resolution": resolution,
        "resolved_class": resolved_class,
    }


# --- manifest coverage: every raw image exactly once ---


def test_every_raw_image_has_exactly_one_eligibility_row():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id)]

    rows = build_eligibility_rows(records, groups, review_rows)
    assert len(rows) == len(records)
    assert {r["path"] for r in rows} == {r.path for r in records}


def test_eligibility_row_schema_matches_expected_columns():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    rows = build_eligibility_rows(records, groups, [])
    assert set(rows[0].keys()) == set(ELIGIBILITY_COLUMNS)


# --- unresolved cross-class -> excluded ---


def test_unresolved_cross_class_group_is_excluded():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]

    rows = build_eligibility_rows(records, groups, review_rows)
    by_path = {r["path"]: r for r in rows}

    for path in ("Alpha/a2.jpg", "Beta/b1.jpg"):
        row = by_path[path]
        assert row["eligible"] == ELIGIBLE_FALSE
        assert row["exclusion_reason"] == EXCLUSION_UNRESOLVED_CROSS_CLASS
        assert row["duplicate_type"] == "cross_class"
        assert row["duplicate_group_id"] == cross_group.group_id


def test_unaffected_image_remains_eligible_and_uncategorized():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]

    rows = build_eligibility_rows(records, groups, review_rows)
    by_path = {r["path"]: r for r in rows}

    row = by_path["Alpha/a1.jpg"]
    assert row["eligible"] == ELIGIBLE_TRUE
    assert row["exclusion_reason"] == ""
    assert row["duplicate_type"] == ""
    assert row["duplicate_group_id"] == ""


# --- resolved KEEP_CLASS -> eligible, no label ever invented ---


def test_keep_class_resolution_makes_group_eligible_without_changing_canonical_class():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_KEEP_CLASS, "Alpha")]

    rows = build_eligibility_rows(records, groups, review_rows)
    by_path = {r["path"]: r for r in rows}

    a2 = by_path["Alpha/a2.jpg"]
    b1 = by_path["Beta/b1.jpg"]
    assert a2["eligible"] == ELIGIBLE_TRUE
    assert b1["eligible"] == ELIGIBLE_TRUE
    assert a2["exclusion_reason"] == ""
    assert b1["exclusion_reason"] == ""
    # canonical_class must remain each file's ORIGINAL raw-directory label -
    # never overwritten with the adjudicated "Alpha" resolved_class.
    assert a2["canonical_class"] == "Alpha"
    assert b1["canonical_class"] == "Beta"


# --- resolved EXCLUDE_GROUP -> excluded, distinct reason from UNRESOLVED ---


def test_exclude_group_resolution_is_excluded_with_distinct_reason():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_EXCLUDE_GROUP)]

    rows = build_eligibility_rows(records, groups, review_rows)
    by_path = {r["path"]: r for r in rows}

    for path in ("Alpha/a2.jpg", "Beta/b1.jpg"):
        row = by_path[path]
        assert row["eligible"] == ELIGIBLE_FALSE
        assert row["exclusion_reason"] == EXCLUSION_HUMAN_EXCLUDED_CROSS_CLASS
        assert row["exclusion_reason"] != EXCLUSION_UNRESOLVED_CROSS_CLASS


# --- same-class duplicates -> eligible by default ---


def test_same_class_duplicate_is_eligible_by_default():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    rows = build_eligibility_rows(records, groups, [])
    by_path = {r["path"]: r for r in rows}

    for path in ("Gamma/g1.jpg", "Gamma/g2.jpg"):
        row = by_path[path]
        assert row["eligible"] == ELIGIBLE_TRUE
        assert row["duplicate_type"] == "same_class"
        assert row["exclusion_reason"] == ""


def test_same_class_duplicate_excluded_when_policy_overridden():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    rows = build_eligibility_rows(
        records, groups, [], duplicate_policy={"same_class_exact_duplicate": "exclude"}
    )
    by_path = {r["path"]: r for r in rows}
    assert by_path["Gamma/g1.jpg"]["eligible"] == ELIGIBLE_FALSE
    assert by_path["Gamma/g1.jpg"]["exclusion_reason"] == EXCLUSION_SAME_CLASS_POLICY


def test_policy_config_is_honored_for_unresolved_cross_class():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]

    # Explicit non-default override: keep unresolved groups eligible.
    rows = build_eligibility_rows(
        records, groups, review_rows, duplicate_policy={"unresolved_cross_class": "keep"}
    )
    by_path = {r["path"]: r for r in rows}
    assert by_path["Alpha/a2.jpg"]["eligible"] == ELIGIBLE_TRUE
    assert by_path["Alpha/a2.jpg"]["exclusion_reason"] == ""


# --- validation ---


def test_validate_eligibility_rows_accepts_well_formed_rows():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "Alpha",
            "sha256": "h1",
            "eligible": ELIGIBLE_TRUE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        }
    ]
    assert validate_eligibility_rows(rows, CANONICAL_CLASSES) == []


def test_validate_eligibility_rows_rejects_unknown_exclusion_reason():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "Alpha",
            "sha256": "h1",
            "eligible": ELIGIBLE_FALSE,
            "exclusion_reason": "some_made_up_reason",
            "duplicate_group_id": "G1",
            "duplicate_type": "cross_class",
        }
    ]
    errors = validate_eligibility_rows(rows, CANONICAL_CLASSES)
    assert any("unknown exclusion_reason" in e for e in errors)


def test_validate_eligibility_rows_rejects_unknown_canonical_class():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "NotARealClass",
            "sha256": "h1",
            "eligible": ELIGIBLE_TRUE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        }
    ]
    errors = validate_eligibility_rows(rows, CANONICAL_CLASSES)
    assert any("unknown canonical_class" in e for e in errors)


def test_validate_eligibility_rows_detects_missing_eligibility_decision():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "Alpha",
            "sha256": "h1",
            "eligible": ELIGIBLE_TRUE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        }
    ]
    errors = validate_eligibility_rows(rows, CANONICAL_CLASSES, expected_paths={"Alpha/a1.jpg", "Beta/b1.jpg"})
    assert any("Beta/b1.jpg" in e and "no eligibility decision" in e for e in errors)


def test_validate_eligibility_rows_detects_conflicting_records_for_same_path():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "Alpha",
            "sha256": "h1",
            "eligible": ELIGIBLE_TRUE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        },
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "Alpha",
            "sha256": "h1",
            "eligible": ELIGIBLE_FALSE,
            "exclusion_reason": EXCLUSION_UNRESOLVED_CROSS_CLASS,
            "duplicate_group_id": "G1",
            "duplicate_type": "cross_class",
        },
    ]
    errors = validate_eligibility_rows(rows, CANONICAL_CLASSES)
    assert any("conflicting eligibility records" in e for e in errors)


def test_validate_eligibility_rows_requires_reason_when_excluded():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "Alpha",
            "sha256": "h1",
            "eligible": ELIGIBLE_FALSE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        }
    ]
    errors = validate_eligibility_rows(rows, CANONICAL_CLASSES)
    assert any("requires a non-empty exclusion_reason" in e for e in errors)


def test_assert_valid_eligibility_rows_raises_on_error():
    rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "NotReal",
            "sha256": "h1",
            "eligible": ELIGIBLE_TRUE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        }
    ]
    with pytest.raises(EligibilityValidationError):
        assert_valid_eligibility_rows(rows, CANONICAL_CLASSES)


# --- post-review class statistics ---


def test_eligible_class_distribution_counts_are_correct():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]

    rows = build_eligibility_rows(records, groups, review_rows)
    distribution = build_eligible_class_distribution_rows(rows, CANONICAL_CLASSES)
    by_class = {r["canonical_class"]: r for r in distribution}

    assert by_class["Alpha"]["raw_count"] == 2
    assert by_class["Alpha"]["excluded_count"] == 1  # a2.jpg unresolved cross-class
    assert by_class["Alpha"]["eligible_count"] == 1
    assert by_class["Beta"]["raw_count"] == 1
    assert by_class["Beta"]["excluded_count"] == 1
    assert by_class["Gamma"]["raw_count"] == 2
    assert by_class["Gamma"]["excluded_count"] == 0
    assert by_class["Gamma"]["eligible_count"] == 2


def test_eligible_class_distribution_includes_zero_count_classes():
    rows = []
    distribution = build_eligible_class_distribution_rows(rows, CANONICAL_CLASSES)
    assert {r["canonical_class"] for r in distribution} == set(CANONICAL_CLASSES)
    assert all(r["raw_count"] == 0 for r in distribution)
    assert all(r["eligible_percentage"] == 0.0 for r in distribution)


def test_eligibility_summary_counts():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    same_class_groups = [g for g in groups if not g.is_cross_class]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]

    eligibility_rows = build_eligibility_rows(records, groups, review_rows)
    summary = build_eligibility_summary_rows(eligibility_rows, same_class_groups, review_rows)
    as_dict = {r["metric"]: r["value"] for r in summary}

    assert as_dict["total_raw_images"] == 5
    assert as_dict["excluded_images"] == 2
    assert as_dict["eligible_images"] == 3
    assert as_dict["same_class_duplicate_groups"] == 1
    assert as_dict["unresolved_groups"] == 1
    assert as_dict["resolved_groups"] == 0
    assert as_dict["excluded_groups"] == 0


# --- future split guard ---


def _base_eligibility_rows():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_KEEP_CLASS, "Alpha")]
    rows = build_eligibility_rows(records, groups, review_rows)
    return rows, review_rows, groups


def test_split_guard_rejects_excluded_file_in_split_assignment():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]
    rows = build_eligibility_rows(records, groups, review_rows)

    split_assignment = {"Alpha/a2.jpg": "train"}  # excluded (unresolved cross-class)
    with pytest.raises(SplitGuardError, match="excluded path"):
        assert_split_is_valid(
            rows, split_assignment, CANONICAL_CLASSES, review_rows,
            require_review_before_split=False,
        )


def test_split_guard_rejects_duplicate_group_split_across_splits():
    rows, review_rows, groups = _base_eligibility_rows()
    same_class_group = [g for g in groups if not g.is_cross_class][0]

    split_assignment = {
        "Gamma/g1.jpg": "train",
        "Gamma/g2.jpg": "val",  # same duplicate group as g1 -> must match
    }
    with pytest.raises(SplitGuardError, match="split across multiple splits"):
        assert_split_is_valid(
            rows, split_assignment, CANONICAL_CLASSES, review_rows,
            require_review_before_split=False,
        )


def test_split_guard_accepts_duplicate_group_kept_together():
    rows, review_rows, _ = _base_eligibility_rows()
    split_assignment = {"Gamma/g1.jpg": "train", "Gamma/g2.jpg": "train"}
    assert_split_is_valid(  # must not raise
        rows, split_assignment, CANONICAL_CLASSES, review_rows,
        require_review_before_split=False,
    )


def test_split_guard_rejects_when_unresolved_groups_present_and_review_required():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]
    rows = build_eligibility_rows(records, groups, review_rows)

    with pytest.raises(SplitGuardError, match="human review"):
        assert_split_is_valid(
            rows, {}, CANONICAL_CLASSES, review_rows, require_review_before_split=True
        )


def test_split_guard_allows_proceeding_when_review_not_required_and_no_bad_assignment():
    records = _fixture_records()
    groups = find_exact_duplicate_groups(records)
    cross_group = [g for g in groups if g.is_cross_class][0]
    review_rows = [_review_row("hashCross", cross_group.group_id, RESOLUTION_UNRESOLVED)]
    rows = build_eligibility_rows(records, groups, review_rows)

    assert_split_is_valid(  # must not raise
        rows, {}, CANONICAL_CLASSES, review_rows, require_review_before_split=False
    )


def test_split_guard_rejects_path_with_no_eligibility_record():
    rows, review_rows, _ = _base_eligibility_rows()
    with pytest.raises(SplitGuardError, match="no eligibility record"):
        assert_split_is_valid(
            rows, {"Unknown/x.jpg": "train"}, CANONICAL_CLASSES, review_rows,
            require_review_before_split=False,
        )


def test_split_guard_propagates_eligibility_validation_error_for_malformed_manifest():
    bad_rows = [
        {
            "path": "Alpha/a1.jpg",
            "canonical_class": "NotReal",
            "sha256": "h1",
            "eligible": ELIGIBLE_TRUE,
            "exclusion_reason": "",
            "duplicate_group_id": "",
            "duplicate_type": "",
        }
    ]
    with pytest.raises(EligibilityValidationError):
        assert_split_is_valid(bad_rows, {}, CANONICAL_CLASSES, [], require_review_before_split=False)

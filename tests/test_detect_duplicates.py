"""Tests for src.data.detect_duplicates: exact SHA-256 duplicate detection."""

from src.data.detect_duplicates import (
    find_exact_duplicate_groups,
    get_cross_class_duplicate_groups,
    get_same_class_duplicate_groups,
)
from src.data.records import ImageRecord


def _record(path, sha256, canonical_class="Alpha", class_directory="Alpha", size=100):
    filename = path.split("/")[-1]
    return ImageRecord(
        path=path,
        filename=filename,
        extension=".jpg",
        class_directory=class_directory,
        canonical_class=canonical_class,
        file_size_bytes=size,
        width=20,
        height=20,
        mode="RGB",
        sha256=sha256,
        is_readable=True,
        error_message="",
    )


def test_identical_bytes_different_filenames_grouped():
    records = [
        _record("Alpha/a1.jpg", sha256="hash1"),
        _record("Alpha/a2.jpg", sha256="hash1"),
        _record("Alpha/a3.jpg", sha256="hash2"),
    ]
    groups = find_exact_duplicate_groups(records)
    assert len(groups) == 1
    assert groups[0].sha256 == "hash1"
    assert groups[0].size == 2


def test_different_images_not_marked_duplicates():
    records = [
        _record("Alpha/a1.jpg", sha256="hash1"),
        _record("Alpha/a2.jpg", sha256="hash2"),
        _record("Alpha/a3.jpg", sha256="hash3"),
    ]
    groups = find_exact_duplicate_groups(records)
    assert groups == []


def test_duplicate_group_across_directories_same_class():
    records = [
        _record("Alpha/sub1/a1.jpg", sha256="hash1", class_directory="Alpha"),
        _record("Alpha/sub2/a2.jpg", sha256="hash1", class_directory="Alpha"),
    ]
    groups = find_exact_duplicate_groups(records)
    assert len(groups) == 1
    assert not groups[0].is_cross_class


def test_cross_class_duplicate_detected():
    records = [
        _record("Alpha/a1.jpg", sha256="hash1", canonical_class="Alpha", class_directory="Alpha"),
        _record(
            "Beta/b1.jpg", sha256="hash1", canonical_class="Beta", class_directory="Beta [Raw]"
        ),
    ]
    groups = find_exact_duplicate_groups(records)
    assert len(groups) == 1
    assert groups[0].is_cross_class
    assert groups[0].classes == ["Alpha", "Beta"]

    cross = get_cross_class_duplicate_groups(groups)
    same = get_same_class_duplicate_groups(groups)
    assert len(cross) == 1
    assert len(same) == 0


def test_same_class_and_cross_class_groups_both_reported_distinctly():
    records = [
        _record("Alpha/a1.jpg", sha256="hash1", canonical_class="Alpha", class_directory="Alpha"),
        _record("Alpha/a2.jpg", sha256="hash1", canonical_class="Alpha", class_directory="Alpha"),
        _record("Alpha/a3.jpg", sha256="hash2", canonical_class="Alpha", class_directory="Alpha"),
        _record(
            "Beta/b1.jpg", sha256="hash2", canonical_class="Beta", class_directory="Beta [Raw]"
        ),
    ]
    groups = find_exact_duplicate_groups(records)
    assert len(groups) == 2

    cross = get_cross_class_duplicate_groups(groups)
    same = get_same_class_duplicate_groups(groups)
    assert len(cross) == 1
    assert len(same) == 1
    assert cross[0].sha256 == "hash2"
    assert same[0].sha256 == "hash1"


def test_group_ids_are_deterministic_and_unique():
    records = [
        _record("Alpha/a1.jpg", sha256="zzz"),
        _record("Alpha/a2.jpg", sha256="zzz"),
        _record("Alpha/b1.jpg", sha256="aaa"),
        _record("Alpha/b2.jpg", sha256="aaa"),
    ]
    groups_first = find_exact_duplicate_groups(records)
    groups_second = find_exact_duplicate_groups(list(reversed(records)))

    ids_first = [(g.group_id, g.sha256) for g in groups_first]
    ids_second = [(g.group_id, g.sha256) for g in groups_second]
    assert ids_first == ids_second  # independent of input order
    assert len({g.group_id for g in groups_first}) == len(groups_first)
    # Lowest sha256 ("aaa") gets group index 1 (sorted ascending).
    assert groups_first[0].sha256 == "aaa"


def test_build_duplicate_report_rows_one_row_per_file():
    from src.data.detect_duplicates import build_duplicate_report_rows

    records = [
        _record("Alpha/a1.jpg", sha256="hash1"),
        _record("Alpha/a2.jpg", sha256="hash1"),
    ]
    groups = find_exact_duplicate_groups(records)
    rows = build_duplicate_report_rows(groups)
    assert len(rows) == 2
    assert all(row["duplicate_group_id"] == "DUPGROUP_0001" for row in rows)
    assert all(row["is_cross_class"] is False for row in rows)

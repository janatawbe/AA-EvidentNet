"""Tests for src.data.dataset_statistics: pure summary functions over records."""

from src.data.dataset_statistics import (
    build_class_distribution_rows,
    count_by_class,
    extension_distribution,
    file_size_stats,
    image_dimension_stats,
    mode_distribution,
    percentage_by_class,
    total_image_count,
)
from src.data.records import ImageRecord


def _record(
    canonical_class,
    filename="a.jpg",
    width=100,
    height=200,
    mode="RGB",
    extension=".jpg",
    file_size_bytes=1000,
    is_readable=True,
):
    return ImageRecord(
        path=f"{canonical_class}/{filename}",
        filename=filename,
        extension=extension,
        class_directory=canonical_class,
        canonical_class=canonical_class,
        file_size_bytes=file_size_bytes,
        width=width,
        height=height,
        mode=mode,
        sha256="deadbeef",
        is_readable=is_readable,
        error_message="",
    )


def test_total_image_count():
    records = [_record("Alpha"), _record("Alpha"), _record("Beta")]
    assert total_image_count(records) == 3


def test_total_image_count_empty():
    assert total_image_count([]) == 0


def test_count_by_class():
    records = [_record("Alpha"), _record("Alpha"), _record("Beta")]
    assert count_by_class(records) == {"Alpha": 2, "Beta": 1}


def test_percentage_by_class():
    records = [_record("Alpha")] * 3 + [_record("Beta")] * 1
    pct = percentage_by_class(records)
    assert pct["Alpha"] == 75.0
    assert pct["Beta"] == 25.0


def test_percentage_by_class_empty_dataset():
    assert percentage_by_class([]) == {}


def test_extension_distribution():
    records = [
        _record("Alpha", filename="a.jpg", extension=".jpg"),
        _record("Alpha", filename="b.jpeg", extension=".jpeg"),
        _record("Alpha", filename="c.jpg", extension=".jpg"),
    ]
    assert extension_distribution(records) == {".jpeg": 1, ".jpg": 2}


def test_mode_distribution():
    records = [
        _record("Alpha", mode="RGB"),
        _record("Alpha", mode="RGB"),
        _record("Alpha", mode="L"),
    ]
    assert mode_distribution(records) == {"L": 1, "RGB": 2}


def test_image_dimension_stats():
    records = [
        _record("Alpha", width=100, height=200),
        _record("Alpha", width=200, height=400),
    ]
    stats = image_dimension_stats(records)
    assert stats["width"]["min"] == 100
    assert stats["width"]["max"] == 200
    assert stats["width"]["mean"] == 150.0
    assert stats["height"]["min"] == 200
    assert stats["height"]["max"] == 400


def test_image_dimension_stats_excludes_unreadable():
    records = [
        _record("Alpha", width=100, height=200),
        _record("Alpha", width=None, height=None, is_readable=False),
    ]
    stats = image_dimension_stats(records)
    assert stats["width"]["count"] == 1
    assert stats["width"]["min"] == 100


def test_file_size_stats():
    records = [
        _record("Alpha", file_size_bytes=1000),
        _record("Alpha", file_size_bytes=3000),
    ]
    stats = file_size_stats(records)
    assert stats["min"] == 1000.0
    assert stats["max"] == 3000.0
    assert stats["mean"] == 2000.0


def test_build_class_distribution_rows_includes_zero_count_classes():
    records = [_record("Alpha"), _record("Alpha")]
    mapping = {"Alpha": "Alpha", "Beta": "Beta [Raw]"}
    rows = build_class_distribution_rows(records, mapping)
    by_class = {r["canonical_class"]: r for r in rows}

    assert by_class["Alpha"]["image_count"] == 2
    assert by_class["Alpha"]["percentage"] == 100.0
    assert by_class["Beta"]["image_count"] == 0
    assert by_class["Beta"]["percentage"] == 0.0
    assert by_class["Beta"]["raw_directory"] == "Beta [Raw]"


def test_build_class_distribution_rows_sorted_by_canonical_class():
    records = [_record("Zeta"), _record("Alpha")]
    mapping = {"Zeta": "Zeta", "Alpha": "Alpha"}
    rows = build_class_distribution_rows(records, mapping)
    assert [r["canonical_class"] for r in rows] == ["Alpha", "Zeta"]

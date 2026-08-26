"""Reusable, dependency-light statistics over a dataset inventory.

Every function here takes a list of image records (see
`src.data.audit_dataset.ImageRecord`) and returns plain dicts/numbers. None
of these functions read from disk, hard-code expected counts, or depend on
any particular dataset — they simply summarize whatever records they are
given, so they work identically on the real DS2 dataset and on tiny test
fixtures.
"""

import statistics as _statistics
from collections import Counter
from typing import Any, Dict, Iterable, List

from pathlib import Path

from src.data.records import ImageRecord, write_csv

CLASS_DISTRIBUTION_COLUMNS = ["canonical_class", "raw_directory", "image_count", "percentage"]


def total_image_count(records: Iterable[ImageRecord]) -> int:
    return sum(1 for _ in records)


def count_by_class(records: Iterable[ImageRecord]) -> Dict[str, int]:
    counts: Counter = Counter(r.canonical_class for r in records)
    return dict(sorted(counts.items()))


def percentage_by_class(records: Iterable[ImageRecord]) -> Dict[str, float]:
    records = list(records)
    total = len(records)
    counts = count_by_class(records)
    if total == 0:
        return {cls: 0.0 for cls in counts}
    return {cls: round(100.0 * count / total, 4) for cls, count in counts.items()}


def extension_distribution(records: Iterable[ImageRecord]) -> Dict[str, int]:
    counts: Counter = Counter(r.extension.lower() for r in records)
    return dict(sorted(counts.items()))


def mode_distribution(records: Iterable[ImageRecord]) -> Dict[str, int]:
    counts: Counter = Counter(r.mode for r in records if r.mode is not None)
    return dict(sorted(counts.items()))


def _numeric_summary(values: List[float]) -> Dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
            "stdev": None,
        }
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": round(_statistics.fmean(values), 4),
        "median": _statistics.median(values),
        "stdev": round(_statistics.pstdev(values), 4) if len(values) > 1 else 0.0,
    }


def image_dimension_stats(records: Iterable[ImageRecord]) -> Dict[str, Any]:
    """Summarize width/height for records with known dimensions.

    Unreadable images (width/height is None) are excluded, since their
    dimensions are unknown, not zero.
    """
    widths = [r.width for r in records if r.width is not None]
    heights = [r.height for r in records if r.height is not None]
    return {
        "width": _numeric_summary(widths),
        "height": _numeric_summary(heights),
    }


def file_size_stats(records: Iterable[ImageRecord]) -> Dict[str, Any]:
    sizes = [float(r.file_size_bytes) for r in records]
    return _numeric_summary(sizes)


def readable_count(records: Iterable[ImageRecord]) -> Dict[str, int]:
    records = list(records)
    readable = sum(1 for r in records if r.is_readable)
    unreadable = len(records) - readable
    return {"readable": readable, "unreadable": unreadable}


def build_class_distribution_rows(
    records: Iterable[ImageRecord], class_directory_mapping: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Build class_distribution.csv rows for every CONFIGURED class.

    Every canonical class in class_directory_mapping gets a row, even if its
    observed count is zero (a zero-count configured class is itself an
    important audit finding, not something to hide by omission). Counts and
    percentages are always computed from the given records, never from a
    hard-coded expectation.
    """
    records = list(records)
    total = len(records)
    counts = count_by_class(records)

    rows = []
    for canonical_class in sorted(class_directory_mapping.keys()):
        count = counts.get(canonical_class, 0)
        percentage = round(100.0 * count / total, 4) if total > 0 else 0.0
        rows.append(
            {
                "canonical_class": canonical_class,
                "raw_directory": class_directory_mapping[canonical_class],
                "image_count": count,
                "percentage": percentage,
            }
        )
    return rows


def write_class_distribution_csv(
    records: Iterable[ImageRecord],
    class_directory_mapping: Dict[str, str],
    output_path: Path,
) -> None:
    rows = build_class_distribution_rows(records, class_directory_mapping)
    write_csv(rows, CLASS_DISTRIBUTION_COLUMNS, output_path)


def summarize(records: Iterable[ImageRecord]) -> Dict[str, Any]:
    """Convenience bundle of every statistic above, for logging/summaries."""
    records = list(records)
    return {
        "total_images": total_image_count(records),
        "count_by_class": count_by_class(records),
        "percentage_by_class": percentage_by_class(records),
        "extension_distribution": extension_distribution(records),
        "mode_distribution": mode_distribution(records),
        "image_dimensions": image_dimension_stats(records),
        "file_size_bytes": file_size_stats(records),
        "readability": readable_count(records),
    }

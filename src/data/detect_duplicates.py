"""Exact (byte-identical) duplicate detection using SHA-256 content identity.

This module only detects and reports duplicates. It never deletes, moves,
or merges files, and it never uses perceptual/similarity hashing — two
images are only considered duplicates if their raw file bytes hash to the
same SHA-256 digest.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Union

from src.data.records import ImageRecord, write_csv

DUPLICATE_REPORT_COLUMNS = [
    "duplicate_group_id",
    "sha256",
    "group_size",
    "is_cross_class",
    "classes_in_group",
    "file_size_bytes",
    "canonical_class",
    "class_directory",
    "filename",
    "path",
]


@dataclass(frozen=True)
class DuplicateGroup:
    group_id: str
    sha256: str
    records: List[ImageRecord]

    @property
    def size(self) -> int:
        return len(self.records)

    @property
    def classes(self) -> List[str]:
        return sorted({r.canonical_class for r in self.records})

    @property
    def is_cross_class(self) -> bool:
        return len(self.classes) > 1


def find_exact_duplicate_groups(records: Iterable[ImageRecord]) -> List[DuplicateGroup]:
    """Group records by SHA-256, keeping only groups with 2+ members.

    Group ordering (and hence group_id assignment) is deterministic: sorted
    by sha256 ascending. Records within a group are sorted by
    (canonical_class, filename, path) for stable, reviewable output.
    """
    by_hash: Dict[str, List[ImageRecord]] = defaultdict(list)
    for record in records:
        by_hash[record.sha256].append(record)

    duplicate_hashes = sorted(h for h, group in by_hash.items() if len(group) > 1)

    groups = []
    for index, sha256 in enumerate(duplicate_hashes, start=1):
        group_records = sorted(
            by_hash[sha256], key=lambda r: (r.canonical_class, r.filename, r.path)
        )
        groups.append(
            DuplicateGroup(
                group_id=f"DUPGROUP_{index:04d}",
                sha256=sha256,
                records=group_records,
            )
        )
    return groups


def get_cross_class_duplicate_groups(groups: Iterable[DuplicateGroup]) -> List[DuplicateGroup]:
    return [g for g in groups if g.is_cross_class]


def get_same_class_duplicate_groups(groups: Iterable[DuplicateGroup]) -> List[DuplicateGroup]:
    return [g for g in groups if not g.is_cross_class]


def build_duplicate_report_rows(groups: Iterable[DuplicateGroup]) -> List[dict]:
    rows = []
    for group in groups:
        classes_in_group = ";".join(group.classes)
        for record in group.records:
            rows.append(
                {
                    "duplicate_group_id": group.group_id,
                    "sha256": group.sha256,
                    "group_size": group.size,
                    "is_cross_class": group.is_cross_class,
                    "classes_in_group": classes_in_group,
                    "file_size_bytes": record.file_size_bytes,
                    "canonical_class": record.canonical_class,
                    "class_directory": record.class_directory,
                    "filename": record.filename,
                    "path": record.path,
                }
            )
    return rows


def write_duplicate_report_csv(
    groups: Iterable[DuplicateGroup], output_path: Union[str, Path]
) -> None:
    rows = build_duplicate_report_rows(groups)
    write_csv(rows, DUPLICATE_REPORT_COLUMNS, output_path)

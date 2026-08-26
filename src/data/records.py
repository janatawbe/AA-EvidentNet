"""Shared data model and CSV writing helper for the dataset audit.

Kept in its own module (rather than inside audit_dataset.py) so that
detect_duplicates.py, detect_augmented_families.py, and
dataset_statistics.py can all depend on ImageRecord without creating an
import cycle with audit_dataset.py, which orchestrates all of them.
"""

import csv
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union


@dataclass(frozen=True)
class ImageRecord:
    """One row of the raw dataset inventory.

    `path` is POSIX-style and relative to the configured raw_dir, so the
    inventory is portable across machines/OSes.
    """

    path: str
    filename: str
    extension: str
    class_directory: str
    canonical_class: str
    file_size_bytes: int
    width: Optional[int]
    height: Optional[int]
    mode: Optional[str]
    sha256: str
    is_readable: bool
    error_message: str


INVENTORY_COLUMNS: Sequence[str] = [f.name for f in fields(ImageRecord)]


def write_csv(
    rows: Iterable[Union[Dict[str, Any], "ImageRecord"]],
    columns: Sequence[str],
    output_path: Union[str, Path],
) -> None:
    """Write rows (dicts or ImageRecord instances) to a CSV with a fixed
    column order. Always writes a header, even for zero rows, so downstream
    tools see a valid, correctly-shaped (if empty) report.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            if isinstance(row, ImageRecord):
                row = {k: getattr(row, k) for k in columns}
            writer.writerow(row)


def records_to_relative_posix(base_dir: Path, path: Path) -> str:
    return path.relative_to(base_dir).as_posix()

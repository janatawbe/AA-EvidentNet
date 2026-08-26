"""Human-review workflow for cross-class exact duplicate conflicts.

The raw-data audit (src/data/audit_dataset.py) can detect that the exact
same photograph (byte-identical, same SHA-256) appears under two or more
contradictory canonical class labels. THIS MODULE NEVER DECIDES WHICH LABEL
IS CORRECT. It only:

  1. builds/updates a human-editable review manifest
     (data/audit/cross_class_duplicate_review.csv), preserving any
     resolutions a human has already entered when the audit is re-run;
  2. validates that manifest's structure (valid resolution values, valid
     canonical classes, no missing/duplicate/unknown groups);
  3. reports whether every group has been resolved, so a later stage
     (dataset splitting) can refuse to proceed while conflicts remain.

No function here may infer a label from class frequency, directory name,
filename, majority vote, a model prediction, or any other automated
heuristic. The only three valid resolution states are set by a human
editing the CSV directly.
"""

import csv
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

from src.data.detect_duplicates import DuplicateGroup
from src.data.records import ImageRecord, write_csv

RESOLUTION_UNRESOLVED = "UNRESOLVED"
RESOLUTION_KEEP_CLASS = "KEEP_CLASS"
RESOLUTION_EXCLUDE_GROUP = "EXCLUDE_GROUP"
VALID_RESOLUTIONS = {RESOLUTION_UNRESOLVED, RESOLUTION_KEEP_CLASS, RESOLUTION_EXCLUDE_GROUP}

REVIEW_COLUMNS = [
    "duplicate_group_id",
    "sha256",
    "num_files",
    "classes",
    "paths",
    "resolution",
    "resolved_class",
    "reviewer",
    "notes",
]

SAME_CLASS_REPORT_COLUMNS = [
    "duplicate_group_id",
    "sha256",
    "num_files",
    "canonical_class",
    "class_directory",
    "filenames",
    "paths",
    "note",
]

SUMMARY_COLUMNS = ["metric", "key", "value"]

SAME_CLASS_NOTE = (
    "Same-class exact duplicate - no label conflict. Not auto-deleted; "
    "consider for redundancy / split-integrity review in Task 3."
)


class ReviewValidationError(Exception):
    """Raised when a review manifest (existing or freshly merged) is
    structurally invalid: unknown resolution/class, duplicate groups, or a
    mismatch against the currently audited set of cross-class groups."""


class HumanReviewRequiredError(Exception):
    """Raised when dataset splitting is attempted while one or more
    cross-class duplicate groups are still UNRESOLVED."""


def _format_members(records: Sequence[ImageRecord]) -> str:
    return ";".join(f"{r.canonical_class}|{r.path}" for r in records)


def parse_members_field(paths_field: str) -> List[Tuple[str, str]]:
    """Parse a REVIEW_COLUMNS 'paths' field back into (canonical_class, path) pairs."""
    members = []
    for entry in (paths_field or "").split(";"):
        if not entry:
            continue
        canonical_class, _, path = entry.partition("|")
        members.append((canonical_class, path))
    return members


def build_cross_class_review_rows(cross_class_groups: Iterable[DuplicateGroup]) -> List[Dict[str, Any]]:
    """Build fresh review rows for cross-class groups. Every row starts
    UNRESOLVED with no resolved_class/reviewer/notes — resolutions are only
    ever entered by a human, never inferred here."""
    rows = []
    for group in cross_class_groups:
        rows.append(
            {
                "duplicate_group_id": group.group_id,
                "sha256": group.sha256,
                "num_files": group.size,
                "classes": ";".join(group.classes),
                "paths": _format_members(group.records),
                "resolution": RESOLUTION_UNRESOLVED,
                "resolved_class": "",
                "reviewer": "",
                "notes": "",
            }
        )
    return rows


def build_same_class_report_rows(same_class_groups: Iterable[DuplicateGroup]) -> List[Dict[str, Any]]:
    rows = []
    for group in same_class_groups:
        records = group.records
        rows.append(
            {
                "duplicate_group_id": group.group_id,
                "sha256": group.sha256,
                "num_files": group.size,
                "canonical_class": records[0].canonical_class,
                "class_directory": records[0].class_directory,
                "filenames": ";".join(r.filename for r in records),
                "paths": ";".join(r.path for r in records),
                "note": SAME_CLASS_NOTE,
            }
        )
    return rows


def load_review_csv(path: Union[str, Path]) -> List[Dict[str, str]]:
    """Load an existing review CSV, or return [] if it doesn't exist yet."""
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def merge_review_rows(
    existing_rows: Iterable[Dict[str, Any]], fresh_rows: Iterable[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Merge freshly computed group facts with any pre-existing human
    resolutions, keyed by sha256 (stable as long as raw_dir is unchanged).

    Facts derived from the dataset (num_files, classes, paths) always come
    from `fresh_rows`. A human's resolution/resolved_class/reviewer/notes
    are carried forward from `existing_rows` whenever the sha256 matches, so
    re-running the audit never discards completed review work. Groups only
    present in `existing_rows` (e.g. a group that no longer exists because
    raw_dir changed, which should not normally happen) are dropped.
    """
    existing_by_sha256 = {row["sha256"]: row for row in existing_rows}

    merged = []
    for fresh in fresh_rows:
        sha256 = fresh["sha256"]
        old = existing_by_sha256.get(sha256)
        row = dict(fresh)
        if old is not None:
            row["resolution"] = old.get("resolution") or RESOLUTION_UNRESOLVED
            row["resolved_class"] = old.get("resolved_class", "") or ""
            row["reviewer"] = old.get("reviewer", "") or ""
            row["notes"] = old.get("notes", "") or ""
        merged.append(row)

    merged.sort(key=lambda r: r["duplicate_group_id"])
    return merged


def validate_review_rows(
    rows: Iterable[Dict[str, Any]],
    canonical_classes: Iterable[str],
    expected_sha256_set: Optional[Set[str]] = None,
) -> List[str]:
    """Pure structural validation. Returns a list of human-readable error
    strings (empty list == valid). Never raises; callers decide what to do.
    """
    errors: List[str] = []
    canonical_set = set(canonical_classes)
    seen_sha256: Set[str] = set()
    seen_group_ids: Set[str] = set()

    for i, row in enumerate(rows):
        sha256 = row.get("sha256", "") or ""
        group_id = row.get("duplicate_group_id", "") or ""
        resolution = row.get("resolution", "") or ""
        resolved_class = row.get("resolved_class", "") or ""
        label = group_id or sha256 or f"row {i}"

        if sha256 and sha256 in seen_sha256:
            errors.append(f"{label}: sha256 '{sha256}' appears in more than one row")
        seen_sha256.add(sha256)

        if group_id and group_id in seen_group_ids:
            errors.append(f"{label}: duplicate_group_id '{group_id}' appears in more than one row")
        seen_group_ids.add(group_id)

        if resolution not in VALID_RESOLUTIONS:
            errors.append(
                f"{label}: unknown resolution '{resolution}' "
                f"(must be one of {sorted(VALID_RESOLUTIONS)})"
            )
            continue

        if resolution == RESOLUTION_KEEP_CLASS:
            if not resolved_class:
                errors.append(f"{label}: resolution KEEP_CLASS requires a non-empty resolved_class")
            elif resolved_class not in canonical_set:
                errors.append(
                    f"{label}: resolved_class '{resolved_class}' is not one of the 10 canonical classes"
                )
        else:
            if resolved_class:
                errors.append(
                    f"{label}: resolution '{resolution}' must not set resolved_class "
                    f"(got '{resolved_class}')"
                )

    if expected_sha256_set is not None:
        missing = expected_sha256_set - seen_sha256
        extra = seen_sha256 - expected_sha256_set
        for sha256 in sorted(missing):
            errors.append(
                f"missing review row for an audited cross-class duplicate group (sha256={sha256})"
            )
        for sha256 in sorted(extra):
            errors.append(
                f"review row for sha256={sha256} does not match any currently audited "
                "cross-class duplicate group"
            )

    return errors


def assert_valid_review_rows(
    rows: Iterable[Dict[str, Any]],
    canonical_classes: Iterable[str],
    expected_sha256_set: Optional[Set[str]] = None,
) -> None:
    errors = validate_review_rows(rows, canonical_classes, expected_sha256_set)
    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors)
        raise ReviewValidationError(f"Invalid cross-class duplicate review manifest:\n{formatted}")


def count_unresolved(rows: Iterable[Dict[str, Any]]) -> int:
    return sum(1 for r in rows if r.get("resolution") == RESOLUTION_UNRESOLVED)


def get_unresolved_group_ids(rows: Iterable[Dict[str, Any]]) -> List[str]:
    return sorted(r["duplicate_group_id"] for r in rows if r.get("resolution") == RESOLUTION_UNRESOLVED)


def write_cross_class_review_csv(rows: Iterable[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    write_csv(rows, REVIEW_COLUMNS, output_path)


def write_same_class_report_csv(rows: Iterable[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    write_csv(rows, SAME_CLASS_REPORT_COLUMNS, output_path)


def build_summary_rows(
    review_rows: Sequence[Dict[str, Any]], canonical_classes: Iterable[str]
) -> List[Dict[str, Any]]:
    total_groups = len(review_rows)
    total_files = sum(int(r["num_files"]) for r in review_rows)
    unresolved = sum(1 for r in review_rows if r.get("resolution") == RESOLUTION_UNRESOLVED)
    excluded = sum(1 for r in review_rows if r.get("resolution") == RESOLUTION_EXCLUDE_GROUP)

    rows: List[Dict[str, Any]] = [
        {"metric": "total_conflicting_groups", "key": "", "value": total_groups},
        {"metric": "total_affected_files", "key": "", "value": total_files},
        {"metric": "groups_unresolved", "key": "", "value": unresolved},
        {"metric": "groups_excluded", "key": "", "value": excluded},
    ]

    for canonical_class in sorted(canonical_classes):
        count = sum(
            1
            for r in review_rows
            if r.get("resolution") == RESOLUTION_KEEP_CLASS and r.get("resolved_class") == canonical_class
        )
        rows.append({"metric": "groups_resolved_to_class", "key": canonical_class, "value": count})

    pair_counts = Counter(r["classes"] for r in review_rows)
    for combo, count in sorted(pair_counts.items()):
        rows.append({"metric": "groups_by_class_pair", "key": combo, "value": count})

    return rows


def write_summary_csv(
    review_rows: Sequence[Dict[str, Any]],
    canonical_classes: Iterable[str],
    output_path: Union[str, Path],
) -> None:
    rows = build_summary_rows(review_rows, canonical_classes)
    write_csv(rows, SUMMARY_COLUMNS, output_path)


def assert_ready_for_split(
    review_rows: Sequence[Dict[str, Any]],
    canonical_classes: Iterable[str],
    expected_sha256_set: Optional[Set[str]] = None,
) -> None:
    """For future use by the dataset-splitting stage (Task 3+): raise
    unless every cross-class duplicate group has a human-entered resolution.

    Never resolves anything itself — it only refuses to let unresolved
    conflicts pass silently into a split.
    """
    assert_valid_review_rows(review_rows, canonical_classes, expected_sha256_set)
    unresolved = get_unresolved_group_ids(review_rows)
    if unresolved:
        raise HumanReviewRequiredError(
            f"{len(unresolved)} cross-class exact-duplicate group(s) still require human review "
            "before dataset splitting can proceed: "
            + ", ".join(unresolved)
            + ". Resolve each group's 'resolution' to KEEP_CLASS or EXCLUDE_GROUP in the review "
            "manifest — this must be a human decision and must never be inferred automatically."
        )

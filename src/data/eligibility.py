"""Dataset eligibility layer: which raw images may participate in model
development, derived from the raw-data audit and the human-reviewed
cross-class duplicate manifest.

Pipeline shape this module sits in:

    raw dataset (data/raw/, immutable)
        -> audit (src/data/audit_dataset.py: inventory, integrity, duplicates)
        -> eligibility review (src/data/duplicate_review.py: human resolves
           cross-class label conflicts in cross_class_duplicate_review.csv)
        -> ELIGIBLE DATASET (this module: data/audit/dataset_eligibility.csv)
        -> future stratified split (not yet implemented)

The eligible dataset is the POPULATION future splitting draws from — this
module never splits, balances, augments, or trains anything. Exclusion is a
data-quality decision (an unresolved or human-rejected label conflict),
never a class-balancing tool.

This module never assigns or changes a canonical class label. `canonical_class`
in every row is always the image's original raw-directory-derived label; for
a KEEP_CLASS-resolved cross-class group, the human-adjudicated label lives in
cross_class_duplicate_review.csv (`resolved_class`, joinable by
`duplicate_group_id`) and is never written back into this file or into
data/raw/.
"""

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

from src.data.detect_duplicates import DuplicateGroup
from src.data.duplicate_review import (
    RESOLUTION_EXCLUDE_GROUP,
    RESOLUTION_KEEP_CLASS,
    RESOLUTION_UNRESOLVED,
    get_unresolved_group_ids,
)
from src.data.records import ImageRecord, write_csv

ELIGIBLE_TRUE = "true"
ELIGIBLE_FALSE = "false"

DUPLICATE_TYPE_NONE = ""
DUPLICATE_TYPE_SAME_CLASS = "same_class"
DUPLICATE_TYPE_CROSS_CLASS = "cross_class"

# The one required by the scientific policy for an unresolved conflict.
EXCLUSION_UNRESOLVED_CROSS_CLASS = "unresolved_cross_class_exact_duplicate"
# A human explicitly decided (EXCLUDE_GROUP) rather than leaving it unresolved.
EXCLUSION_HUMAN_EXCLUDED_CROSS_CLASS = "human_excluded_cross_class_duplicate"
# Only reachable if duplicate_policy.same_class_exact_duplicate is set to "exclude".
EXCLUSION_SAME_CLASS_POLICY = "policy_excluded_same_class_duplicate"

VALID_EXCLUSION_REASONS = {
    "",
    EXCLUSION_UNRESOLVED_CROSS_CLASS,
    EXCLUSION_HUMAN_EXCLUDED_CROSS_CLASS,
    EXCLUSION_SAME_CLASS_POLICY,
}
VALID_DUPLICATE_TYPES = {DUPLICATE_TYPE_NONE, DUPLICATE_TYPE_SAME_CLASS, DUPLICATE_TYPE_CROSS_CLASS}

DEFAULT_DUPLICATE_POLICY = {
    "unresolved_cross_class": "exclude",
    "same_class_exact_duplicate": "keep",
    "require_review_before_split": True,
}

ELIGIBILITY_COLUMNS = [
    "path",
    "canonical_class",
    "sha256",
    "eligible",
    "exclusion_reason",
    "duplicate_group_id",
    "duplicate_type",
]

ELIGIBLE_CLASS_DISTRIBUTION_COLUMNS = [
    "canonical_class",
    "raw_count",
    "excluded_count",
    "eligible_count",
    "eligible_percentage",
]

ELIGIBILITY_SUMMARY_COLUMNS = ["metric", "value"]


class EligibilityValidationError(Exception):
    """Raised when an eligibility manifest is structurally invalid or
    inconsistent with the set of raw images it is supposed to cover."""


class SplitGuardError(Exception):
    """Raised by assert_split_is_valid() when a proposed path->split
    assignment would violate eligibility or duplicate-group integrity.
    Reusable by the future dataset-splitting stage; implements no splitting
    logic itself."""


def _row(record: ImageRecord, eligible: bool, exclusion_reason: str, group_id: str, duplicate_type: str) -> Dict[str, Any]:
    return {
        "path": record.path,
        "canonical_class": record.canonical_class,
        "sha256": record.sha256,
        "eligible": ELIGIBLE_TRUE if eligible else ELIGIBLE_FALSE,
        "exclusion_reason": exclusion_reason,
        "duplicate_group_id": group_id,
        "duplicate_type": duplicate_type,
    }


def build_eligibility_rows(
    records: Iterable[ImageRecord],
    duplicate_groups: Iterable[DuplicateGroup],
    cross_class_review_rows: Iterable[Dict[str, Any]],
    duplicate_policy: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Derive one eligibility row per raw image.

    Never hard-codes which files are excluded: exclusion is entirely
    derived from (a) which duplicate group a file's SHA-256 belongs to and
    (b) that group's current `resolution` in cross_class_review_rows.
    """
    policy = {**DEFAULT_DUPLICATE_POLICY, **(duplicate_policy or {})}

    group_by_sha256: Dict[str, DuplicateGroup] = {g.sha256: g for g in duplicate_groups}
    resolution_by_sha256 = {
        row["sha256"]: row.get("resolution", RESOLUTION_UNRESOLVED) for row in cross_class_review_rows
    }

    rows: List[Dict[str, Any]] = []
    for record in records:
        group = group_by_sha256.get(record.sha256)

        if group is None:
            rows.append(_row(record, True, "", "", DUPLICATE_TYPE_NONE))
            continue

        if group.is_cross_class:
            resolution = resolution_by_sha256.get(record.sha256, RESOLUTION_UNRESOLVED)
            if resolution == RESOLUTION_KEEP_CLASS:
                rows.append(_row(record, True, "", group.group_id, DUPLICATE_TYPE_CROSS_CLASS))
            elif resolution == RESOLUTION_EXCLUDE_GROUP:
                rows.append(
                    _row(
                        record, False, EXCLUSION_HUMAN_EXCLUDED_CROSS_CLASS,
                        group.group_id, DUPLICATE_TYPE_CROSS_CLASS,
                    )
                )
            else:  # RESOLUTION_UNRESOLVED
                if policy["unresolved_cross_class"] == "exclude":
                    rows.append(
                        _row(
                            record, False, EXCLUSION_UNRESOLVED_CROSS_CLASS,
                            group.group_id, DUPLICATE_TYPE_CROSS_CLASS,
                        )
                    )
                else:
                    rows.append(_row(record, True, "", group.group_id, DUPLICATE_TYPE_CROSS_CLASS))
        else:
            if policy["same_class_exact_duplicate"] == "keep":
                rows.append(_row(record, True, "", group.group_id, DUPLICATE_TYPE_SAME_CLASS))
            else:
                rows.append(
                    _row(
                        record, False, EXCLUSION_SAME_CLASS_POLICY,
                        group.group_id, DUPLICATE_TYPE_SAME_CLASS,
                    )
                )

    rows.sort(key=lambda r: r["path"])
    return rows


def write_eligibility_csv(rows: Iterable[Dict[str, Any]], output_path: Union[str, Path]) -> None:
    write_csv(rows, ELIGIBILITY_COLUMNS, output_path)


def load_eligibility_csv(path: Union[str, Path]):
    import csv

    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def validate_eligibility_rows(
    rows: Iterable[Dict[str, Any]],
    canonical_classes: Iterable[str],
    expected_paths: Optional[Set[str]] = None,
) -> List[str]:
    """Pure structural validation. Returns a list of error strings (empty
    list == valid); never raises."""
    errors: List[str] = []
    canonical_set = set(canonical_classes)
    seen_paths: Set[str] = set()

    for row in rows:
        path = row.get("path", "")
        eligible = row.get("eligible", "")
        exclusion_reason = row.get("exclusion_reason", "") or ""
        canonical_class = row.get("canonical_class", "")
        duplicate_type = row.get("duplicate_type", "") or ""
        duplicate_group_id = row.get("duplicate_group_id", "") or ""

        if path in seen_paths:
            errors.append(f"{path}: conflicting eligibility records (path appears more than once)")
        seen_paths.add(path)

        if canonical_class not in canonical_set:
            errors.append(f"{path}: unknown canonical_class '{canonical_class}'")

        if str(eligible) not in (ELIGIBLE_TRUE, ELIGIBLE_FALSE):
            errors.append(f"{path}: eligible must be '{ELIGIBLE_TRUE}' or '{ELIGIBLE_FALSE}', got '{eligible}'")
            continue

        if exclusion_reason not in VALID_EXCLUSION_REASONS:
            errors.append(f"{path}: unknown exclusion_reason '{exclusion_reason}'")

        if str(eligible) == ELIGIBLE_TRUE and exclusion_reason:
            errors.append(f"{path}: eligible=true but exclusion_reason is set ('{exclusion_reason}')")
        if str(eligible) == ELIGIBLE_FALSE and not exclusion_reason:
            errors.append(f"{path}: eligible=false requires a non-empty exclusion_reason")

        if duplicate_type not in VALID_DUPLICATE_TYPES:
            errors.append(f"{path}: unknown duplicate_type '{duplicate_type}'")
        if duplicate_type == DUPLICATE_TYPE_NONE and duplicate_group_id:
            errors.append(f"{path}: duplicate_type is empty but duplicate_group_id is set ('{duplicate_group_id}')")
        if duplicate_type != DUPLICATE_TYPE_NONE and not duplicate_group_id:
            errors.append(f"{path}: duplicate_type '{duplicate_type}' requires a non-empty duplicate_group_id")

    if expected_paths is not None:
        missing = expected_paths - seen_paths
        extra = seen_paths - expected_paths
        for path in sorted(missing):
            errors.append(f"raw image '{path}' has no eligibility decision")
        for path in sorted(extra):
            errors.append(f"eligibility record for '{path}' does not correspond to any known raw image")

    return errors


def assert_valid_eligibility_rows(
    rows: Iterable[Dict[str, Any]],
    canonical_classes: Iterable[str],
    expected_paths: Optional[Set[str]] = None,
) -> None:
    errors = validate_eligibility_rows(rows, canonical_classes, expected_paths)
    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors)
        raise EligibilityValidationError(f"Invalid dataset eligibility manifest:\n{formatted}")


def build_eligible_class_distribution_rows(
    rows: Sequence[Dict[str, Any]], canonical_classes: Iterable[str]
) -> List[Dict[str, Any]]:
    """One row per configured canonical class (even if raw_count is 0),
    counts always computed from the eligibility rows, never hard-coded."""
    by_class: Dict[str, Dict[str, int]] = {c: {"raw": 0, "excluded": 0} for c in canonical_classes}
    for row in rows:
        cls = row.get("canonical_class")
        if cls not in by_class:
            continue  # unknown class rows are a validation error, not a stats concern
        by_class[cls]["raw"] += 1
        if str(row.get("eligible")) == ELIGIBLE_FALSE:
            by_class[cls]["excluded"] += 1

    output = []
    for cls in sorted(by_class.keys()):
        raw_count = by_class[cls]["raw"]
        excluded_count = by_class[cls]["excluded"]
        eligible_count = raw_count - excluded_count
        pct = round(100.0 * eligible_count / raw_count, 4) if raw_count > 0 else 0.0
        output.append(
            {
                "canonical_class": cls,
                "raw_count": raw_count,
                "excluded_count": excluded_count,
                "eligible_count": eligible_count,
                "eligible_percentage": pct,
            }
        )
    return output


def write_eligible_class_distribution_csv(
    rows: Sequence[Dict[str, Any]], canonical_classes: Iterable[str], output_path: Union[str, Path]
) -> None:
    distribution_rows = build_eligible_class_distribution_rows(rows, canonical_classes)
    write_csv(distribution_rows, ELIGIBLE_CLASS_DISTRIBUTION_COLUMNS, output_path)


def build_eligibility_summary_rows(
    eligibility_rows: Sequence[Dict[str, Any]],
    same_class_groups: Sequence[DuplicateGroup],
    cross_class_review_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    total_raw = len(eligibility_rows)
    eligible = sum(1 for r in eligibility_rows if str(r.get("eligible")) == ELIGIBLE_TRUE)
    excluded = total_raw - eligible

    total_cross_class_groups = len(cross_class_review_rows)
    unresolved_groups = sum(
        1 for r in cross_class_review_rows if r.get("resolution") == RESOLUTION_UNRESOLVED
    )
    exclude_group_groups = sum(
        1 for r in cross_class_review_rows if r.get("resolution") == RESOLUTION_EXCLUDE_GROUP
    )
    keep_class_groups = sum(
        1 for r in cross_class_review_rows if r.get("resolution") == RESOLUTION_KEEP_CLASS
    )
    resolved_groups = exclude_group_groups + keep_class_groups
    excluded_cross_class_groups = total_cross_class_groups - keep_class_groups

    return [
        {"metric": "total_raw_images", "value": total_raw},
        {"metric": "eligible_images", "value": eligible},
        {"metric": "excluded_images", "value": excluded},
        {"metric": "excluded_cross_class_groups", "value": excluded_cross_class_groups},
        {"metric": "same_class_duplicate_groups", "value": len(same_class_groups)},
        {"metric": "unresolved_groups", "value": unresolved_groups},
        {"metric": "resolved_groups", "value": resolved_groups},
        {"metric": "excluded_groups", "value": exclude_group_groups},
    ]


def write_eligibility_summary_csv(
    eligibility_rows: Sequence[Dict[str, Any]],
    same_class_groups: Sequence[DuplicateGroup],
    cross_class_review_rows: Sequence[Dict[str, Any]],
    output_path: Union[str, Path],
) -> None:
    rows = build_eligibility_summary_rows(eligibility_rows, same_class_groups, cross_class_review_rows)
    write_csv(rows, ELIGIBILITY_SUMMARY_COLUMNS, output_path)


def assert_split_is_valid(
    eligibility_rows: Sequence[Dict[str, Any]],
    split_assignment: Dict[str, str],
    canonical_classes: Iterable[str],
    cross_class_review_rows: Sequence[Dict[str, Any]] = (),
    expected_paths: Optional[Set[str]] = None,
    require_review_before_split: bool = True,
) -> None:
    """Reusable guard for the future dataset-splitting stage. Never picks a
    split or a label itself — it only refuses invalid input. Call this
    BEFORE writing any split manifest.

    Args:
        eligibility_rows: rows from dataset_eligibility.csv.
        split_assignment: proposed mapping of {relative_path: split_name},
            e.g. {"Healthy/Healthy1.jpg": "train", ...}. Not required to
            cover every eligible path (this only validates what's given).
        canonical_classes: the 10 configured canonical classes.
        cross_class_review_rows: rows from cross_class_duplicate_review.csv,
            used for the require_review_before_split gate.
        expected_paths: if given, every raw image path must have an
            eligibility decision (and no unknown extra ones).
        require_review_before_split: if True, refuse to validate any split
            at all while a cross-class duplicate group remains UNRESOLVED,
            even though such groups are already excluded from eligibility.

    Raises:
        EligibilityValidationError: the eligibility manifest itself is
            structurally invalid (propagated from validate_eligibility_rows).
        SplitGuardError: the eligibility manifest is valid, but the proposed
            split assignment violates eligibility or duplicate-group rules.
    """
    assert_valid_eligibility_rows(eligibility_rows, canonical_classes, expected_paths)

    errors: List[str] = []

    if require_review_before_split:
        unresolved = get_unresolved_group_ids(cross_class_review_rows)
        if unresolved:
            errors.append(
                f"{len(unresolved)} cross-class duplicate group(s) still require human review "
                f"before any split may be created: {', '.join(unresolved)}"
            )

    eligibility_by_path = {row["path"]: row for row in eligibility_rows}

    for path, split_name in split_assignment.items():
        row = eligibility_by_path.get(path)
        if row is None:
            errors.append(f"path '{path}' in split assignment has no eligibility record")
            continue
        if str(row.get("eligible")) != ELIGIBLE_TRUE:
            errors.append(
                f"excluded path '{path}' (reason={row.get('exclusion_reason')}) "
                f"appears in the split assignment (as '{split_name}')"
            )

    group_splits: Dict[str, Set[str]] = {}
    for path, split_name in split_assignment.items():
        row = eligibility_by_path.get(path)
        if row is None:
            continue
        group_id = row.get("duplicate_group_id") or ""
        if not group_id:
            continue
        group_splits.setdefault(group_id, set()).add(split_name)
    for group_id, splits in group_splits.items():
        if len(splits) > 1:
            errors.append(
                f"duplicate group '{group_id}' is split across multiple splits: {sorted(splits)}"
            )

    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors)
        raise SplitGuardError(f"Proposed split assignment failed validation:\n{formatted}")

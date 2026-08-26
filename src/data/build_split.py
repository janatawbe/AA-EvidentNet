"""Deterministic, leakage-safe 70/20/10 stratified split of the ELIGIBLE
original images (see src/data/eligibility.py) into train/val/test.

Pipeline position:

    raw dataset -> audit -> eligibility review -> ELIGIBLE DATASET
        -> THIS MODULE: original split -> future augmentation/balancing

This module never augments, balances, or trains anything, and never
modifies data/raw/. It only reads dataset_eligibility.csv (and
cross_class_duplicate_review.csv, for the rare case of an eligible
KEEP_CLASS-resolved cross-class group) and writes the three original
manifests plus provenance/validation artifacts under data/manifests/ and
data/audit/.

Design notes worth reading before changing this file:

- Exact-duplicate groups (same-class, or an eligible cross-class group
  once a human resolves it KEEP_CLASS) are treated as a single atomic
  "split unit" — every member always lands in the same split, never
  divided, regardless of how the 70/20/10 targets round.
- A unit's *stratification/training class* is its members' shared
  canonical_class for an ordinary image or same-class group. For an
  eligible cross-class group (KEEP_CLASS resolved), it is the human's
  `resolved_class` from cross_class_duplicate_review.csv — this is the
  ONLY place that adjudicated label is used, and it is written only into
  the split manifest's `class` column, never back into
  dataset_eligibility.csv or data/raw/.
- `original_id` is computed from each file's own immutable
  canonical_class (never the adjudicated resolved_class), so an ID never
  changes if a group's resolution changes later. See compute_original_id().
- Split allocation is a deterministic largest-remainder-plus-greedy-deficit
  algorithm seeded via src.utils.seeding.set_seed(); no uncontrolled
  randomness is used anywhere.
- `duplicate_policy.require_review_before_split` does NOT block this
  module from producing a split while cross-class conflicts remain
  UNRESOLVED — those images are already excluded via eligibility, which is
  the actual leakage-prevention mechanism. Honoring the flag by refusing
  to run at all would make the audit+eligibility pipeline pointless to
  execute end-to-end. Instead, this module prints and records a prominent
  "provisional split" notice whenever unresolved groups exist, so nobody
  mistakes this build for a final, fully-reviewed dataset.
"""

import json
import platform
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from src.data import duplicate_review as review
from src.data import eligibility as elig
from src.data.records import write_csv
from src.utils.config import hash_config, load_config
from src.utils.env_info import collect_environment_info
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file, hash_string
from src.utils.seeding import DEFAULT_SEED, set_seed

SPLIT_NAMES: Tuple[str, str, str] = ("train", "val", "test")

IS_ORIGINAL_TRUE = "true"
AUGMENTATION_TYPE_ORIGINAL = "original"

MANIFEST_COLUMNS = [
    "path",
    "class",
    "split",
    "original_id",
    "parent_original_id",
    "is_original",
    "augmentation_type",
]

SPLIT_DISTRIBUTION_COLUMNS = ["split", "class", "count", "percentage_of_class", "percentage_of_split"]
SPLIT_LEAKAGE_COLUMNS = ["check", "status", "details"]


class SplitBuildError(Exception):
    """Fatal, unrecoverable problem building the split (missing audit
    artifacts, a raw file missing/changed since the audit, an eligible
    duplicate group with no valid class resolution, etc.)."""


class SplitValidationError(Exception):
    """The split was built but failed one or more mandatory integrity
    checks (see validate_split_manifests). No manifest is trusted/kept
    valid when this is raised — check split_leakage_report.csv."""


def compute_original_id(canonical_class: str, relative_path: str, sha256_hash: str) -> str:
    """Deterministic, machine/path-independent image ID.

    sha256("<canonical_class>|<relative_path>|<sha256_hash>") — depends
    only on stable, dataset-relative facts (never an absolute path,
    timestamp, or random UUID), so repeated runs on the same dataset
    produce identical IDs. Always uses the file's own immutable
    canonical_class (raw-directory-derived), never a later human-adjudicated
    resolved_class, so an ID never changes if a duplicate group's
    resolution changes.
    """
    return hash_string(f"{canonical_class}|{relative_path}|{sha256_hash}")


@dataclass(frozen=True)
class SplitUnit:
    """One atomic thing the split allocator moves as a whole: either a
    single non-duplicated image, or an entire exact-duplicate group."""

    unit_id: str
    stratify_class: str
    members: Tuple[Dict[str, Any], ...]


@dataclass(frozen=True)
class CheckResult:
    check: str
    status: str  # "PASS" or "FAIL"
    details: str


def verify_files_against_eligibility(
    eligible_rows: Sequence[Dict[str, Any]], raw_dir: Path
) -> List[str]:
    """Re-verify every eligible row's file still exists under raw_dir and
    still hashes to the SHA-256 recorded during the audit. Read-only.

    Returns a list of human-readable error strings (empty == all good).
    """
    errors: List[str] = []
    for row in eligible_rows:
        file_path = raw_dir / row["path"]
        if not file_path.is_file():
            errors.append(f"{row['path']}: file not found under raw_dir")
            continue
        actual_hash = hash_file(file_path)
        if actual_hash != row["sha256"]:
            errors.append(
                f"{row['path']}: SHA-256 mismatch (audit recorded {row['sha256']}, "
                f"actual is {actual_hash}) - raw_dir may have changed since the audit ran"
            )
    return errors


def build_split_units(
    eligible_rows: Iterable[Dict[str, Any]],
    cross_class_review_rows: Iterable[Dict[str, Any]],
) -> List[SplitUnit]:
    """Group eligible rows into atomic split units.

    Raises SplitBuildError if an eligible duplicate group spans more than
    one canonical_class without a valid KEEP_CLASS resolution recording an
    authoritative class — this should be structurally impossible given the
    eligibility layer, and is treated as a hard stop rather than a guess.
    """
    review_by_group = {r["duplicate_group_id"]: r for r in cross_class_review_rows}

    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    singles: List[Dict[str, Any]] = []
    for row in eligible_rows:
        group_id = row.get("duplicate_group_id") or ""
        if group_id:
            grouped[group_id].append(row)
        else:
            singles.append(row)

    units: List[SplitUnit] = [
        SplitUnit(unit_id=f"SINGLE::{row['path']}", stratify_class=row["canonical_class"], members=(row,))
        for row in singles
    ]

    for group_id, members in grouped.items():
        classes = {m["canonical_class"] for m in members}
        if len(classes) == 1:
            stratify_class = next(iter(classes))
        else:
            review_row = review_by_group.get(group_id)
            resolved_class = (review_row or {}).get("resolved_class", "")
            resolution = (review_row or {}).get("resolution", "")
            if resolution != review.RESOLUTION_KEEP_CLASS or not resolved_class:
                raise SplitBuildError(
                    f"Eligible duplicate group '{group_id}' spans multiple canonical classes "
                    f"({sorted(classes)}) but has no valid KEEP_CLASS resolution recording an "
                    "authoritative class. Refusing to guess a label - this indicates a bug in "
                    "the eligibility layer, since such a group should only be eligible when "
                    "KEEP_CLASS has been set."
                )
            stratify_class = resolved_class
        units.append(SplitUnit(unit_id=group_id, stratify_class=stratify_class, members=tuple(members)))

    units.sort(key=lambda u: u.unit_id)
    return units


def compute_split_targets(total: int, ratios: Sequence[float]) -> Dict[str, int]:
    """Largest-remainder rounding of `total` into SPLIT_NAMES buckets by
    `ratios`, guaranteeing the targets sum exactly to `total` while
    minimizing deviation from the requested ratios."""
    raw = [total * r for r in ratios]
    floors = [int(x) for x in raw]
    remainder = total - sum(floors)
    fractional_order = sorted(range(len(SPLIT_NAMES)), key=lambda i: raw[i] - floors[i], reverse=True)
    for i in range(remainder):
        floors[fractional_order[i % len(SPLIT_NAMES)]] += 1
    return dict(zip(SPLIT_NAMES, floors))


def allocate_splits(
    units: Sequence[SplitUnit], ratios: Sequence[float] = (0.70, 0.20, 0.10)
) -> Tuple[Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Deterministically assign every unit to a split, stratified by
    stratify_class, keeping every unit (and hence every duplicate group)
    atomic.

    Relies on the caller having already called src.utils.seeding.set_seed()
    — this function advances the global `random` module's state via
    random.shuffle(), processing classes in sorted order, so the entire
    sequence of random decisions is fully determined by that seed.

    Per class: units are shuffled, then greedily assigned to whichever
    split currently has the largest remaining deficit against its
    (largest-remainder-rounded) target — this keeps variable-size units
    (duplicate groups are 2-3 files; everything else is 1) as close to the
    requested ratios as integer/atomicity constraints allow, without ever
    splitting a unit.

    Returns (assignment: unit_id -> split_name, per_class_targets, per_class_counts).
    """
    by_class: Dict[str, List[SplitUnit]] = defaultdict(list)
    for unit in units:
        by_class[unit.stratify_class].append(unit)

    assignment: Dict[str, str] = {}
    per_class_targets: Dict[str, Dict[str, int]] = {}
    per_class_counts: Dict[str, Dict[str, int]] = {}

    for canonical_class in sorted(by_class.keys()):
        class_units = list(by_class[canonical_class])  # already unit_id-sorted from build_split_units
        random.shuffle(class_units)

        total_members = sum(len(u.members) for u in class_units)
        targets = compute_split_targets(total_members, ratios)
        counts = {s: 0 for s in SPLIT_NAMES}

        for unit in class_units:
            deficits = {s: targets[s] - counts[s] for s in SPLIT_NAMES}
            best_split = max(SPLIT_NAMES, key=lambda s: (deficits[s], -SPLIT_NAMES.index(s)))
            assignment[unit.unit_id] = best_split
            counts[best_split] += len(unit.members)

        per_class_targets[canonical_class] = targets
        per_class_counts[canonical_class] = counts

    return assignment, per_class_targets, per_class_counts


def build_manifest_rows(
    units: Sequence[SplitUnit], assignment: Dict[str, str]
) -> Dict[str, List[Dict[str, Any]]]:
    manifests: Dict[str, List[Dict[str, Any]]] = {s: [] for s in SPLIT_NAMES}
    for unit in units:
        split_name = assignment[unit.unit_id]
        for member in unit.members:
            original_id = compute_original_id(member["canonical_class"], member["path"], member["sha256"])
            manifests[split_name].append(
                {
                    "path": member["path"],
                    "class": unit.stratify_class,
                    "split": split_name,
                    "original_id": original_id,
                    "parent_original_id": original_id,
                    "is_original": IS_ORIGINAL_TRUE,
                    "augmentation_type": AUGMENTATION_TYPE_ORIGINAL,
                }
            )

    for split_name in SPLIT_NAMES:
        manifests[split_name].sort(key=lambda r: (r["class"], r["path"]))
    return manifests


def _pairwise_overlap(name: str, a: set, b: set) -> CheckResult:
    overlap = a & b
    status = "FAIL" if overlap else "PASS"
    details = f"{len(overlap)} overlapping path(s)" if overlap else "no overlap"
    return CheckResult(name, status, details)


def validate_split_manifests(
    manifests: Dict[str, List[Dict[str, Any]]],
    eligible_rows: Sequence[Dict[str, Any]],
    all_eligibility_rows: Sequence[Dict[str, Any]],
    cross_class_review_rows: Sequence[Dict[str, Any]],
    canonical_classes: Iterable[str],
    raw_dir: Optional[Path] = None,
) -> List[CheckResult]:
    """Run every mandatory split-integrity check. Reusable by later tasks
    (e.g. to validate a hand-edited or historical manifest set) — reads
    only what is passed in; never touches data/raw/ unless raw_dir is given.

    Checks implemented (see module docstring / README for the full list):
      1-3. train/val/test pairwise path overlap
      4. cross-split SHA-256 overlap (same content under different paths)
      5. duplicate-group split integrity (every group in exactly one split)
      6. excluded-file leakage (an eligible=false path in a manifest)
      7. unresolved-group leakage (an UNRESOLVED cross-class path in a manifest)
      8. augmentation leakage (is_original/augmentation_type/parent_original_id)
      9. class-label consistency (manifest `class` matches the authoritative source)
      10. intra-manifest duplicate rows (same path twice in one split file)
      11. file existence + SHA-256 re-verification against raw_dir (if given)
      12. structural completeness (every eligible row appears exactly once, total)
    """
    results: List[CheckResult] = []

    eligibility_by_path = {r["path"]: r for r in all_eligibility_rows}
    review_by_group = {r["duplicate_group_id"]: r for r in cross_class_review_rows}

    train_paths = {r["path"] for r in manifests["train"]}
    val_paths = {r["path"] for r in manifests["val"]}
    test_paths = {r["path"] for r in manifests["test"]}

    results.append(_pairwise_overlap("train_val_overlap", train_paths, val_paths))
    results.append(_pairwise_overlap("train_test_overlap", train_paths, test_paths))
    results.append(_pairwise_overlap("val_test_overlap", val_paths, test_paths))

    # Cross-split SHA-256 overlap (defense in depth beyond path overlap).
    sha_to_splits: Dict[str, set] = defaultdict(set)
    for split_name, rows in manifests.items():
        for row in rows:
            elig_row = eligibility_by_path.get(row["path"])
            if elig_row is not None:
                sha_to_splits[elig_row["sha256"]].add(split_name)
    sha_violations = {h: s for h, s in sha_to_splits.items() if len(s) > 1}
    results.append(
        CheckResult(
            "cross_split_sha256_overlap",
            "FAIL" if sha_violations else "PASS",
            f"{len(sha_violations)} sha256 value(s) span multiple splits" if sha_violations else "no overlap",
        )
    )

    # Duplicate-group split integrity.
    path_to_group = {r["path"]: (r.get("duplicate_group_id") or "") for r in all_eligibility_rows}
    group_splits: Dict[str, set] = defaultdict(set)
    for split_name, rows in manifests.items():
        for row in rows:
            group_id = path_to_group.get(row["path"], "")
            if group_id:
                group_splits[group_id].add(split_name)
    group_violations = {g: s for g, s in group_splits.items() if len(s) > 1}
    results.append(
        CheckResult(
            "duplicate_group_split_integrity",
            "FAIL" if group_violations else "PASS",
            f"{len(group_violations)} duplicate group(s) span multiple splits: {list(group_violations)[:5]}"
            if group_violations
            else "every duplicate group is confined to exactly one split",
        )
    )

    # Excluded-file leakage.
    excluded_leaks = [
        row["path"]
        for rows in manifests.values()
        for row in rows
        if eligibility_by_path.get(row["path"], {}).get("eligible") != elig.ELIGIBLE_TRUE
    ]
    results.append(
        CheckResult(
            "excluded_file_leakage",
            "FAIL" if excluded_leaks else "PASS",
            f"{len(excluded_leaks)} excluded path(s) found in a split" if excluded_leaks else "none found",
        )
    )

    # Unresolved cross-class-duplicate leakage.
    unresolved_leaks = []
    for rows in manifests.values():
        for row in rows:
            elig_row = eligibility_by_path.get(row["path"])
            if elig_row and elig_row.get("duplicate_type") == elig.DUPLICATE_TYPE_CROSS_CLASS:
                group_row = review_by_group.get(elig_row.get("duplicate_group_id"), {})
                if group_row.get("resolution", review.RESOLUTION_UNRESOLVED) == review.RESOLUTION_UNRESOLVED:
                    unresolved_leaks.append(row["path"])
    results.append(
        CheckResult(
            "unresolved_cross_class_leakage",
            "FAIL" if unresolved_leaks else "PASS",
            f"{len(unresolved_leaks)} unresolved cross-class path(s) found in a split"
            if unresolved_leaks
            else "none found",
        )
    )

    # Augmentation leakage.
    aug_leaks = [
        row["path"]
        for rows in manifests.values()
        for row in rows
        if row.get("is_original") != IS_ORIGINAL_TRUE
        or row.get("augmentation_type") != AUGMENTATION_TYPE_ORIGINAL
        or row.get("parent_original_id") != row.get("original_id")
    ]
    results.append(
        CheckResult(
            "augmentation_leakage",
            "FAIL" if aug_leaks else "PASS",
            f"{len(aug_leaks)} non-original row(s) found in an original manifest" if aug_leaks else "none found",
        )
    )

    # Class-label consistency.
    label_mismatches = []
    for rows in manifests.values():
        for row in rows:
            elig_row = eligibility_by_path.get(row["path"])
            if elig_row is None:
                continue
            if elig_row.get("duplicate_type") == elig.DUPLICATE_TYPE_CROSS_CLASS:
                expected = review_by_group.get(elig_row.get("duplicate_group_id"), {}).get("resolved_class", "")
            else:
                expected = elig_row.get("canonical_class")
            if row["class"] != expected:
                label_mismatches.append(row["path"])
    results.append(
        CheckResult(
            "class_label_consistency",
            "FAIL" if label_mismatches else "PASS",
            f"{len(label_mismatches)} row(s) with a class not matching their eligibility/review record"
            if label_mismatches
            else "all manifest classes match their authoritative source",
        )
    )

    # Intra-manifest duplicate rows.
    intra_dupes = {}
    for split_name, rows in manifests.items():
        counts = Counter(r["path"] for r in rows)
        dupes = [p for p, c in counts.items() if c > 1]
        if dupes:
            intra_dupes[split_name] = dupes
    results.append(
        CheckResult(
            "intra_manifest_duplicate_rows",
            "FAIL" if intra_dupes else "PASS",
            f"duplicate rows within a single manifest: {intra_dupes}" if intra_dupes else "none found",
        )
    )

    # File existence + SHA-256 re-verification.
    if raw_dir is not None:
        file_errors = verify_files_against_eligibility(eligible_rows, raw_dir)
        results.append(
            CheckResult(
                "raw_file_existence_and_hash",
                "FAIL" if file_errors else "PASS",
                "; ".join(file_errors[:5]) if file_errors else "all eligible files exist and match their recorded SHA-256",
            )
        )
    else:
        results.append(CheckResult("raw_file_existence_and_hash", "PASS", "skipped (no raw_dir given)"))

    # Structural completeness: every eligible row accounted for, exactly once, total.
    total_manifest_rows = sum(len(rows) for rows in manifests.values())
    total_eligible = len(eligible_rows)
    results.append(
        CheckResult(
            "structural_completeness",
            "PASS" if total_manifest_rows == total_eligible else "FAIL",
            f"{total_manifest_rows} manifest rows vs {total_eligible} eligible rows",
        )
    )

    return results


def write_split_leakage_report_csv(check_results: Sequence[CheckResult], output_path: Union[str, Path]) -> None:
    rows = [{"check": c.check, "status": c.status, "details": c.details} for c in check_results]
    write_csv(rows, SPLIT_LEAKAGE_COLUMNS, output_path)


def build_split_distribution_rows(
    manifests: Dict[str, List[Dict[str, Any]]], canonical_classes: Iterable[str]
) -> List[Dict[str, Any]]:
    all_rows = [row for rows in manifests.values() for row in rows]
    class_totals = Counter(row["class"] for row in all_rows)
    split_totals = {s: len(manifests[s]) for s in SPLIT_NAMES}

    all_classes = sorted(set(canonical_classes) | set(class_totals.keys()))
    rows = []
    for canonical_class in all_classes:
        for split_name in SPLIT_NAMES:
            count = sum(1 for r in manifests[split_name] if r["class"] == canonical_class)
            class_total = class_totals.get(canonical_class, 0)
            split_total = split_totals[split_name]
            rows.append(
                {
                    "split": split_name,
                    "class": canonical_class,
                    "count": count,
                    "percentage_of_class": round(100.0 * count / class_total, 4) if class_total else 0.0,
                    "percentage_of_split": round(100.0 * count / split_total, 4) if split_total else 0.0,
                }
            )
    return rows


@dataclass
class SplitSummary:
    seed: int
    ratios: Tuple[float, float, float]
    manifests: Dict[str, List[Dict[str, Any]]]
    manifest_paths: Dict[str, Path]
    manifest_hashes: Dict[str, str]
    per_class_targets: Dict[str, Dict[str, int]]
    per_class_counts: Dict[str, Dict[str, int]]
    check_results: List[CheckResult]
    unresolved_group_count: int
    eligibility_hash: str
    config_hash: str
    git_commit: Optional[str]
    metadata_path: Path
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_eligible(self) -> int:
        return sum(len(rows) for rows in self.manifests.values())

    def format_report(self) -> str:
        lines = [
            "=" * 70,
            "DATASET SPLIT SUMMARY (original images only)",
            "=" * 70,
            f"Seed: {self.seed}",
            f"Ratios (train/val/test): {self.ratios}",
            f"Git commit: {self.git_commit}",
            "",
        ]
        total = self.total_eligible
        for split_name in SPLIT_NAMES:
            count = len(self.manifests[split_name])
            pct = round(100.0 * count / total, 2) if total else 0.0
            lines.append(f"{split_name}: {count} ({pct}%) -> {self.manifest_paths[split_name]}")

        lines.append("")
        lines.append("Per-class counts (train / val / test):")
        for canonical_class in sorted(self.per_class_counts.keys()):
            counts = self.per_class_counts[canonical_class]
            lines.append(
                f"  {canonical_class}: {counts['train']} / {counts['val']} / {counts['test']} "
                f"(total {sum(counts.values())})"
            )

        lines.append("")
        lines.append("Split integrity checks:")
        for check in self.check_results:
            lines.append(f"  [{check.status}] {check.check}: {check.details}")

        lines.append("")
        for split_name in SPLIT_NAMES:
            lines.append(f"{split_name}_original.csv sha256: {self.manifest_hashes[split_name]}")

        if self.unresolved_group_count:
            lines.append("")
            lines.append(
                f"NOTE: {self.unresolved_group_count} cross-class duplicate group(s) remain UNRESOLVED "
                "and were excluded from this split. This split is PROVISIONAL until human review of "
                "those groups is complete (see data/audit/cross_class_duplicate_review.csv)."
            )

        lines.append("=" * 70)
        return "\n".join(lines)


def run_build_split(
    config_path: Union[str, Path] = "configs/dataset.yaml",
    seed: int = DEFAULT_SEED,
    raw_dir_override: Optional[Union[str, Path]] = None,
    audit_dir_override: Optional[Union[str, Path]] = None,
    manifests_dir_override: Optional[Union[str, Path]] = None,
) -> SplitSummary:
    """Build, validate, and write the deterministic 70/20/10 original-image
    split. Read-only with respect to data/raw/.

    Raises:
        SplitBuildError: a fatal, unrecoverable problem (missing audit
            artifacts, a raw file missing/changed since the audit, an
            eligible duplicate group with no valid class resolution).
        SplitValidationError: the split was built but failed a mandatory
            integrity check (see split_leakage_report.csv for details).
    """
    set_seed(seed)

    config = load_config(config_path)
    class_directory_mapping = config["class_directory_mapping"]
    canonical_classes = sorted(class_directory_mapping.keys())

    raw_dir = Path(raw_dir_override) if raw_dir_override is not None else Path(config["paths"]["raw_dir"])
    audit_dir = Path(audit_dir_override) if audit_dir_override is not None else Path(config["paths"]["audit_dir"])
    manifests_dir = (
        Path(manifests_dir_override) if manifests_dir_override is not None else Path(config["paths"]["manifests_dir"])
    )

    eligibility_csv_path = audit_dir / "dataset_eligibility.csv"
    review_csv_path = audit_dir / "cross_class_duplicate_review.csv"

    if not eligibility_csv_path.is_file():
        raise SplitBuildError(
            f"{eligibility_csv_path} not found. Run `python run_pipeline.py audit` first to "
            "generate the eligibility manifest before preparing the split."
        )

    all_eligibility_rows = elig.load_eligibility_csv(eligibility_csv_path)
    elig.assert_valid_eligibility_rows(all_eligibility_rows, canonical_classes)

    cross_class_review_rows = review.load_review_csv(review_csv_path)
    if cross_class_review_rows:
        review.assert_valid_review_rows(cross_class_review_rows, canonical_classes)

    duplicate_policy = {**elig.DEFAULT_DUPLICATE_POLICY, **(config.get("duplicate_policy", {}) or {})}
    require_review_before_split = bool(duplicate_policy.get("require_review_before_split", True))
    unresolved_group_ids = review.get_unresolved_group_ids(cross_class_review_rows)

    eligible_rows = [r for r in all_eligibility_rows if r.get("eligible") == elig.ELIGIBLE_TRUE]
    if not eligible_rows:
        raise SplitBuildError("No eligible images found in the eligibility manifest - nothing to split.")

    file_errors = verify_files_against_eligibility(eligible_rows, raw_dir)
    if file_errors:
        raise SplitBuildError(
            "Eligible files failed re-verification against data/raw/ (missing file or SHA-256 "
            "mismatch since the audit was run):\n" + "\n".join(f"  - {e}" for e in file_errors)
        )

    units = build_split_units(eligible_rows, cross_class_review_rows)

    split_cfg = config.get("split", {}) or {}
    ratios = (
        float(split_cfg.get("train_fraction", 0.70)),
        float(split_cfg.get("val_fraction", 0.20)),
        float(split_cfg.get("test_fraction", 0.10)),
    )

    assignment, per_class_targets, per_class_counts = allocate_splits(units, ratios)
    manifests = build_manifest_rows(units, assignment)

    check_results = validate_split_manifests(
        manifests, eligible_rows, all_eligibility_rows, cross_class_review_rows, canonical_classes, raw_dir=raw_dir
    )
    failures = [c for c in check_results if c.status == "FAIL"]
    if failures:
        raise SplitValidationError(
            "Split failed mandatory integrity checks:\n"
            + "\n".join(f"  - {c.check}: {c.details}" for c in failures)
        )

    manifests_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)

    manifest_paths: Dict[str, Path] = {}
    manifest_hashes: Dict[str, str] = {}
    for split_name in SPLIT_NAMES:
        output_path = manifests_dir / f"{split_name}_original.csv"
        write_csv(manifests[split_name], MANIFEST_COLUMNS, output_path)
        manifest_paths[split_name] = output_path
        manifest_hashes[split_name] = hash_file(output_path)
        (audit_dir / f"{split_name}_manifest_hash.txt").write_text(
            manifest_hashes[split_name] + "\n", encoding="utf-8"
        )

    write_split_leakage_report_csv(check_results, audit_dir / "split_leakage_report.csv")

    distribution_rows = build_split_distribution_rows(manifests, canonical_classes)
    write_csv(distribution_rows, SPLIT_DISTRIBUTION_COLUMNS, audit_dir / "split_distribution.csv")

    eligibility_hash = hash_file(eligibility_csv_path)
    config_hash = hash_config(config)
    git_commit = get_git_commit()

    metadata = {
        "seed": seed,
        "split_ratios": {"train": ratios[0], "val": ratios[1], "test": ratios[2]},
        "eligibility_manifest_hash": eligibility_hash,
        "config_hash": config_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit,
        "environment": collect_environment_info(),
        "manifest_hashes": manifest_hashes,
        "total_eligible": len(eligible_rows),
        "counts": {s: len(manifests[s]) for s in SPLIT_NAMES},
        "duplicate_policy": duplicate_policy,
        "unresolved_cross_class_groups": len(unresolved_group_ids),
    }
    metadata_path = audit_dir / "split_metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    summary = SplitSummary(
        seed=seed,
        ratios=ratios,
        manifests=manifests,
        manifest_paths=manifest_paths,
        manifest_hashes=manifest_hashes,
        per_class_targets=per_class_targets,
        per_class_counts=per_class_counts,
        check_results=check_results,
        unresolved_group_count=len(unresolved_group_ids),
        eligibility_hash=eligibility_hash,
        config_hash=config_hash,
        git_commit=git_commit,
        metadata_path=metadata_path,
        metadata=metadata,
    )

    print(summary.format_report())
    if require_review_before_split and unresolved_group_ids:
        print(
            f"[build_split] NOTE: {len(unresolved_group_ids)} cross-class duplicate group(s) remain "
            f"UNRESOLVED and were excluded from this split (see {review_csv_path}). This split is "
            "PROVISIONAL until human review is complete.",
            file=sys.stderr,
        )

    return summary

"""Raw dataset inventory, integrity checking, and audit orchestration.

This module is strictly READ-ONLY with respect to the raw dataset: it only
opens files to read bytes/pixels for hashing and decode-checking. It never
renames, moves, deletes, resizes, augments, or otherwise writes into
raw_dir. All generated reports are written under the configured audit_dir.

Reuses (does not duplicate):
  - src.utils.config.load_config for reading configs/dataset.yaml
  - src.utils.hashing.hash_file for SHA-256 content hashing
  - configs/dataset.yaml: class_directory_mapping as the single source of
    truth for canonical-class <-> raw-directory-name resolution
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image

from src.data import dataset_statistics as stats
from src.data import detect_augmented_families as aug
from src.data import detect_duplicates as dup
from src.data import duplicate_review as review
from src.data.records import INVENTORY_COLUMNS, ImageRecord, write_csv
from src.utils.config import load_config
from src.utils.hashing import hash_file

CORRUPTED_IMAGE_COLUMNS = [
    "path",
    "filename",
    "class_directory",
    "canonical_class",
    "file_size_bytes",
    "sha256",
    "error_message",
]

LEAKAGE_REPORT_COLUMNS = [
    "stage",
    "issue_type",
    "severity",
    "canonical_class",
    "class_directory",
    "path",
    "filename",
    "details",
]

# This audit only covers raw-data integrity, before any train/val/test split
# exists. Every row written to the leakage report carries this constant so
# it is never confused with future split-leakage checks (a separate task).
RAW_DATA_AUDIT_STAGE = "raw_data_audit"

DEFAULT_SUPPORTED_EXTENSIONS = [".jpg", ".jpeg"]
DEFAULT_POLICIES = {
    "missing_class_directory": "fail",
    "unexpected_directory": "fail",
    "corrupted_image": "fail",
    "cross_class_duplicate": "fail",
    "same_class_duplicate": "warn",
    "highly_suspicious_augmentation_family": "warn",
}


class AuditConfigError(Exception):
    """Raised when the dataset config/raw directory is fatally invalid and
    no meaningful audit can be run at all (e.g. raw_dir missing, mapping
    malformed)."""


class AuditFailedError(Exception):
    """Raised when the audit ran to completion, wrote all reports, but
    found one or more issues whose configured policy is "fail"."""


@dataclass(frozen=True)
class DirectoryValidation:
    """Result of comparing configured class_directory_mapping against what
    actually exists on disk under raw_dir."""

    missing_mapped_directories: Dict[str, str]  # canonical_class -> expected dir name
    unexpected_directories: List[str]  # dir names on disk not in the mapping
    valid_mapped_directories: Dict[str, str]  # canonical_class -> dir name, present

    @property
    def ok(self) -> bool:
        return not self.missing_mapped_directories and not self.unexpected_directories


def validate_dataset_config(config: Dict[str, Any]) -> None:
    """Sanity-check configs/dataset.yaml's class schema before doing anything
    else. Raises AuditConfigError (fatal) on any structural inconsistency.
    """
    class_names = config.get("class_names")
    mapping = config.get("class_directory_mapping")
    num_classes = config.get("num_classes")

    if not class_names:
        raise AuditConfigError("configs/dataset.yaml is missing 'class_names'")
    if not mapping:
        raise AuditConfigError("configs/dataset.yaml is missing 'class_directory_mapping'")
    if len(class_names) != len(set(class_names)):
        raise AuditConfigError("configs/dataset.yaml 'class_names' contains duplicates")
    if set(class_names) != set(mapping.keys()):
        missing = set(class_names) - set(mapping.keys())
        extra = set(mapping.keys()) - set(class_names)
        raise AuditConfigError(
            "class_directory_mapping keys must exactly match class_names "
            f"(missing from mapping: {sorted(missing)}, "
            f"unexpected in mapping: {sorted(extra)})"
        )
    if num_classes != len(class_names):
        raise AuditConfigError(
            f"num_classes ({num_classes}) does not match len(class_names) ({len(class_names)})"
        )


def validate_class_directories(
    raw_dir: Path, class_directory_mapping: Dict[str, str]
) -> DirectoryValidation:
    """Compare the exact class_directory_mapping against raw_dir's actual
    subdirectories. Uses exact-name lookups only, never substring matching.
    """
    if not raw_dir.is_dir():
        raise AuditConfigError(f"raw_dir does not exist or is not a directory: {raw_dir}")

    actual_dirs = {p.name for p in raw_dir.iterdir() if p.is_dir()}
    mapped_dirs = set(class_directory_mapping.values())

    missing = {
        canonical: dirname
        for canonical, dirname in class_directory_mapping.items()
        if dirname not in actual_dirs
    }
    unexpected = sorted(actual_dirs - mapped_dirs)
    valid = {
        canonical: dirname
        for canonical, dirname in class_directory_mapping.items()
        if dirname in actual_dirs
    }

    return DirectoryValidation(
        missing_mapped_directories=missing,
        unexpected_directories=unexpected,
        valid_mapped_directories=valid,
    )


def _check_image(path: Path):
    """Verify-then-load an image without ever writing to it.

    Pillow's verify() catches many structural/truncation errors but leaves
    the file handle unusable afterwards, so a second open+load is needed to
    both confirm full decodability and read width/height/mode.
    """
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img2:
            img2.load()
            width, height = img2.size
            mode = img2.mode
        return True, width, height, mode, ""
    except Exception as e:  # noqa: BLE001 - any decode failure means "unreadable"
        return False, None, None, None, f"{type(e).__name__}: {e}"


def build_inventory(
    raw_dir: Path,
    class_directory_mapping: Dict[str, str],
    supported_extensions: Sequence[str] = DEFAULT_SUPPORTED_EXTENSIONS,
) -> List[ImageRecord]:
    """Recursively (one level deep, per class) inventory raw_dir.

    Deterministic: canonical classes are visited in sorted order, and files
    within each class directory are visited in sorted (case-insensitive)
    filename order. Only directories present in class_directory_mapping AND
    present on disk are scanned; missing directories are reported by
    validate_class_directories, not silently skipped without record.
    """
    extensions = {ext.lower() for ext in supported_extensions}
    records: List[ImageRecord] = []

    for canonical_class in sorted(class_directory_mapping.keys()):
        directory_name = class_directory_mapping[canonical_class]
        class_dir = raw_dir / directory_name
        if not class_dir.is_dir():
            continue

        file_paths = sorted(
            (p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions),
            key=lambda p: p.name.lower(),
        )

        for file_path in file_paths:
            is_readable, width, height, mode, error_message = _check_image(file_path)
            sha256 = hash_file(file_path)
            records.append(
                ImageRecord(
                    path=file_path.relative_to(raw_dir).as_posix(),
                    filename=file_path.name,
                    extension=file_path.suffix.lower(),
                    class_directory=directory_name,
                    canonical_class=canonical_class,
                    file_size_bytes=file_path.stat().st_size,
                    width=width,
                    height=height,
                    mode=mode,
                    sha256=sha256,
                    is_readable=is_readable,
                    error_message=error_message,
                )
            )

    return records


def write_corrupted_images_csv(records: Sequence[ImageRecord], output_path: Union[str, Path]) -> None:
    """Write only unreadable records. Always writes a valid header, even
    when the dataset has zero corrupted images."""
    rows = [
        {
            "path": r.path,
            "filename": r.filename,
            "class_directory": r.class_directory,
            "canonical_class": r.canonical_class,
            "file_size_bytes": r.file_size_bytes,
            "sha256": r.sha256,
            "error_message": r.error_message,
        }
        for r in records
        if not r.is_readable
    ]
    write_csv(rows, CORRUPTED_IMAGE_COLUMNS, output_path)


def _severity_for(policy: str) -> str:
    return "critical" if policy == "fail" else "warning"


def _build_leakage_report(
    directory_validation: DirectoryValidation,
    records: Sequence[ImageRecord],
    duplicate_groups: Sequence[dup.DuplicateGroup],
    augmentation_findings: Sequence[aug.AugmentationFinding],
    policies: Dict[str, str],
    cross_class_review_rows: Sequence[Dict[str, Any]] = (),
):
    """Build leakage_report.csv rows and determine which issue types should
    fail the run, per configured policy. Returns (rows, issue_counts, failing_issue_types).

    A cross-class duplicate group only counts toward the "cross_class_duplicate"
    failing policy while it is still UNRESOLVED in cross_class_review_rows —
    once a human has set KEEP_CLASS or EXCLUDE_GROUP, it is reported as
    resolved (severity "info") rather than blocking the run.
    """
    rows: List[Dict[str, Any]] = []
    issue_counts: Dict[str, int] = {}
    failing_issue_types: set = set()

    def add_row(issue_type, severity, canonical_class, class_directory, path, filename, details):
        rows.append(
            {
                "stage": RAW_DATA_AUDIT_STAGE,
                "issue_type": issue_type,
                "severity": severity,
                "canonical_class": canonical_class or "",
                "class_directory": class_directory or "",
                "path": path or "",
                "filename": filename or "",
                "details": details or "",
            }
        )
        issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1

    # 1. Missing configured class directories.
    policy = policies.get("missing_class_directory", DEFAULT_POLICIES["missing_class_directory"])
    for canonical_class, dirname in sorted(directory_validation.missing_mapped_directories.items()):
        add_row(
            "missing_class_directory",
            _severity_for(policy),
            canonical_class,
            dirname,
            None,
            None,
            f"Configured directory '{dirname}' for class '{canonical_class}' not found under raw_dir.",
        )
    if directory_validation.missing_mapped_directories and policy == "fail":
        failing_issue_types.add("missing_class_directory")

    # 2. Unexpected directories not covered by class_directory_mapping.
    policy = policies.get("unexpected_directory", DEFAULT_POLICIES["unexpected_directory"])
    for dirname in directory_validation.unexpected_directories:
        add_row(
            "unexpected_directory",
            _severity_for(policy),
            None,
            dirname,
            None,
            None,
            f"Directory '{dirname}' exists under raw_dir but is not in class_directory_mapping.",
        )
    if directory_validation.unexpected_directories and policy == "fail":
        failing_issue_types.add("unexpected_directory")

    # 3. Corrupted / unreadable images.
    policy = policies.get("corrupted_image", DEFAULT_POLICIES["corrupted_image"])
    corrupted = [r for r in records if not r.is_readable]
    for r in corrupted:
        add_row(
            "corrupted_image",
            _severity_for(policy),
            r.canonical_class,
            r.class_directory,
            r.path,
            r.filename,
            r.error_message,
        )
    if corrupted and policy == "fail":
        failing_issue_types.add("corrupted_image")

    # 4. Exact duplicate groups (same-class and cross-class, reported distinctly).
    same_class_policy = policies.get("same_class_duplicate", DEFAULT_POLICIES["same_class_duplicate"])
    cross_class_policy = policies.get("cross_class_duplicate", DEFAULT_POLICIES["cross_class_duplicate"])
    resolution_by_sha256 = {
        row["sha256"]: row.get("resolution", review.RESOLUTION_UNRESOLVED)
        for row in cross_class_review_rows
    }
    has_same_class_dup = False
    has_unresolved_cross_class_dup = False
    for group in duplicate_groups:
        files_desc = "; ".join(f"{r.canonical_class}/{r.path}" for r in group.records)
        if group.is_cross_class:
            resolution = resolution_by_sha256.get(group.sha256, review.RESOLUTION_UNRESOLVED)
            if resolution == review.RESOLUTION_UNRESOLVED:
                has_unresolved_cross_class_dup = True
                severity = _severity_for(cross_class_policy)
            else:
                severity = "info"  # human-resolved; no longer blocking
            add_row(
                "cross_class_duplicate_group",
                severity,
                ";".join(group.classes),
                None,
                None,
                None,
                f"{group.group_id} sha256={group.sha256} size={group.size} "
                f"resolution={resolution} files=[{files_desc}]",
            )
        else:
            has_same_class_dup = True
            add_row(
                "exact_duplicate_group",
                _severity_for(same_class_policy),
                group.records[0].canonical_class,
                group.records[0].class_directory,
                None,
                None,
                f"{group.group_id} sha256={group.sha256} size={group.size} files=[{files_desc}]",
            )
    if has_unresolved_cross_class_dup and cross_class_policy == "fail":
        failing_issue_types.add("cross_class_duplicate")
    if has_same_class_dup and same_class_policy == "fail":
        failing_issue_types.add("same_class_duplicate")

    # 5. Suspicious augmentation-family findings.
    highly_susp_policy = policies.get(
        "highly_suspicious_augmentation_family",
        DEFAULT_POLICIES["highly_suspicious_augmentation_family"],
    )
    has_highly_suspicious = False
    for finding in augmentation_findings:
        if finding.classification == aug.NORMAL:
            continue
        if finding.classification == aug.HIGHLY_SUSPICIOUS:
            has_highly_suspicious = True
            severity = _severity_for(highly_susp_policy)
        else:
            severity = "info"
        add_row(
            "suspicious_augmentation_family",
            severity,
            finding.record.canonical_class,
            finding.record.class_directory,
            finding.record.path,
            finding.record.filename,
            f"classification={finding.classification} reason={finding.reason}",
        )
    if has_highly_suspicious and highly_susp_policy == "fail":
        failing_issue_types.add("highly_suspicious_augmentation_family")

    return rows, issue_counts, failing_issue_types


@dataclass
class AuditSummary:
    total_images: int
    directory_validation: DirectoryValidation
    dataset_stats: Dict[str, Any]
    corrupted_count: int
    duplicate_groups: List[dup.DuplicateGroup]
    augmentation_findings: List[aug.AugmentationFinding]
    issue_counts: Dict[str, int]
    failing_issue_types: set
    audit_dir: Path
    config_class_names: List[str] = field(default_factory=list)
    cross_class_review_rows: List[Dict[str, Any]] = field(default_factory=list)
    review_csv_path: Optional[Path] = None

    @property
    def passed(self) -> bool:
        return not self.failing_issue_types

    def format_report(self) -> str:
        cross_class = dup.get_cross_class_duplicate_groups(self.duplicate_groups)
        same_class = dup.get_same_class_duplicate_groups(self.duplicate_groups)
        highly_suspicious = sum(
            1 for f in self.augmentation_findings if f.classification == aug.HIGHLY_SUSPICIOUS
        )
        suspicious = sum(
            1 for f in self.augmentation_findings if f.classification == aug.SUSPICIOUS
        )
        unresolved_cross_class = review.count_unresolved(self.cross_class_review_rows)

        lines = [
            "=" * 70,
            "DATASET AUDIT SUMMARY (raw_data_audit)",
            "=" * 70,
            f"Total images inventoried: {self.total_images}",
            f"Configured classes: {len(self.config_class_names)}",
            f"Missing configured class directories: {len(self.directory_validation.missing_mapped_directories)}",
            f"Unexpected directories: {len(self.directory_validation.unexpected_directories)}",
            f"Corrupted/unreadable images: {self.corrupted_count}",
            f"Exact duplicate groups (same-class): {len(same_class)}",
            f"Exact duplicate groups (CROSS-CLASS): {len(cross_class)}",
            f"  of which still UNRESOLVED (human review required): {unresolved_cross_class}",
            f"Highly-suspicious augmentation-family findings: {highly_suspicious}",
            f"Suspicious (weaker) augmentation-family findings: {suspicious}",
            "",
            "Per-class counts:",
        ]
        for cls, count in self.dataset_stats["count_by_class"].items():
            pct = self.dataset_stats["percentage_by_class"].get(cls, 0.0)
            lines.append(f"  {cls}: {count} ({pct}%)")

        lines.append("")
        lines.append(f"Reports written to: {self.audit_dir}")
        if cross_class:
            lines.append(f"Cross-class duplicate review manifest: {self.review_csv_path}")
            lines.append(
                "Inspect a group visually: python -m src.data.review_duplicates --group-id <ID>"
            )
        if self.passed:
            lines.append("RESULT: PASSED (no policy-'fail' issues found)")
        else:
            lines.append(
                "RESULT: FAILED - policy-'fail' issue types: "
                + ", ".join(sorted(self.failing_issue_types))
            )
        lines.append("=" * 70)
        return "\n".join(lines)


def run_dataset_audit(
    config_path: Union[str, Path] = "configs/dataset.yaml",
    raw_dir_override: Optional[Union[str, Path]] = None,
    audit_dir_override: Optional[Union[str, Path]] = None,
) -> AuditSummary:
    """Run the full raw-dataset audit and write all reports under audit_dir.

    Read-only with respect to raw_dir. Always writes every report file
    (even ones with zero rows) before deciding whether to raise
    AuditFailedError, so evidence is never lost even when the audit fails.

    Raises:
        AuditConfigError: fatal, unrecoverable config/directory problems
            (cannot run any meaningful audit at all).
        review.ReviewValidationError: an existing
            cross_class_duplicate_review.csv is structurally invalid (bad
            resolution/class values, duplicate or unknown groups). Raised
            before that file is overwritten, to protect human review work
            already recorded in it.
        AuditFailedError: the audit ran and wrote all reports, but one or
            more issue types breached their configured "fail" policy
            (including any still-UNRESOLVED cross-class duplicate group).
    """
    config = load_config(config_path)
    validate_dataset_config(config)

    class_directory_mapping = config["class_directory_mapping"]
    raw_dir = Path(raw_dir_override) if raw_dir_override is not None else Path(config["paths"]["raw_dir"])
    audit_dir = Path(audit_dir_override) if audit_dir_override is not None else Path(config["paths"]["audit_dir"])

    audit_cfg = config.get("audit", {}) or {}
    supported_extensions = audit_cfg.get("supported_extensions", DEFAULT_SUPPORTED_EXTENSIONS)
    policies = {**DEFAULT_POLICIES, **(audit_cfg.get("policies", {}) or {})}
    keywords = audit_cfg.get("augmentation_keywords", [])

    directory_validation = validate_class_directories(raw_dir, class_directory_mapping)

    records = build_inventory(raw_dir, class_directory_mapping, supported_extensions)

    audit_dir.mkdir(parents=True, exist_ok=True)
    write_csv(records, INVENTORY_COLUMNS, audit_dir / "dataset_inventory.csv")
    write_corrupted_images_csv(records, audit_dir / "corrupted_images.csv")
    stats.write_class_distribution_csv(
        records, class_directory_mapping, audit_dir / "class_distribution.csv"
    )

    duplicate_groups = dup.find_exact_duplicate_groups(records)
    dup.write_duplicate_report_csv(duplicate_groups, audit_dir / "duplicate_report.csv")

    augmentation_findings = aug.analyze_augmentation_families(
        records, keywords, supported_extensions
    )
    aug.write_augmentation_report_csv(
        augmentation_findings, audit_dir / "augmentation_family_report.csv"
    )

    # --- Cross-class / same-class duplicate human-review workflow ---
    # Same-class duplicates carry no label conflict; report and move on.
    same_class_groups = dup.get_same_class_duplicate_groups(duplicate_groups)
    same_class_rows = review.build_same_class_report_rows(same_class_groups)
    review.write_same_class_report_csv(same_class_rows, audit_dir / "same_class_duplicate_report.csv")

    # Cross-class duplicates are label conflicts: never auto-resolved. Merge
    # fresh facts with any pre-existing human resolutions, validating the
    # pre-existing file BEFORE overwriting it so a corrupt file can't
    # silently destroy completed review work.
    cross_class_groups = dup.get_cross_class_duplicate_groups(duplicate_groups)
    review_csv_path = audit_dir / "cross_class_duplicate_review.csv"
    existing_review_rows = review.load_review_csv(review_csv_path)
    if existing_review_rows:
        review.assert_valid_review_rows(existing_review_rows, class_directory_mapping.keys())

    fresh_review_rows = review.build_cross_class_review_rows(cross_class_groups)
    merged_review_rows = review.merge_review_rows(existing_review_rows, fresh_review_rows)
    review.assert_valid_review_rows(
        merged_review_rows,
        class_directory_mapping.keys(),
        expected_sha256_set={g.sha256 for g in cross_class_groups},
    )
    review.write_cross_class_review_csv(merged_review_rows, review_csv_path)
    review.write_summary_csv(
        merged_review_rows,
        class_directory_mapping.keys(),
        audit_dir / "cross_class_duplicate_summary.csv",
    )

    leakage_rows, issue_counts, failing_issue_types = _build_leakage_report(
        directory_validation, records, duplicate_groups, augmentation_findings, policies,
        cross_class_review_rows=merged_review_rows,
    )
    write_csv(leakage_rows, LEAKAGE_REPORT_COLUMNS, audit_dir / "leakage_report.csv")

    summary = AuditSummary(
        total_images=len(records),
        directory_validation=directory_validation,
        dataset_stats=stats.summarize(records),
        corrupted_count=sum(1 for r in records if not r.is_readable),
        duplicate_groups=duplicate_groups,
        augmentation_findings=augmentation_findings,
        issue_counts=issue_counts,
        failing_issue_types=failing_issue_types,
        audit_dir=audit_dir,
        config_class_names=sorted(class_directory_mapping.keys()),
        cross_class_review_rows=merged_review_rows,
        review_csv_path=review_csv_path,
    )

    print(summary.format_report())

    if not summary.passed:
        message_parts = [
            "Dataset audit FAILED policy checks: "
            + ", ".join(sorted(summary.failing_issue_types))
            + "."
        ]
        if "cross_class_duplicate" in summary.failing_issue_types:
            unresolved_count = review.count_unresolved(merged_review_rows)
            message_parts.append(
                f"HUMAN REVIEW REQUIRED: {unresolved_count} cross-class exact-duplicate "
                f"group(s) are unresolved. Resolve each group's 'resolution' column to "
                f"KEEP_CLASS or EXCLUDE_GROUP in {review_csv_path} (never inferred "
                f"automatically). Inspect a group visually with: "
                f"python -m src.data.review_duplicates --group-id <ID>."
            )
        message_parts.append(f"See {audit_dir} for full reports.")
        raise AuditFailedError(" ".join(message_parts))

    return summary

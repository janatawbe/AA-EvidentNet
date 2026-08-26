"""End-to-end tests for run_dataset_audit() on tiny fixture datasets.

None of these tests touch the real 1.7GB DS2 dataset under data/raw/ — each
builds its own miniature raw_dir + dataset.yaml under tmp_path.
"""

import csv

import pytest

from src.data.audit_dataset import AuditConfigError, AuditFailedError, run_dataset_audit
from src.data.duplicate_review import ReviewValidationError
from tests.conftest import make_image, make_invalid_image, write_min_dataset_config

MAPPING = {
    "Alpha": "Alpha",
    "Beta Disease": "Beta Disease [Raw]",
}


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_clean_fixture_dataset_passes_audit(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Alpha" / "a2.jpg", color=(0, 255, 0))
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg", color=(0, 0, 255))

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    summary = run_dataset_audit(config_path=config_path)

    assert summary.passed
    assert summary.total_images == 3
    assert summary.corrupted_count == 0
    assert summary.duplicate_groups == []

    for name in [
        "dataset_inventory.csv",
        "corrupted_images.csv",
        "class_distribution.csv",
        "duplicate_report.csv",
        "augmentation_family_report.csv",
        "leakage_report.csv",
    ]:
        assert (audit_dir / name).exists(), f"missing report: {name}"

    inventory_rows = _read_csv(audit_dir / "dataset_inventory.csv")
    assert len(inventory_rows) == 3

    distribution_rows = _read_csv(audit_dir / "class_distribution.csv")
    by_class = {r["canonical_class"]: r for r in distribution_rows}
    assert by_class["Alpha"]["image_count"] == "2"
    assert by_class["Beta Disease"]["image_count"] == "1"


def test_unexpected_directory_fails_audit_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")
    make_image(raw_dir / "Mystery Class" / "x1.jpg")  # not in mapping

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    with pytest.raises(AuditFailedError, match="unexpected_directory"):
        run_dataset_audit(config_path=config_path)

    # Reports must still be written even though the audit failed.
    leakage_rows = _read_csv(audit_dir / "leakage_report.csv")
    issue_types = {row["issue_type"] for row in leakage_rows}
    assert "unexpected_directory" in issue_types


def test_missing_class_directory_fails_audit_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    # "Beta Disease [Raw]" never created.

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    with pytest.raises(AuditFailedError, match="missing_class_directory"):
        run_dataset_audit(config_path=config_path)


def test_entirely_missing_raw_dir_raises_config_error(tmp_path):
    raw_dir = tmp_path / "does_not_exist"
    audit_dir = tmp_path / "audit"
    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    with pytest.raises(AuditConfigError):
        run_dataset_audit(config_path=config_path)


def test_corrupted_image_fails_audit_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "good.jpg")
    make_invalid_image(raw_dir / "Alpha" / "broken.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    with pytest.raises(AuditFailedError, match="corrupted_image"):
        run_dataset_audit(config_path=config_path)

    corrupted_rows = _read_csv(audit_dir / "corrupted_images.csv")
    assert len(corrupted_rows) == 1
    assert corrupted_rows[0]["filename"] == "broken.jpg"


def test_cross_class_duplicate_fails_audit_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    # Byte-identical content in two different classes.
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(30, 30), color=(1, 2, 3))
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Beta Disease [Raw]" / "b1.jpg")

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    with pytest.raises(AuditFailedError, match="cross_class_duplicate"):
        run_dataset_audit(config_path=config_path)

    dup_rows = _read_csv(audit_dir / "duplicate_report.csv")
    assert len(dup_rows) == 2
    assert all(row["is_cross_class"] == "True" for row in dup_rows)

    leakage_rows = _read_csv(audit_dir / "leakage_report.csv")
    cross_class_rows = [r for r in leakage_rows if r["issue_type"] == "cross_class_duplicate_group"]
    assert len(cross_class_rows) == 1


def test_same_class_duplicate_warns_but_does_not_fail_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(30, 30), color=(9, 9, 9))
    import shutil

    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Alpha" / "a2.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    summary = run_dataset_audit(config_path=config_path)  # must not raise
    assert summary.passed

    leakage_rows = _read_csv(audit_dir / "leakage_report.csv")
    same_class_rows = [r for r in leakage_rows if r["issue_type"] == "exact_duplicate_group"]
    assert len(same_class_rows) == 1
    assert same_class_rows[0]["severity"] == "warning"


def test_suspicious_augmentation_family_warns_but_does_not_fail_by_default(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Alpha" / "a1_flipped.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")

    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    summary = run_dataset_audit(config_path=config_path)  # must not raise (policy=warn)
    assert summary.passed

    aug_rows = _read_csv(audit_dir / "augmentation_family_report.csv")
    assert len(aug_rows) == 1
    assert aug_rows[0]["classification"] == "highly-suspicious"

    leakage_rows = _read_csv(audit_dir / "leakage_report.csv")
    aug_leakage_rows = [r for r in leakage_rows if r["issue_type"] == "suspicious_augmentation_family"]
    assert len(aug_leakage_rows) == 1
    assert aug_leakage_rows[0]["severity"] == "warning"


def test_policy_override_can_make_highly_suspicious_fail(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1_flipped.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")

    config_path = write_min_dataset_config(
        tmp_path,
        MAPPING,
        raw_dir,
        audit_dir,
        policies={"highly_suspicious_augmentation_family": "fail"},
    )

    with pytest.raises(AuditFailedError, match="highly_suspicious_augmentation_family"):
        run_dataset_audit(config_path=config_path)


def test_policy_override_can_make_unexpected_directory_warn_only(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")
    make_image(raw_dir / "Mystery Class" / "x1.jpg")

    config_path = write_min_dataset_config(
        tmp_path,
        MAPPING,
        raw_dir,
        audit_dir,
        policies={"unexpected_directory": "warn"},
    )

    summary = run_dataset_audit(config_path=config_path)  # must not raise
    assert summary.passed


# --- Human-review workflow: cross-class duplicate manifest, merge, validation ---


def _make_cross_class_duplicate_fixture(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(30, 30), color=(5, 6, 7))
    (raw_dir / "Beta Disease [Raw]").mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Beta Disease [Raw]" / "b1.jpg")
    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)
    return config_path, audit_dir


def test_review_manifest_written_with_all_groups_unresolved(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    review_rows = _read_csv(audit_dir / "cross_class_duplicate_review.csv")
    assert len(review_rows) == 1
    assert review_rows[0]["resolution"] == "UNRESOLVED"
    assert review_rows[0]["resolved_class"] == ""


def test_audit_failure_message_says_human_review_required(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError, match="HUMAN REVIEW REQUIRED"):
        run_dataset_audit(config_path=config_path)


def test_rerun_after_human_keep_class_resolution_passes(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    # Simulate a human editing the review manifest directly.
    review_path = audit_dir / "cross_class_duplicate_review.csv"
    rows = _read_csv(review_path)
    rows[0]["resolution"] = "KEEP_CLASS"
    rows[0]["resolved_class"] = "Alpha"
    rows[0]["reviewer"] = "dr_smith"
    rows[0]["notes"] = "chart review confirms Alpha"
    fieldnames = list(rows[0].keys())
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = run_dataset_audit(config_path=config_path)  # must not raise now
    assert summary.passed

    # The human resolution must survive being re-written by this second run.
    rows_after = _read_csv(review_path)
    assert rows_after[0]["resolution"] == "KEEP_CLASS"
    assert rows_after[0]["resolved_class"] == "Alpha"
    assert rows_after[0]["reviewer"] == "dr_smith"


def test_rerun_after_human_exclude_group_resolution_passes(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    review_path = audit_dir / "cross_class_duplicate_review.csv"
    rows = _read_csv(review_path)
    rows[0]["resolution"] = "EXCLUDE_GROUP"
    rows[0]["resolved_class"] = ""
    fieldnames = list(rows[0].keys())
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = run_dataset_audit(config_path=config_path)  # must not raise
    assert summary.passed


def test_corrupted_existing_review_file_raises_and_is_not_overwritten(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    review_path = audit_dir / "cross_class_duplicate_review.csv"
    rows = _read_csv(review_path)
    rows[0]["resolution"] = "MAYBE_MAYBE_NOT"  # invalid value, simulating a typo
    fieldnames = list(rows[0].keys())
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    corrupted_content = review_path.read_text(encoding="utf-8")

    with pytest.raises(ReviewValidationError):
        run_dataset_audit(config_path=config_path)

    # The corrupted human file must be left untouched, not silently overwritten.
    assert review_path.read_text(encoding="utf-8") == corrupted_content


def test_same_class_duplicate_report_generated_separately(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(30, 30), color=(8, 8, 8))
    import shutil

    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Alpha" / "a2.jpg")
    make_image(raw_dir / "Beta Disease [Raw]" / "b1.jpg")
    config_path = write_min_dataset_config(tmp_path, MAPPING, raw_dir, audit_dir)

    summary = run_dataset_audit(config_path=config_path)
    assert summary.passed

    same_class_rows = _read_csv(audit_dir / "same_class_duplicate_report.csv")
    assert len(same_class_rows) == 1
    assert same_class_rows[0]["canonical_class"] == "Alpha"

    # No cross-class review rows since there's no cross-class conflict here.
    cross_class_rows = _read_csv(audit_dir / "cross_class_duplicate_review.csv")
    assert cross_class_rows == []


def test_cross_class_duplicate_summary_reflects_resolution_state(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    summary_rows = _read_csv(audit_dir / "cross_class_duplicate_summary.csv")
    as_dict = {(r["metric"], r["key"]): r["value"] for r in summary_rows}
    assert as_dict[("total_conflicting_groups", "")] == "1"
    assert as_dict[("groups_unresolved", "")] == "1"
    assert as_dict[("total_affected_files", "")] == "2"


# --- Eligibility layer generated by the full orchestrator ---


def test_eligibility_manifest_generated_with_unresolved_group_excluded(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    eligibility_rows = _read_csv(audit_dir / "dataset_eligibility.csv")
    assert len(eligibility_rows) == 2  # a1.jpg (Alpha) + b1.jpg (Beta Disease), both in the conflict
    by_path = {r["path"]: r for r in eligibility_rows}
    assert by_path["Alpha/a1.jpg"]["eligible"] == "false"
    assert by_path["Alpha/a1.jpg"]["exclusion_reason"] == "unresolved_cross_class_exact_duplicate"
    assert by_path["Beta Disease [Raw]/b1.jpg"]["eligible"] == "false"


def test_eligibility_manifest_updates_after_keep_class_resolution(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    review_path = audit_dir / "cross_class_duplicate_review.csv"
    rows = _read_csv(review_path)
    rows[0]["resolution"] = "KEEP_CLASS"
    rows[0]["resolved_class"] = "Alpha"
    fieldnames = list(rows[0].keys())
    with open(review_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary = run_dataset_audit(config_path=config_path)
    assert summary.passed

    eligibility_rows = _read_csv(audit_dir / "dataset_eligibility.csv")
    by_path = {r["path"]: r for r in eligibility_rows}
    assert by_path["Alpha/a1.jpg"]["eligible"] == "true"
    assert by_path["Beta Disease [Raw]/b1.jpg"]["eligible"] == "true"
    # canonical_class must stay each file's original raw label.
    assert by_path["Beta Disease [Raw]/b1.jpg"]["canonical_class"] == "Beta Disease"


def test_eligible_class_distribution_and_summary_generated(tmp_path):
    config_path, audit_dir = _make_cross_class_duplicate_fixture(tmp_path)

    with pytest.raises(AuditFailedError):
        run_dataset_audit(config_path=config_path)

    distribution_rows = _read_csv(audit_dir / "eligible_class_distribution.csv")
    by_class = {r["canonical_class"]: r for r in distribution_rows}
    assert by_class["Alpha"]["raw_count"] == "1"
    assert by_class["Alpha"]["excluded_count"] == "1"
    assert by_class["Alpha"]["eligible_count"] == "0"

    summary_rows = _read_csv(audit_dir / "eligibility_summary.csv")
    as_dict = {r["metric"]: r["value"] for r in summary_rows}
    assert as_dict["total_raw_images"] == "2"
    assert as_dict["excluded_images"] == "2"
    assert as_dict["eligible_images"] == "0"
    assert as_dict["unresolved_groups"] == "1"

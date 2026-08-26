"""End-to-end tests for run_dataset_audit() on tiny fixture datasets.

None of these tests touch the real 1.7GB DS2 dataset under data/raw/ — each
builds its own miniature raw_dir + dataset.yaml under tmp_path.
"""

import csv

import pytest

from src.data.audit_dataset import AuditConfigError, AuditFailedError, run_dataset_audit
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

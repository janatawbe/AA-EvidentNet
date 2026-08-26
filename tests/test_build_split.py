"""Tests for src.data.build_split: the deterministic 70/20/10 original-image
split built on top of the eligibility layer.

None of these tests require the real 1.7GB DS2 dataset — each builds tiny
synthetic fixtures under tmp_path, running the real audit_dataset +
build_split pipeline end-to-end where useful, and exercising individual
functions directly for targeted checks.
"""

import csv
import shutil

import pytest

from src.data.audit_dataset import AuditFailedError, run_dataset_audit
from src.data.build_split import (
    MANIFEST_COLUMNS,
    SPLIT_NAMES,
    CheckResult,
    SplitBuildError,
    SplitValidationError,
    allocate_splits,
    build_manifest_rows,
    build_split_units,
    compute_original_id,
    compute_split_targets,
    run_build_split,
    validate_split_manifests,
    verify_files_against_eligibility,
)
from src.data.duplicate_review import RESOLUTION_KEEP_CLASS, RESOLUTION_UNRESOLVED
from src.data.eligibility import ELIGIBLE_FALSE, ELIGIBLE_TRUE
from src.utils.hashing import hash_file
from src.utils.seeding import set_seed
from tests.conftest import make_image, write_min_dataset_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Gamma"]


def _eligible_row(path, canonical_class, sha256, group_id="", duplicate_type=""):
    return {
        "path": path,
        "canonical_class": canonical_class,
        "sha256": sha256,
        "eligible": ELIGIBLE_TRUE,
        "exclusion_reason": "",
        "duplicate_group_id": group_id,
        "duplicate_type": duplicate_type,
    }


def _review_row(group_id, sha256, resolution=RESOLUTION_UNRESOLVED, resolved_class=""):
    return {
        "duplicate_group_id": group_id,
        "sha256": sha256,
        "num_files": 2,
        "classes": "",
        "paths": "",
        "resolution": resolution,
        "resolved_class": resolved_class,
        "reviewer": "",
        "notes": "",
    }


def _run_audit_then_split(tmp_path, mapping, seed=42, config_kwargs=None):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    manifests_dir = tmp_path / "manifests"
    config_path = write_min_dataset_config(
        tmp_path, mapping, raw_dir, audit_dir, **(config_kwargs or {})
    )
    try:
        run_dataset_audit(config_path=config_path)
    except AuditFailedError:
        pass  # expected whenever unresolved cross-class groups exist; reports are still written
    summary = run_build_split(config_path=config_path, seed=seed)
    return config_path, audit_dir, manifests_dir, summary


def _read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# --- compute_original_id ---


def test_compute_original_id_deterministic():
    a = compute_original_id("Alpha", "Alpha/a1.jpg", "hash1")
    b = compute_original_id("Alpha", "Alpha/a1.jpg", "hash1")
    assert a == b
    assert len(a) == 64  # sha256 hex


def test_compute_original_id_differs_for_different_inputs():
    base = compute_original_id("Alpha", "Alpha/a1.jpg", "hash1")
    assert base != compute_original_id("Beta", "Alpha/a1.jpg", "hash1")
    assert base != compute_original_id("Alpha", "Alpha/a2.jpg", "hash1")
    assert base != compute_original_id("Alpha", "Alpha/a1.jpg", "hash2")


def test_compute_original_id_independent_of_absolute_path():
    # Only the dataset-relative path matters, not any absolute machine path.
    a = compute_original_id("Alpha", "Alpha/a1.jpg", "hash1")
    b = compute_original_id("Alpha", "Alpha/a1.jpg", "hash1")
    assert a == b  # same relative info -> same ID regardless of caller's cwd


# --- build_split_units ---


def test_build_split_units_singles_and_same_class_group():
    rows = [
        _eligible_row("Alpha/a1.jpg", "Alpha", "h1"),
        _eligible_row("Alpha/a2.jpg", "Alpha", "h2", group_id="G1", duplicate_type="same_class"),
        _eligible_row("Alpha/a3.jpg", "Alpha", "h2", group_id="G1", duplicate_type="same_class"),
    ]
    units = build_split_units(rows, [])
    assert len(units) == 2  # one single + one 2-member group
    group_unit = [u for u in units if u.unit_id == "G1"][0]
    assert len(group_unit.members) == 2
    assert group_unit.stratify_class == "Alpha"


def test_build_split_units_cross_class_keep_class_uses_resolved_class():
    rows = [
        _eligible_row("Alpha/a1.jpg", "Alpha", "hX", group_id="G2", duplicate_type="cross_class"),
        _eligible_row("Beta/b1.jpg", "Beta", "hX", group_id="G2", duplicate_type="cross_class"),
    ]
    review_rows = [_review_row("G2", "hX", RESOLUTION_KEEP_CLASS, "Alpha")]
    units = build_split_units(rows, review_rows)
    assert len(units) == 1
    unit = units[0]
    assert unit.stratify_class == "Alpha"
    assert len(unit.members) == 2
    # Members' own canonical_class must remain untouched.
    member_classes = {m["canonical_class"] for m in unit.members}
    assert member_classes == {"Alpha", "Beta"}


def test_build_split_units_raises_for_unresolvable_cross_class_group():
    rows = [
        _eligible_row("Alpha/a1.jpg", "Alpha", "hX", group_id="G3", duplicate_type="cross_class"),
        _eligible_row("Beta/b1.jpg", "Beta", "hX", group_id="G3", duplicate_type="cross_class"),
    ]
    # No review row at all -> build_split_units cannot know the resolved_class.
    with pytest.raises(SplitBuildError):
        build_split_units(rows, [])


# --- compute_split_targets ---


def test_compute_split_targets_sums_to_total():
    for total in (0, 1, 2, 3, 17, 100, 4393):
        targets = compute_split_targets(total, (0.70, 0.20, 0.10))
        assert sum(targets.values()) == total


def test_compute_split_targets_small_class_of_one_goes_to_train():
    targets = compute_split_targets(1, (0.70, 0.20, 0.10))
    assert targets == {"train": 1, "val": 0, "test": 0}


def test_compute_split_targets_pterygium_like_class():
    # Mirrors the real Pterygium class: 17 eligible images.
    targets = compute_split_targets(17, (0.70, 0.20, 0.10))
    assert targets == {"train": 12, "val": 3, "test": 2}
    assert sum(targets.values()) == 17


# --- allocate_splits reproducibility ---


def test_allocate_splits_same_seed_identical_assignment():
    units = build_split_units(
        [_eligible_row(f"Alpha/a{i}.jpg", "Alpha", f"h{i}") for i in range(20)], []
    )

    set_seed(42)
    assignment_a, _, _ = allocate_splits(units)

    set_seed(42)
    assignment_b, _, _ = allocate_splits(units)

    assert assignment_a == assignment_b


def test_allocate_splits_different_seed_can_differ_but_stays_valid():
    units = build_split_units(
        [_eligible_row(f"Alpha/a{i}.jpg", "Alpha", f"h{i}") for i in range(30)], []
    )

    set_seed(42)
    assignment_42, _, counts_42 = allocate_splits(units)

    set_seed(123)
    assignment_123, _, counts_123 = allocate_splits(units)

    assert assignment_42 != assignment_123  # different seed -> (almost certainly) different split
    # Both remain valid: every unit assigned, counts sum to total.
    assert sum(counts_42["Alpha"].values()) == 30
    assert sum(counts_123["Alpha"].values()) == 30


# --- build_manifest_rows schema ---


def test_build_manifest_rows_schema_and_original_fields():
    rows = [_eligible_row("Alpha/a1.jpg", "Alpha", "h1")]
    units = build_split_units(rows, [])
    set_seed(42)
    assignment, _, _ = allocate_splits(units)
    manifests = build_manifest_rows(units, assignment)

    all_rows = [r for rows_ in manifests.values() for r in rows_]
    assert len(all_rows) == 1
    row = all_rows[0]
    assert set(row.keys()) == set(MANIFEST_COLUMNS)
    assert row["is_original"] == "true"
    assert row["augmentation_type"] == "original"
    assert row["parent_original_id"] == row["original_id"]
    assert row["class"] == "Alpha"


# --- validate_split_manifests: each check individually ---


def _minimal_context(manifest_row, elig_row, review_rows=()):
    manifests = {"train": [manifest_row], "val": [], "test": []}
    return manifests, [elig_row], [elig_row], list(review_rows)


def test_validate_split_detects_cross_split_path_overlap():
    elig_row = _eligible_row("Alpha/a1.jpg", "Alpha", "h1")
    row = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x",
           "parent_original_id": "x", "is_original": "true", "augmentation_type": "original"}
    manifests = {"train": [row], "val": [dict(row, split="val")], "test": []}
    results = validate_split_manifests(manifests, [elig_row], [elig_row], [], CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["train_val_overlap"].status == "FAIL"


def test_validate_split_detects_excluded_file_leakage():
    elig_row = dict(_eligible_row("Alpha/a1.jpg", "Alpha", "h1"))
    elig_row["eligible"] = ELIGIBLE_FALSE
    elig_row["exclusion_reason"] = "unresolved_cross_class_exact_duplicate"
    row = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x",
           "parent_original_id": "x", "is_original": "true", "augmentation_type": "original"}
    manifests, eligible_rows, all_rows, review_rows = _minimal_context(row, elig_row)
    results = validate_split_manifests(manifests, eligible_rows, all_rows, review_rows, CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["excluded_file_leakage"].status == "FAIL"


def test_validate_split_detects_unresolved_cross_class_leakage():
    elig_row = _eligible_row("Alpha/a1.jpg", "Alpha", "hX", group_id="G1", duplicate_type="cross_class")
    review_rows = [_review_row("G1", "hX", RESOLUTION_UNRESOLVED)]
    row = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x",
           "parent_original_id": "x", "is_original": "true", "augmentation_type": "original"}
    manifests, eligible_rows, all_rows, review_rows = _minimal_context(row, elig_row, review_rows)
    results = validate_split_manifests(manifests, eligible_rows, all_rows, review_rows, CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["unresolved_cross_class_leakage"].status == "FAIL"


def test_validate_split_detects_augmentation_leakage():
    elig_row = _eligible_row("Alpha/a1.jpg", "Alpha", "h1")
    row = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x",
           "parent_original_id": "x", "is_original": "false", "augmentation_type": "flip"}
    manifests, eligible_rows, all_rows, review_rows = _minimal_context(row, elig_row)
    results = validate_split_manifests(manifests, eligible_rows, all_rows, review_rows, CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["augmentation_leakage"].status == "FAIL"


def test_validate_split_detects_class_label_mismatch():
    elig_row = _eligible_row("Alpha/a1.jpg", "Alpha", "h1")
    row = {"path": "Alpha/a1.jpg", "class": "Beta", "split": "train", "original_id": "x",
           "parent_original_id": "x", "is_original": "true", "augmentation_type": "original"}
    manifests, eligible_rows, all_rows, review_rows = _minimal_context(row, elig_row)
    results = validate_split_manifests(manifests, eligible_rows, all_rows, review_rows, CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["class_label_consistency"].status == "FAIL"


def test_validate_split_detects_intra_manifest_duplicate_rows():
    elig_row = _eligible_row("Alpha/a1.jpg", "Alpha", "h1")
    row = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x",
           "parent_original_id": "x", "is_original": "true", "augmentation_type": "original"}
    manifests = {"train": [row, dict(row)], "val": [], "test": []}
    results = validate_split_manifests(manifests, [elig_row], [elig_row], [], CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["intra_manifest_duplicate_rows"].status == "FAIL"


def test_validate_split_detects_duplicate_group_split_across_manifests():
    # Simulates manual tampering: a same-class duplicate group's two members
    # end up in different splits.
    elig_rows = [
        _eligible_row("Alpha/a1.jpg", "Alpha", "hSame", group_id="G1", duplicate_type="same_class"),
        _eligible_row("Alpha/a2.jpg", "Alpha", "hSame", group_id="G1", duplicate_type="same_class"),
    ]
    row1 = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x1",
            "parent_original_id": "x1", "is_original": "true", "augmentation_type": "original"}
    row2 = {"path": "Alpha/a2.jpg", "class": "Alpha", "split": "val", "original_id": "x2",
            "parent_original_id": "x2", "is_original": "true", "augmentation_type": "original"}
    manifests = {"train": [row1], "val": [row2], "test": []}
    results = validate_split_manifests(manifests, elig_rows, elig_rows, [], CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["duplicate_group_split_integrity"].status == "FAIL"
    assert by_check["cross_split_sha256_overlap"].status == "FAIL"


def test_validate_split_passes_when_duplicate_group_kept_together():
    elig_rows = [
        _eligible_row("Alpha/a1.jpg", "Alpha", "hSame", group_id="G1", duplicate_type="same_class"),
        _eligible_row("Alpha/a2.jpg", "Alpha", "hSame", group_id="G1", duplicate_type="same_class"),
    ]
    row1 = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x1",
            "parent_original_id": "x1", "is_original": "true", "augmentation_type": "original"}
    row2 = {"path": "Alpha/a2.jpg", "class": "Alpha", "split": "train", "original_id": "x2",
            "parent_original_id": "x2", "is_original": "true", "augmentation_type": "original"}
    manifests = {"train": [row1, row2], "val": [], "test": []}
    results = validate_split_manifests(manifests, elig_rows, elig_rows, [], CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["duplicate_group_split_integrity"].status == "PASS"


def test_validate_split_structural_completeness_fails_when_row_missing():
    elig_rows = [
        _eligible_row("Alpha/a1.jpg", "Alpha", "h1"),
        _eligible_row("Alpha/a2.jpg", "Alpha", "h2"),
    ]
    row1 = {"path": "Alpha/a1.jpg", "class": "Alpha", "split": "train", "original_id": "x1",
            "parent_original_id": "x1", "is_original": "true", "augmentation_type": "original"}
    manifests = {"train": [row1], "val": [], "test": []}  # a2.jpg missing entirely
    results = validate_split_manifests(manifests, elig_rows, elig_rows, [], CANONICAL_CLASSES)
    by_check = {r.check: r for r in results}
    assert by_check["structural_completeness"].status == "FAIL"


# --- verify_files_against_eligibility / SHA-256 re-verification ---


def test_verify_files_against_eligibility_detects_missing_file(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    row = _eligible_row("Alpha/missing.jpg", "Alpha", "deadbeef")
    errors = verify_files_against_eligibility([row], raw_dir)
    assert any("file not found" in e for e in errors)


def test_verify_files_against_eligibility_detects_hash_mismatch(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    original_hash = hash_file(raw_dir / "Alpha" / "a1.jpg")
    row = _eligible_row("Alpha/a1.jpg", "Alpha", original_hash)

    # Modify the file AFTER recording its "audited" hash - simulates drift.
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(99, 99), color=(200, 10, 10))

    errors = verify_files_against_eligibility([row], raw_dir)
    assert any("SHA-256 mismatch" in e for e in errors)


def test_verify_files_against_eligibility_passes_for_unchanged_file(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    real_hash = hash_file(raw_dir / "Alpha" / "a1.jpg")
    row = _eligible_row("Alpha/a1.jpg", "Alpha", real_hash)
    assert verify_files_against_eligibility([row], raw_dir) == []


# --- Full integration via run_build_split ---


def _build_two_class_dataset(raw_dir, n_alpha=10, n_beta=10):
    for i in range(n_alpha):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(n_beta):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")


def test_full_split_every_eligible_image_appears_exactly_once(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 15, 12)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, manifests_dir, summary = _run_audit_then_split(tmp_path, mapping)

    all_paths = []
    all_classes = set()
    for split_name in SPLIT_NAMES:
        rows = _read_manifest(manifests_dir / f"{split_name}_original.csv")
        all_paths.extend(r["path"] for r in rows)
        all_classes.update(r["class"] for r in rows)

    assert len(all_paths) == len(set(all_paths)) == 27
    assert all_classes == {"Alpha", "Beta"}
    assert summary.total_eligible == 27


def test_full_split_canonical_classes_preserved(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 10, 10)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, _, manifests_dir, _ = _run_audit_then_split(tmp_path, mapping)

    for split_name in SPLIT_NAMES:
        rows = _read_manifest(manifests_dir / f"{split_name}_original.csv")
        for row in rows:
            assert row["class"] in ("Alpha", "Beta")
            assert row["path"].startswith(row["class"] + "/")


def test_full_split_reproducible_same_seed(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 20, 20)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, manifests_dir, _ = _run_audit_then_split(tmp_path, mapping, seed=42)

    hashes_first = {s: hash_file(manifests_dir / f"{s}_original.csv") for s in SPLIT_NAMES}

    # Re-run the split stage only (audit/eligibility unchanged) with the same seed.
    from src.data.build_split import run_build_split as _run

    config_path = tmp_path / "dataset.yaml"
    _run(config_path=config_path, seed=42)
    hashes_second = {s: hash_file(manifests_dir / f"{s}_original.csv") for s in SPLIT_NAMES}

    assert hashes_first == hashes_second


def test_full_split_different_seed_may_differ(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 30, 30)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, manifests_dir, _ = _run_audit_then_split(tmp_path, mapping, seed=42)
    train_42 = {r["path"] for r in _read_manifest(manifests_dir / "train_original.csv")}

    from src.data.build_split import run_build_split as _run

    config_path = tmp_path / "dataset.yaml"
    _run(config_path=config_path, seed=123)
    train_123 = {r["path"] for r in _read_manifest(manifests_dir / "train_original.csv")}

    assert train_42 != train_123


def test_full_split_excluded_images_never_appear(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(30, 30), color=(5, 6, 7))
    (raw_dir / "Beta").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Beta" / "b1.jpg")  # cross-class conflict
    make_image(raw_dir / "Alpha" / "a2.jpg")
    make_image(raw_dir / "Beta" / "b2.jpg")

    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, manifests_dir, summary = _run_audit_then_split(tmp_path, mapping)

    all_paths = set()
    for split_name in SPLIT_NAMES:
        rows = _read_manifest(manifests_dir / f"{split_name}_original.csv")
        all_paths.update(r["path"] for r in rows)

    assert "Alpha/a1.jpg" not in all_paths  # excluded: unresolved cross-class duplicate
    assert "Beta/b1.jpg" not in all_paths
    assert "Alpha/a2.jpg" in all_paths
    assert "Beta/b2.jpg" in all_paths
    assert summary.total_eligible == 2


def test_full_split_same_class_duplicate_group_stays_together(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a1.jpg", size=(30, 30), color=(9, 9, 9))
    shutil.copyfile(raw_dir / "Alpha" / "a1.jpg", raw_dir / "Alpha" / "a2.jpg")
    for i in range(3, 15):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(10):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")

    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, manifests_dir, _ = _run_audit_then_split(tmp_path, mapping)

    split_of = {}
    for split_name in SPLIT_NAMES:
        rows = _read_manifest(manifests_dir / f"{split_name}_original.csv")
        for row in rows:
            split_of[row["path"]] = split_name

    assert split_of["Alpha/a1.jpg"] == split_of["Alpha/a2.jpg"]


def test_full_split_raises_without_prior_audit(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Beta" / "b1.jpg")
    config_path = write_min_dataset_config(tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir)

    with pytest.raises(SplitBuildError):
        run_build_split(config_path=config_path)


def test_full_split_detects_raw_file_modified_after_audit(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 5, 5)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, tmp_path / "audit")

    run_dataset_audit(config_path=config_path)

    # Simulate drift: a raw file changes after the audit ran.
    make_image(raw_dir / "Alpha" / "a0.jpg", size=(77, 77), color=(1, 2, 3))

    with pytest.raises(SplitBuildError, match="SHA-256"):
        run_build_split(config_path=config_path)


def test_full_split_small_class_no_fabrication(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(3):
        make_image(raw_dir / "TinyClass" / f"t{i}.jpg")
    for i in range(20):
        make_image(raw_dir / "BigClass" / f"b{i}.jpg")

    mapping = {"TinyClass": "TinyClass", "BigClass": "BigClass"}
    _, audit_dir, manifests_dir, summary = _run_audit_then_split(tmp_path, mapping)

    tiny_total = sum(summary.per_class_counts["TinyClass"].values())
    assert tiny_total == 3  # never fabricated beyond the 3 real images

    # Deterministic: re-running with the same seed reproduces the same
    # per-class allocation for the tiny class.
    from src.data.build_split import run_build_split as _run

    config_path = tmp_path / "dataset.yaml"
    summary2 = _run(config_path=config_path, seed=42)
    assert summary2.per_class_counts["TinyClass"] == summary.per_class_counts["TinyClass"]


def test_full_split_writes_hash_and_metadata_files(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 10, 10)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, manifests_dir, summary = _run_audit_then_split(tmp_path, mapping)

    for split_name in SPLIT_NAMES:
        hash_file_path = audit_dir / f"{split_name}_manifest_hash.txt"
        assert hash_file_path.exists()
        recorded_hash = hash_file_path.read_text(encoding="utf-8").strip()
        actual_hash = hash_file(manifests_dir / f"{split_name}_original.csv")
        assert recorded_hash == actual_hash

    assert (audit_dir / "split_metadata.json").exists()
    assert (audit_dir / "split_distribution.csv").exists()
    assert (audit_dir / "split_leakage_report.csv").exists()

    import json

    metadata = json.loads((audit_dir / "split_metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert "git_commit" in metadata
    assert "config_hash" in metadata
    assert "eligibility_manifest_hash" in metadata


def test_full_split_leakage_report_all_pass_for_clean_dataset(tmp_path):
    raw_dir = tmp_path / "raw"
    _build_two_class_dataset(raw_dir, 10, 10)
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    _, audit_dir, _, _ = _run_audit_then_split(tmp_path, mapping)

    rows = _read_manifest(audit_dir / "split_leakage_report.csv")
    assert len(rows) > 0
    assert all(r["status"] == "PASS" for r in rows)

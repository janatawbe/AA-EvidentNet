"""Tests for src.data.generate_balanced_dataset: the balanced (target/class)
training-set generator built on augmentation of train_original.csv only.

None of these tests require the real 1.7GB DS2 dataset — each builds tiny
synthetic fixtures under tmp_path and runs the real audit -> split ->
balance pipeline end-to-end, or exercises individual functions directly.
"""

import csv
import shutil

import pytest
from PIL import Image

from src.data.audit_dataset import AuditFailedError, run_dataset_audit
from src.data.build_split import IS_ORIGINAL_TRUE, run_build_split
from src.data.generate_balanced_dataset import (
    AUG_COMBINED,
    AUGMENTATION_STATISTICS_COLUMNS,
    BalancedDatasetBuildError,
    BalancedDatasetValidationError,
    RECIPES,
    assert_no_val_test_contamination,
    build_augmentation_statistics_rows,
    compute_generated_id,
    generate_records_for_class,
    get_active_recipes,
    run_generate_balanced_dataset,
    validate_balanced_manifest,
)
from src.utils.seeding import set_seed
from tests.conftest import make_image, write_min_dataset_config

CANONICAL_CLASSES = ["Alpha", "Beta"]


def _original_row(path, canonical_class, original_id):
    return {
        "path": path,
        "class": canonical_class,
        "split": "train",
        "original_id": original_id,
        "parent_original_id": original_id,
        "is_original": "true",
        "augmentation_type": "original",
    }


def _read_manifest(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _run_full_pipeline(tmp_path, mapping, seed=42, augmentation=None):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    manifests_dir = tmp_path / "manifests"
    config_path = write_min_dataset_config(
        tmp_path, mapping, raw_dir, audit_dir, augmentation=augmentation
    )
    try:
        run_dataset_audit(config_path=config_path)
    except AuditFailedError:
        pass
    run_build_split(config_path=config_path, seed=seed)
    summary = run_generate_balanced_dataset(config_path=config_path, seed=seed)
    return config_path, audit_dir, manifests_dir, summary


# --- compute_generated_id ---


def test_compute_generated_id_deterministic():
    a = compute_generated_id("p1", "Alpha", 0, 42, "cfghash")
    b = compute_generated_id("p1", "Alpha", 0, 42, "cfghash")
    assert a == b
    assert len(a) == 64


def test_compute_generated_id_differs_for_different_inputs():
    base = compute_generated_id("p1", "Alpha", 0, 42, "cfghash")
    assert base != compute_generated_id("p2", "Alpha", 0, 42, "cfghash")
    assert base != compute_generated_id("p1", "Beta", 0, 42, "cfghash")
    assert base != compute_generated_id("p1", "Alpha", 1, 42, "cfghash")
    assert base != compute_generated_id("p1", "Alpha", 0, 123, "cfghash")
    assert base != compute_generated_id("p1", "Alpha", 0, 42, "otherhash")


# --- get_active_recipes ---


def test_get_active_recipes_all_enabled_includes_combined():
    cfg = {
        "horizontal_flip": {"enabled": True},
        "rotation": {"enabled": True},
        "brightness": {"enabled": True},
        "contrast": {"enabled": True},
        "affine": {"enabled": True},
        "color_jitter": {"enabled": True},
    }
    recipes = get_active_recipes(cfg)
    assert AUG_COMBINED in recipes
    assert "horizontal_flip" in recipes
    assert set(recipes).issubset(set(RECIPES.keys()))


def test_get_active_recipes_respects_disabled_flags():
    cfg = {
        "horizontal_flip": {"enabled": False},
        "rotation": {"enabled": True},
        "brightness": {"enabled": False},
        "contrast": {"enabled": False},
        "affine": {"enabled": False},
        "color_jitter": {"enabled": False},
    }
    recipes = get_active_recipes(cfg)
    assert "horizontal_flip" not in recipes
    assert "brightness_contrast" not in recipes
    assert "rotation" in recipes
    assert AUG_COMBINED in recipes  # combined stays available since rotation is on


def test_get_active_recipes_empty_when_everything_disabled():
    cfg = {
        "horizontal_flip": {"enabled": False},
        "rotation": {"enabled": False},
        "brightness": {"enabled": False},
        "contrast": {"enabled": False},
        "affine": {"enabled": False},
        "color_jitter": {"enabled": False},
    }
    assert get_active_recipes(cfg) == []


# --- assert_no_val_test_contamination ---


def test_assert_no_val_test_contamination_passes_for_disjoint_sets():
    train_rows = [_original_row("Alpha/a1.jpg", "Alpha", "id1")]
    val_rows = [_original_row("Alpha/a2.jpg", "Alpha", "id2")]
    test_rows = [_original_row("Alpha/a3.jpg", "Alpha", "id3")]
    assert_no_val_test_contamination(train_rows, val_rows, test_rows)  # must not raise


def test_assert_no_val_test_contamination_detects_val_overlap():
    train_rows = [_original_row("Alpha/a1.jpg", "Alpha", "id1")]
    val_rows = [_original_row("Alpha/a1.jpg", "Alpha", "id1")]  # same path+id as train
    with pytest.raises(BalancedDatasetBuildError, match="validation"):
        assert_no_val_test_contamination(train_rows, val_rows, [])


def test_assert_no_val_test_contamination_detects_test_overlap():
    train_rows = [_original_row("Alpha/a1.jpg", "Alpha", "id1")]
    test_rows = [_original_row("Alpha/a1.jpg", "Alpha", "id1")]
    with pytest.raises(BalancedDatasetBuildError, match="test"):
        assert_no_val_test_contamination(train_rows, [], test_rows)


# --- generate_records_for_class: target balancing ---


def test_generate_records_for_class_tiny_classes_hit_exact_target(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    cfg = {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "rotation": {"enabled": True, "degrees": 10},
        "brightness": {"enabled": True, "factor": 0.15},
        "contrast": {"enabled": True, "factor": 0.15},
        "affine": {"enabled": True, "translate": 0.05, "scale": {"min": 0.95, "max": 1.05}},
        "color_jitter": {"enabled": True, "factor": 0.10},
    }

    for name, count in (("A", 2), ("B", 3), ("C", 1)):
        make_image(raw_dir / name / f"{name}0.jpg")
        for i in range(1, count):
            make_image(raw_dir / name / f"{name}{i}.jpg")

    target = 5
    for name, count in (("A", 2), ("B", 3), ("C", 1)):
        original_rows = [
            _original_row(f"{name}/{name}{i}.jpg", name, f"id_{name}_{i}") for i in range(count)
        ]
        rows, generated_files = generate_records_for_class(
            name, original_rows, target, 42, cfg, "cfghash", raw_dir, processed_train_dir
        )
        assert len(rows) == target
        assert len(generated_files) == target - count


def test_generate_records_for_class_needs_zero_when_already_at_target(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    make_image(raw_dir / "Alpha" / "a1.jpg")
    original_rows = [
        _original_row("Alpha/a0.jpg", "Alpha", "id0"),
        _original_row("Alpha/a1.jpg", "Alpha", "id1"),
    ]
    rows, generated_files = generate_records_for_class(
        "Alpha", original_rows, 2, 42, {}, "cfghash", raw_dir, processed_train_dir
    )
    assert rows == original_rows
    assert generated_files == []


def test_generate_records_for_class_raises_when_originals_exceed_target(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    make_image(raw_dir / "Alpha" / "a1.jpg")
    original_rows = [
        _original_row("Alpha/a0.jpg", "Alpha", "id0"),
        _original_row("Alpha/a1.jpg", "Alpha", "id1"),
    ]
    with pytest.raises(BalancedDatasetBuildError, match="already has"):
        generate_records_for_class("Alpha", original_rows, 1, 42, {}, "cfghash", raw_dir, processed_train_dir)


def test_generate_records_for_class_raises_when_no_originals_but_needed(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    with pytest.raises(BalancedDatasetBuildError, match="zero original"):
        generate_records_for_class("Alpha", [], 5, 42, {}, "cfghash", raw_dir, processed_train_dir)


def test_generate_records_for_class_raises_when_no_recipes_active(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "id0")]
    all_disabled_cfg = {
        "horizontal_flip": {"enabled": False},
        "rotation": {"enabled": False},
        "brightness": {"enabled": False},
        "contrast": {"enabled": False},
        "affine": {"enabled": False},
        "color_jitter": {"enabled": False},
    }
    with pytest.raises(BalancedDatasetBuildError, match="no augmentation recipes"):
        generate_records_for_class(
            "Alpha", original_rows, 5, 42, all_disabled_cfg, "cfghash", raw_dir, processed_train_dir
        )


# --- parent lineage / generated record shape ---


def test_generated_records_have_correct_lineage_fields(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "parent_id_0")]
    cfg = {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "rotation": {"enabled": True, "degrees": 10},
    }
    rows, _ = generate_records_for_class("Alpha", original_rows, 4, 42, cfg, "cfghash", raw_dir, processed_train_dir)

    generated = [r for r in rows if r["is_original"] == "false"]
    assert len(generated) == 3
    for row in generated:
        assert row["parent_original_id"] == "parent_id_0"
        assert row["is_original"] == "false"
        assert row["class"] == "Alpha"
        assert row["split"] == "train"
        assert row["augmentation_type"] in RECIPES.keys()
        assert row["augmentation_type"] != "original"
        assert row["original_id"] != row["parent_original_id"]


# --- no recursive augmentation ---


def test_generated_sample_can_never_become_a_parent(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "parent_id_0")]
    cfg = {"rotation": {"enabled": True, "degrees": 10}}
    rows, _ = generate_records_for_class("Alpha", original_rows, 5, 42, cfg, "cfghash", raw_dir, processed_train_dir)

    original_ids = {r["original_id"] for r in rows if r["is_original"] == "true"}
    for row in rows:
        if row["is_original"] == "false":
            # Every generated row's parent must be one of the ORIGINAL ids,
            # never another generated row's id.
            assert row["parent_original_id"] in original_ids


# --- class preservation ---


def test_generated_class_always_equals_parent_class(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Beta" / "b0.jpg")
    original_rows = [_original_row("Beta/b0.jpg", "Beta", "pid")]
    cfg = {"color_jitter": {"enabled": True, "factor": 0.1}}
    rows, _ = generate_records_for_class("Beta", original_rows, 6, 42, cfg, "cfghash", raw_dir, processed_train_dir)
    assert all(r["class"] == "Beta" for r in rows)


# --- augmentation validity: readable, dimensions preserved ---


def test_generated_images_are_readable_and_preserve_dimensions(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg", size=(64, 48))
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "pid")]
    cfg = {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "rotation": {"enabled": True, "degrees": 10},
        "affine": {"enabled": True, "translate": 0.05, "scale": {"min": 0.95, "max": 1.05}},
    }
    rows, generated_files = generate_records_for_class(
        "Alpha", original_rows, 6, 42, cfg, "cfghash", raw_dir, processed_train_dir
    )
    assert len(generated_files) == 5
    for gf in generated_files:
        assert gf.output_path.is_file()
        with Image.open(gf.output_path) as img:
            img.verify()
        with Image.open(gf.output_path) as img2:
            img2.load()
            assert img2.size == (64, 48)  # transforms preserve canvas size


def test_generated_images_are_not_pixel_identical_to_parent(tmp_path):
    raw_dir = tmp_path / "raw"
    processed_train_dir = tmp_path / "processed" / "train"
    make_image(raw_dir / "Alpha" / "a0.jpg", size=(50, 50), color=(30, 60, 90))
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "pid")]
    cfg = {
        "horizontal_flip": {"enabled": True, "probability": 0.5},
        "rotation": {"enabled": True, "degrees": 10},
        "brightness": {"enabled": True, "factor": 0.15},
        "contrast": {"enabled": True, "factor": 0.15},
        "affine": {"enabled": True, "translate": 0.05, "scale": {"min": 0.95, "max": 1.05}},
        "color_jitter": {"enabled": True, "factor": 0.10},
    }
    _, generated_files = generate_records_for_class(
        "Alpha", original_rows, 7, 42, cfg, "cfghash", raw_dir, processed_train_dir
    )
    with open(raw_dir / "Alpha" / "a0.jpg", "rb") as f:
        parent_bytes = f.read()
    for gf in generated_files:
        with open(gf.output_path, "rb") as f:
            generated_bytes = f.read()
        assert generated_bytes != parent_bytes  # never a disguised duplicate


# --- determinism ---


def test_generate_records_for_class_deterministic_same_seed(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "pid")]
    cfg = {"rotation": {"enabled": True, "degrees": 10}}

    rows_a, _ = generate_records_for_class(
        "Alpha", original_rows, 5, 42, cfg, "cfghash", raw_dir, tmp_path / "run1"
    )
    rows_b, _ = generate_records_for_class(
        "Alpha", original_rows, 5, 42, cfg, "cfghash", raw_dir, tmp_path / "run2"
    )

    ids_a = [r["original_id"] for r in rows_a]
    ids_b = [r["original_id"] for r in rows_b]
    assert ids_a == ids_b

    # Underlying bytes should also be identical given identical seed/config.
    for i in range(len(rows_a)):
        if rows_a[i]["is_original"] == "false":
            path_a = tmp_path / "run1" / rows_a[i]["path"]
            path_b = tmp_path / "run2" / rows_b[i]["path"]
            assert path_a.read_bytes() == path_b.read_bytes()


def test_generate_records_for_class_different_seed_different_ids(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Alpha" / "a0.jpg")
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "pid")]
    cfg = {"rotation": {"enabled": True, "degrees": 10}}

    rows_42, _ = generate_records_for_class(
        "Alpha", original_rows, 5, 42, cfg, "cfghash", raw_dir, tmp_path / "run42"
    )
    rows_123, _ = generate_records_for_class(
        "Alpha", original_rows, 5, 123, cfg, "cfghash", raw_dir, tmp_path / "run123"
    )
    ids_42 = {r["original_id"] for r in rows_42 if r["is_original"] == "false"}
    ids_123 = {r["original_id"] for r in rows_123 if r["is_original"] == "false"}
    assert ids_42 != ids_123


# --- augmentation statistics ---


def test_build_augmentation_statistics_rows_reflects_counts():
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "id0"), _original_row("Alpha/a1.jpg", "Alpha", "id1")]
    generated_rows = [
        {"parent_original_id": "id0", "augmentation_type": "rotation"},
        {"parent_original_id": "id0", "augmentation_type": "rotation"},
        {"parent_original_id": "id1", "augmentation_type": "horizontal_flip"},
    ]
    rows = build_augmentation_statistics_rows("Alpha", original_rows, generated_rows)
    assert set(rows[0].keys()) == set(AUGMENTATION_STATISTICS_COLUMNS)

    by_type = {r["augmentation_type"]: r for r in rows}
    assert by_type["rotation"]["augmentation_count"] == 2
    assert by_type["horizontal_flip"]["augmentation_count"] == 1
    assert by_type["rotation"]["original_count"] == 2
    assert by_type["rotation"]["generated_count"] == 3
    assert by_type["rotation"]["total_count"] == 5
    assert by_type["rotation"]["unique_parent_count"] == 2
    assert by_type["rotation"]["max_generated_per_parent"] == 2


def test_build_augmentation_statistics_rows_zero_generated():
    original_rows = [_original_row("Alpha/a0.jpg", "Alpha", "id0")]
    rows = build_augmentation_statistics_rows("Alpha", original_rows, [])
    assert len(rows) == 1
    assert rows[0]["generated_count"] == 0
    assert rows[0]["augmentation_type"] == ""


# --- full pipeline integration ---


def test_full_pipeline_produces_exact_target_per_class(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(4):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(2):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, summary = _run_full_pipeline(
        tmp_path, mapping, augmentation={"target_samples_per_class": 8}
    )

    rows = _read_manifest(manifests_dir / "train_balanced.csv")
    counts = {}
    for r in rows:
        counts[r["class"]] = counts.get(r["class"], 0) + 1
    assert counts == {"Alpha": 8, "Beta": 8}
    assert summary.total_samples == 16


def test_full_pipeline_original_rows_are_retained_exactly(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(3):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(3):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, _ = _run_full_pipeline(
        tmp_path, mapping, augmentation={"target_samples_per_class": 12}
    )

    train_original_rows = _read_manifest(manifests_dir / "train_original.csv")
    balanced_rows = _read_manifest(manifests_dir / "train_balanced.csv")
    balanced_ids = {r["original_id"] for r in balanced_rows if r["is_original"] == "true"}
    for row in train_original_rows:
        assert row["original_id"] in balanced_ids


def test_full_pipeline_validation_test_never_used_as_parent(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(20):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(20):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, summary = _run_full_pipeline(
        tmp_path, mapping, augmentation={"target_samples_per_class": 30}
    )

    val_rows = _read_manifest(manifests_dir / "val_original.csv")
    test_rows = _read_manifest(manifests_dir / "test_original.csv")
    val_test_ids = {r["original_id"] for r in val_rows} | {r["original_id"] for r in test_rows}
    val_test_paths = {r["path"] for r in val_rows} | {r["path"] for r in test_rows}

    balanced_rows = _read_manifest(manifests_dir / "train_balanced.csv")
    parent_ids_used = {r["parent_original_id"] for r in balanced_rows if r["is_original"] == "false"}
    balanced_paths = {r["path"] for r in balanced_rows}

    assert not (parent_ids_used & val_test_ids)
    assert not (balanced_paths & val_test_paths)


def test_full_pipeline_small_class_expansion(tmp_path):
    raw_dir = tmp_path / "raw"
    make_image(raw_dir / "Tiny" / "t0.jpg")
    make_image(raw_dir / "Tiny" / "t1.jpg")
    for i in range(10):
        make_image(raw_dir / "Big" / f"b{i}.jpg")
    mapping = {"Tiny": "Tiny", "Big": "Big"}

    _, audit_dir, manifests_dir, summary = _run_full_pipeline(
        tmp_path, mapping, augmentation={"target_samples_per_class": 50}
    )

    tiny_total = summary.per_class_counts["Tiny"]["total"]
    assert tiny_total == 50
    assert summary.per_class_counts["Tiny"]["generated"] == 50 - summary.per_class_counts["Tiny"]["original"]


def test_full_pipeline_raw_never_written_to(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(5):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(5):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    files_before = sorted(p.name for p in (raw_dir / "Alpha").iterdir())

    _run_full_pipeline(tmp_path, mapping, augmentation={"target_samples_per_class": 20})

    files_after = sorted(p.name for p in (raw_dir / "Alpha").iterdir())
    assert files_before == files_after  # nothing added/removed/renamed in raw_dir


def test_full_pipeline_generated_files_only_under_processed_train(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(3):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(3):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, _ = _run_full_pipeline(
        tmp_path, mapping, augmentation={"target_samples_per_class": 10}
    )

    balanced_rows = _read_manifest(manifests_dir / "train_balanced.csv")
    processed_train_dir = tmp_path / "processed" / "train"
    for row in balanced_rows:
        if row["is_original"] == "false":
            assert (processed_train_dir / row["path"]).is_file()


def test_full_pipeline_deterministic_same_seed(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(5):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(5):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, _ = _run_full_pipeline(
        tmp_path, mapping, seed=42, augmentation={"target_samples_per_class": 15}
    )
    from src.utils.hashing import hash_file

    hash_first = hash_file(manifests_dir / "train_balanced.csv")

    config_path = tmp_path / "dataset.yaml"
    run_generate_balanced_dataset(config_path=config_path, seed=42)
    hash_second = hash_file(manifests_dir / "train_balanced.csv")

    assert hash_first == hash_second


def test_full_pipeline_different_seed_may_differ(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(5):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(5):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, _ = _run_full_pipeline(
        tmp_path, mapping, seed=42, augmentation={"target_samples_per_class": 15}
    )
    rows_42 = {r["original_id"] for r in _read_manifest(manifests_dir / "train_balanced.csv")}

    config_path = tmp_path / "dataset.yaml"
    run_generate_balanced_dataset(config_path=config_path, seed=123)
    rows_123 = {r["original_id"] for r in _read_manifest(manifests_dir / "train_balanced.csv")}

    assert rows_42 != rows_123


def test_full_pipeline_writes_hash_and_metadata(tmp_path):
    raw_dir = tmp_path / "raw"
    for i in range(5):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(5):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}

    _, audit_dir, manifests_dir, summary = _run_full_pipeline(
        tmp_path, mapping, augmentation={"target_samples_per_class": 12}
    )

    from src.utils.hashing import hash_file

    hash_path = audit_dir / "train_balanced_manifest_hash.txt"
    assert hash_path.exists()
    assert hash_path.read_text(encoding="utf-8").strip() == hash_file(manifests_dir / "train_balanced.csv")

    assert (audit_dir / "balanced_dataset_metadata.json").exists()
    assert (audit_dir / "augmentation_statistics.csv").exists()

    import json

    metadata = json.loads((audit_dir / "balanced_dataset_metadata.json").read_text(encoding="utf-8"))
    assert metadata["seed"] == 42
    assert metadata["target_samples_per_class"] == 12
    assert metadata["total_samples"] == 24
    assert "git_commit" in metadata
    assert "source_manifest_hash" in metadata
    # No absolute machine paths in the metadata.
    assert str(tmp_path) not in json.dumps(metadata)


def test_full_pipeline_raises_without_prior_split(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Beta" / "b1.jpg")
    config_path = write_min_dataset_config(tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir)

    with pytest.raises(BalancedDatasetBuildError, match="train_original.csv"):
        run_generate_balanced_dataset(config_path=config_path)


# --- manifest validation ---


def _valid_balanced_context():
    original_rows = [
        _original_row("Alpha/a0.jpg", "Alpha", "id0"),
        _original_row("Alpha/a1.jpg", "Alpha", "id1"),
    ]
    generated_row = {
        "path": "Alpha/gen0.jpg",
        "class": "Alpha",
        "split": "train",
        "original_id": "gen_id0",
        "parent_original_id": "id0",
        "is_original": "false",
        "augmentation_type": "rotation",
    }
    balanced_rows = original_rows + [generated_row]
    return balanced_rows, original_rows


def test_validate_balanced_manifest_detects_wrong_total_count():
    balanced_rows, original_rows = _valid_balanced_context()
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=5)
    by_check = {r.check: r for r in results}
    assert by_check["total_row_count"].status == "FAIL"


def test_validate_balanced_manifest_detects_wrong_per_class_count():
    balanced_rows, original_rows = _valid_balanced_context()
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=3)
    by_check = {r.check: r for r in results}
    assert by_check["per_class_exact_target"].status == "PASS"  # 3 rows == target 3
    results2 = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=10)
    by_check2 = {r.check: r for r in results2}
    assert by_check2["per_class_exact_target"].status == "FAIL"


def test_validate_balanced_manifest_detects_validation_image_present():
    balanced_rows, original_rows = _valid_balanced_context()
    val_rows = [_original_row("Alpha/a0.jpg", "Alpha", "id0")]  # same id as a train row
    results = validate_balanced_manifest(balanced_rows, original_rows, val_rows, [], [], ["Alpha"], target_per_class=3)
    by_check = {r.check: r for r in results}
    assert by_check["no_validation_images"].status == "FAIL"


def test_validate_balanced_manifest_detects_test_image_present():
    balanced_rows, original_rows = _valid_balanced_context()
    test_rows = [_original_row("Alpha/a1.jpg", "Alpha", "id1")]
    results = validate_balanced_manifest(balanced_rows, original_rows, [], test_rows, [], ["Alpha"], target_per_class=3)
    by_check = {r.check: r for r in results}
    assert by_check["no_test_images"].status == "FAIL"


def test_validate_balanced_manifest_detects_missing_parent():
    balanced_rows, original_rows = _valid_balanced_context()
    balanced_rows[-1] = dict(balanced_rows[-1], parent_original_id="")
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=3)
    by_check = {r.check: r for r in results}
    assert by_check["generated_has_parent"].status == "FAIL"


def test_validate_balanced_manifest_detects_recursive_augmentation():
    balanced_rows, original_rows = _valid_balanced_context()
    # Point the generated row's parent at ANOTHER generated row's id (not in train_original_rows).
    balanced_rows.append(
        {
            "path": "Alpha/gen1.jpg",
            "class": "Alpha",
            "split": "train",
            "original_id": "gen_id1",
            "parent_original_id": "gen_id0",  # a generated id, not an original one
            "is_original": "false",
            "augmentation_type": "rotation",
        }
    )
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=4)
    by_check = {r.check: r for r in results}
    assert by_check["parent_is_original_training_image"].status == "FAIL"


def test_validate_balanced_manifest_detects_class_mismatch():
    balanced_rows, original_rows = _valid_balanced_context()
    balanced_rows[-1] = dict(balanced_rows[-1], **{"class": "Beta"})
    results = validate_balanced_manifest(
        balanced_rows, original_rows, [], [], [], ["Alpha", "Beta"], target_per_class=3
    )
    by_check = {r.check: r for r in results}
    assert by_check["generated_class_matches_parent"].status == "FAIL"


def test_validate_balanced_manifest_detects_wrong_split():
    balanced_rows, original_rows = _valid_balanced_context()
    balanced_rows[-1] = dict(balanced_rows[-1], split="val")
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=3)
    by_check = {r.check: r for r in results}
    assert by_check["split_is_train"].status == "FAIL"


def test_validate_balanced_manifest_detects_missing_original():
    balanced_rows, original_rows = _valid_balanced_context()
    balanced_rows = [r for r in balanced_rows if r["original_id"] != "id1"]  # drop an original
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=2)
    by_check = {r.check: r for r in results}
    assert by_check["all_originals_retained"].status == "FAIL"


def test_validate_balanced_manifest_detects_duplicate_ids():
    balanced_rows, original_rows = _valid_balanced_context()
    balanced_rows.append(dict(balanced_rows[-1]))  # duplicate the generated row's id
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=4)
    by_check = {r.check: r for r in results}
    assert by_check["no_duplicate_ids"].status == "FAIL"


def test_validate_balanced_manifest_detects_is_original_flag_mismatch():
    balanced_rows, original_rows = _valid_balanced_context()
    balanced_rows[-1] = dict(balanced_rows[-1], is_original="true")  # generated but flagged original
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=3)
    by_check = {r.check: r for r in results}
    assert by_check["is_original_flag_consistency"].status == "FAIL"


def test_validate_balanced_manifest_passes_for_clean_context():
    balanced_rows, original_rows = _valid_balanced_context()
    results = validate_balanced_manifest(balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=3)
    assert all(r.status == "PASS" for r in results)


def test_validate_balanced_manifest_detects_missing_generated_file(tmp_path):
    balanced_rows, original_rows = _valid_balanced_context()
    processed_train_dir = tmp_path / "processed" / "train"  # no file written here
    results = validate_balanced_manifest(
        balanced_rows, original_rows, [], [], [], ["Alpha"], target_per_class=3,
        processed_train_dir=processed_train_dir,
    )
    by_check = {r.check: r for r in results}
    assert by_check["generated_files_exist"].status == "FAIL"


def test_run_generate_balanced_dataset_raises_when_target_below_original_count(tmp_path):
    # A target smaller than the number of original training images for a
    # class must be rejected at the orchestration layer, not silently
    # accepted by dropping originals.
    raw_dir = tmp_path / "raw"
    for i in range(20):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(20):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    mapping = {"Alpha": "Alpha", "Beta": "Beta"}
    audit_dir = tmp_path / "audit"
    config_path = write_min_dataset_config(
        tmp_path, mapping, raw_dir, audit_dir, augmentation={"target_samples_per_class": 1}
    )
    try:
        run_dataset_audit(config_path=config_path)
    except AuditFailedError:
        pass
    run_build_split(config_path=config_path, seed=42)

    with pytest.raises(BalancedDatasetBuildError, match="already has"):
        run_generate_balanced_dataset(config_path=config_path, seed=42)

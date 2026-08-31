"""Integration tests for src.evaluation.final_test.run_final_test (Task 8).

CRITICAL: none of these tests ever touch data/manifests/test_original.csv
or data/raw/ - every fixture builds its own tiny, self-contained synthetic
dataset/checkpoint under tmp_path, exactly like
tests/test_run_baseline.py/test_run_aa_evidentnet.py already do for
training. Uses real create_model()/build_checkpoint()/save_checkpoint()
(pretrained=False, fast/offline) since there is no lightweight substitute
for "the actual final_test orchestration works end to end".
"""

import json

import pytest
import torch

from src.data.dataset import DatasetManifestError
from src.data.records import write_csv
from src.evaluation.final_test import FinalTestError, run_final_test
from src.models.factory import create_model
from src.training.checkpointing import CheckpointIncompatibleError, build_checkpoint, save_checkpoint
from src.training.registry import load_registry, register_run
from src.utils.hashing import hash_file
from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

CANONICAL_CLASSES = ["Alpha", "Beta", "Delta", "Gamma"]  # already alphabetical - matches build_class_to_idx's ordering
MANIFEST_COLUMNS = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]


def _write_manifest(manifest_path, rows):
    write_csv(rows, MANIFEST_COLUMNS, manifest_path)


def _make_test_manifest_rows(raw_dir, canonical_classes, n_per_class=2, split="test", is_original="true"):
    rows = []
    for class_name in canonical_classes:
        for i in range(n_per_class):
            filename = f"{class_name}_{i}.jpg"
            rel_path = f"{class_name}/{filename}"
            make_image(raw_dir / rel_path)
            sample_id = f"sample_{class_name}_{i}"
            rows.append(
                {
                    "path": rel_path,
                    "class": class_name,
                    "split": split,
                    "original_id": sample_id,
                    "parent_original_id": sample_id,
                    "is_original": is_original,
                    "augmentation_type": "original" if is_original == "true" else "combined",
                }
            )
    return rows


def _setup(tmp_path, model_name="resnet50", num_classes_override=None, n_per_class=2, manifest_rows=None):
    canonical_classes = CANONICAL_CLASSES[: num_classes_override or len(CANONICAL_CLASSES)]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = manifest_rows if manifest_rows is not None else _make_test_manifest_rows(raw_dir, canonical_classes, n_per_class)
    _write_manifest(manifests_dir / "test_original.csv", rows)

    include_aa = model_name == "aa_evidentnet"
    models_cfg = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=include_aa)

    return dataset_cfg, models_cfg, canonical_classes, manifests_dir


def _build_checkpoint(tmp_path, model_name, models_cfg, num_classes, run_id="20260101_000000_testrun_seed42_abcdef"):
    import yaml

    models_config = yaml.safe_load(models_cfg.read_text(encoding="utf-8"))
    model = create_model(model_name, models_config)
    checkpoint = build_checkpoint(
        model=model,
        optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3),
        scheduler=None,
        epoch=7,
        best_metric=0.85,
        monitor_metric="val_macro_f1",
        training_config={},
        seed=42,
        model_name=model_name,
        architecture=model.architecture,
        num_classes=num_classes,
        dataset_manifest_hash="deadbeef",
        git_commit="abc123",
    )
    checkpoint_dir = tmp_path / "checkpoints" / run_id
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)
    return checkpoint_path


# --- end-to-end: resnet50 ---


def test_run_final_test_end_to_end_resnet50(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=2)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    registry_path = tmp_path / "registry.csv"

    summary = run_final_test(
        model_name="resnet50",
        checkpoint_path=checkpoint_path,
        dataset_config_path=dataset_cfg,
        models_config_path=models_cfg,
        seed=42,
        device_override="cpu",
        num_workers_override=0,
        registry_path=registry_path,
    )

    assert summary.model_name == "resnet50"
    assert summary.num_samples == len(canonical_classes) * 2
    assert summary.class_names == canonical_classes
    assert summary.registry_updated is False  # no matching row was ever registered

    from pathlib import Path

    assert Path(summary.predictions_path).is_file()
    assert Path(summary.overall_metrics_path).is_file()
    assert Path(summary.per_class_metrics_path).is_file()
    assert Path(summary.confusion_matrix_path).is_file()
    assert Path(summary.metadata_path).is_file()


def test_predictions_csv_row_count_matches_manifest_count(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=3)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    import csv as csv_module

    with open(summary.predictions_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == len(canonical_classes) * 3 == summary.num_samples


def test_predictions_csv_schema_and_sample_ids_unique(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=2)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    import csv as csv_module

    with open(summary.predictions_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))

    base_fields = {
        "sample_id", "image_path", "true_class_index", "true_class_name",
        "predicted_class_index", "predicted_class_name", "correct", "max_probability",
    }
    assert base_fields.issubset(rows[0].keys())
    for k in range(len(canonical_classes)):
        assert f"logit_{k}" in rows[0]
        assert f"prob_{k}" in rows[0]
    # No AA-EvidentNet-only columns for a baseline.
    assert "evidence_0" not in rows[0]
    assert "uncertainty" not in rows[0]

    sample_ids = [r["sample_id"] for r in rows]
    assert len(set(sample_ids)) == len(sample_ids)

    for row in rows:
        expected_correct = str(row["true_class_index"] == row["predicted_class_index"])
        assert row["correct"] == expected_correct


def test_probabilities_sum_to_one_and_logits_present_for_every_class(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )
    import csv as csv_module

    with open(summary.predictions_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    for row in rows:
        total = sum(float(row[f"prob_{k}"]) for k in range(len(canonical_classes)))
        assert total == pytest.approx(1.0, abs=1e-4)
        max_prob = max(float(row[f"prob_{k}"]) for k in range(len(canonical_classes)))
        assert float(row["max_probability"]) == pytest.approx(max_prob, abs=1e-6)


# --- AA-EvidentNet evidential export ---


def test_aa_evidentnet_evidential_columns_and_dimensions(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "aa_evidentnet", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "aa_evidentnet", models_cfg, len(canonical_classes))

    summary = run_final_test(
        model_name="aa_evidentnet", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    import csv as csv_module

    with open(summary.predictions_path, newline="", encoding="utf-8") as f:
        rows = list(csv_module.DictReader(f))
    assert len(rows) == len(canonical_classes)

    num_classes = len(canonical_classes)
    for row in rows:
        evidence = [float(row[f"evidence_{k}"]) for k in range(num_classes)]
        alpha = [float(row[f"dirichlet_alpha_{k}"]) for k in range(num_classes)]
        evidential_probs = [float(row[f"evidential_prob_{k}"]) for k in range(num_classes)]
        uncertainty = float(row["uncertainty"])

        assert all(e >= 0 for e in evidence)
        assert all(a == pytest.approx(e + 1.0) for a, e in zip(alpha, evidence))
        assert sum(evidential_probs) == pytest.approx(1.0, abs=1e-4)
        strength = sum(alpha)
        assert uncertainty == pytest.approx(num_classes / strength, abs=1e-4)
        assert 0.0 < uncertainty <= 1.0


# --- overall/per-class/confusion-matrix outputs ---


def test_overall_and_per_class_metrics_files_are_well_formed(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=2)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    with open(summary.overall_metrics_path, encoding="utf-8") as f:
        overall = json.load(f)
    for key in ("accuracy", "balanced_accuracy", "macro_precision", "macro_recall", "macro_f1", "macro_roc_auc", "macro_pr_auc"):
        assert key in overall
    assert overall["num_samples"] == summary.num_samples

    import csv as csv_module

    with open(summary.per_class_metrics_path, newline="", encoding="utf-8") as f:
        per_class_rows = list(csv_module.DictReader(f))
    assert len(per_class_rows) == len(canonical_classes)
    assert [r["class_name"] for r in per_class_rows] == canonical_classes

    with open(summary.confusion_matrix_path, newline="", encoding="utf-8") as f:
        cm_rows = list(csv_module.DictReader(f))
    assert len(cm_rows) == len(canonical_classes)
    for row in cm_rows:
        row_total = sum(int(row[f"pred_{c}"]) for c in canonical_classes)
        assert row_total == 2  # n_per_class=2 for this true class


# --- reproducibility metadata ---


def test_metadata_includes_checkpoint_and_manifest_hashes(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    expected_checkpoint_hash = hash_file(checkpoint_path)
    expected_manifest_hash = hash_file(manifests_dir / "test_original.csv")

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, seed=42, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    with open(summary.metadata_path, encoding="utf-8") as f:
        metadata = json.load(f)

    assert metadata["checkpoint_sha256"] == expected_checkpoint_hash == summary.checkpoint_hash
    assert metadata["test_manifest_sha256"] == expected_manifest_hash
    assert metadata["class_names"] == canonical_classes
    assert metadata["model_name"] == "resnet50"
    assert metadata["checkpoint_best_epoch"] == 7
    assert metadata["checkpoint_best_metric"] == 0.85
    assert metadata["num_evaluated_samples"] == summary.num_samples
    assert "git_commit" in metadata
    assert "timestamp_utc" in metadata
    assert metadata["device"] == "cpu"


def test_checkpoint_file_is_never_modified(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    hash_before = hash_file(checkpoint_path)

    run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    assert hash_file(checkpoint_path) == hash_before


# --- eval-mode / no-grad / no-training guarantees ---


def test_evaluation_calls_model_eval(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    calls = []
    original_eval = torch.nn.Module.eval

    def spy_eval(self):
        calls.append(self)
        return original_eval(self)

    monkeypatch.setattr(torch.nn.Module, "eval", spy_eval)
    run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )
    assert len(calls) >= 1


def test_evaluation_never_constructs_an_optimizer(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    def forbidden_init(self, *a, **kw):
        raise AssertionError("no optimizer should ever be constructed during final_test")

    monkeypatch.setattr(torch.optim.AdamW, "__init__", forbidden_init)
    # Must not raise - proves AdamW.__init__ is never reached.
    run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )


def test_evaluation_never_calls_backward(tmp_path, monkeypatch):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    def forbidden_backward(self, *a, **kw):
        raise AssertionError("no backward pass should ever occur during final_test")

    monkeypatch.setattr(torch.Tensor, "backward", forbidden_backward)
    run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )


# --- test-manifest safeguards ---


def test_rejects_manifest_with_non_test_split(tmp_path):
    raw_dir = tmp_path / "raw"
    bad_rows = _make_test_manifest_rows(raw_dir, CANONICAL_CLASSES[:2], n_per_class=1, split="train")
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(
        tmp_path, "resnet50", num_classes_override=2, manifest_rows=bad_rows
    )
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, 2)

    with pytest.raises(DatasetManifestError, match="split"):
        run_final_test(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=tmp_path / "registry.csv",
        )


def test_rejects_manifest_with_augmented_sample(tmp_path):
    raw_dir = tmp_path / "raw"
    bad_rows = _make_test_manifest_rows(raw_dir, CANONICAL_CLASSES[:2], n_per_class=1, is_original="false")
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(
        tmp_path, "resnet50", num_classes_override=2, manifest_rows=bad_rows
    )
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, 2)

    with pytest.raises(DatasetManifestError):
        run_final_test(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=tmp_path / "registry.csv",
        )


def test_missing_test_manifest_raises_final_test_error(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in CANONICAL_CLASSES[:2]}
    dataset_cfg = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)  # no test_original.csv written
    models_cfg = write_min_models_config(tmp_path, num_classes=2)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, 2)

    with pytest.raises(FinalTestError, match="test_original.csv"):
        run_final_test(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=tmp_path / "registry.csv",
        )


def test_unknown_model_name_raises_final_test_error(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    with pytest.raises(FinalTestError, match="Unknown model"):
        run_final_test(
            model_name="not_a_real_model", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=tmp_path / "registry.csv",
        )


# --- checkpoint/model compatibility ---


def test_incompatible_checkpoint_model_name_rejected(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    with pytest.raises(CheckpointIncompatibleError):
        run_final_test(
            model_name="efficientnetb0", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=tmp_path / "registry.csv",
        )


def test_incompatible_checkpoint_num_classes_rejected(tmp_path):
    # Checkpoint trained for 4 classes; dataset config here only has 2.
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, num_classes=99)  # deliberately wrong

    with pytest.raises(CheckpointIncompatibleError, match="num_classes"):
        run_final_test(
            model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=tmp_path / "registry.csv",
        )


# --- registry behavior ---


def test_registry_updated_when_training_run_id_matches_checkpoint_directory(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    run_id = "20260101_000000_testrun_seed42_abcdef"
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes), run_id=run_id)
    registry_path = tmp_path / "registry.csv"
    register_run(experiment_id=run_id, model="resnet50", seed=42, config="configs/training.yaml", registry_path=registry_path)

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=registry_path,
    )

    assert summary.registry_updated is True
    rows = load_registry(registry_path)
    matching = [r for r in rows if r["experiment_id"] == run_id]
    assert len(matching) == 1
    assert matching[0]["test_result"] != ""
    assert "macro_f1=" in matching[0]["test_result"]


def test_registry_not_updated_and_no_new_row_when_run_id_not_found(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes), run_id="nonexistent_run_id")
    registry_path = tmp_path / "registry.csv"
    register_run(experiment_id="some_other_run", model="resnet50", seed=42, config="c", registry_path=registry_path)

    summary = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=registry_path,
    )

    assert summary.registry_updated is False
    rows = load_registry(registry_path)
    assert len(rows) == 1  # no new row was created
    assert rows[0]["experiment_id"] == "some_other_run"
    assert rows[0]["test_result"] == ""


def test_registry_untouched_when_evaluation_fails(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))
    registry_path = tmp_path / "registry.csv"
    register_run(experiment_id=checkpoint_path.parent.name, model="resnet50", seed=42, config="c", registry_path=registry_path)
    before = registry_path.read_text(encoding="utf-8")

    with pytest.raises(CheckpointIncompatibleError):
        run_final_test(
            model_name="efficientnetb0",  # wrong model -> fails before reaching the registry update
            checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
            models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
            registry_path=registry_path,
        )

    after = registry_path.read_text(encoding="utf-8")
    assert before == after


# --- outputs do not collide across separate runs ---


def test_two_invocations_get_distinct_non_overwriting_output_directories(tmp_path):
    dataset_cfg, models_cfg, canonical_classes, manifests_dir = _setup(tmp_path, "resnet50", n_per_class=1)
    checkpoint_path = _build_checkpoint(tmp_path, "resnet50", models_cfg, len(canonical_classes))

    summary_a = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )
    summary_b = run_final_test(
        model_name="resnet50", checkpoint_path=checkpoint_path, dataset_config_path=dataset_cfg,
        models_config_path=models_cfg, device_override="cpu", num_workers_override=0,
        registry_path=tmp_path / "registry.csv",
    )

    from pathlib import Path

    assert summary_a.eval_run_id != summary_b.eval_run_id
    assert summary_a.predictions_path != summary_b.predictions_path
    assert Path(summary_a.predictions_path).is_file()
    assert Path(summary_b.predictions_path).is_file()

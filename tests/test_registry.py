"""Tests for src.training.registry: experiments/registry.csv, using a
tmp_path registry file so the real experiments/registry.csv is never
touched by pytest."""

import csv

import pytest

from src.training.registry import (
    REGISTRY_COLUMNS,
    RegistryError,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    load_registry,
    register_run,
    update_run,
)


def test_load_registry_returns_empty_list_when_missing(tmp_path):
    assert load_registry(tmp_path / "registry.csv") == []


def test_register_run_creates_file_with_correct_schema(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "configs/training.yaml", registry_path=registry_path)

    with open(registry_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == REGISTRY_COLUMNS
        rows = list(reader)
    assert len(rows) == 1
    assert rows[0]["experiment_id"] == "exp1"
    assert rows[0]["status"] == STATUS_RUNNING
    assert rows[0]["test_result"] == ""


def test_register_run_duplicate_id_raises(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "cfg.yaml", registry_path=registry_path)
    with pytest.raises(RegistryError, match="already registered"):
        register_run("exp1", "efficientnetb0", 123, "cfg.yaml", registry_path=registry_path)


def test_register_run_rejects_invalid_status(tmp_path):
    registry_path = tmp_path / "registry.csv"
    with pytest.raises(RegistryError, match="Unknown status"):
        register_run("exp1", "resnet50", 42, "cfg.yaml", status="bogus", registry_path=registry_path)


def test_update_run_sets_completed_status(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "cfg.yaml", registry_path=registry_path)
    update_run("exp1", registry_path=registry_path, status=STATUS_COMPLETED, checkpoint="ckpt.pt")

    rows = load_registry(registry_path)
    assert rows[0]["status"] == STATUS_COMPLETED
    assert rows[0]["checkpoint"] == "ckpt.pt"


def test_update_run_sets_failed_status_with_notes(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "cfg.yaml", registry_path=registry_path)
    update_run("exp1", registry_path=registry_path, status=STATUS_FAILED, notes="CUDA OOM")

    rows = load_registry(registry_path)
    assert rows[0]["status"] == STATUS_FAILED
    assert rows[0]["notes"] == "CUDA OOM"


def test_update_run_unknown_id_raises(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "cfg.yaml", registry_path=registry_path)
    with pytest.raises(RegistryError, match="not found"):
        update_run("does_not_exist", registry_path=registry_path, status=STATUS_COMPLETED)


def test_update_run_does_not_clobber_other_rows(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "cfg.yaml", registry_path=registry_path)
    register_run("exp2", "efficientnetb0", 123, "cfg.yaml", registry_path=registry_path)

    update_run("exp1", registry_path=registry_path, status=STATUS_COMPLETED)

    rows = {r["experiment_id"]: r for r in load_registry(registry_path)}
    assert rows["exp1"]["status"] == STATUS_COMPLETED
    assert rows["exp2"]["status"] == STATUS_RUNNING  # untouched
    assert rows["exp2"]["model"] == "efficientnetb0"


def test_test_result_stays_empty_unless_explicitly_set(tmp_path):
    registry_path = tmp_path / "registry.csv"
    register_run("exp1", "resnet50", 42, "cfg.yaml", registry_path=registry_path)
    update_run("exp1", registry_path=registry_path, status=STATUS_COMPLETED)
    rows = load_registry(registry_path)
    assert rows[0]["test_result"] == ""

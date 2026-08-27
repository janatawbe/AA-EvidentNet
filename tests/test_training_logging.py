"""Tests for src.training.logging: run IDs and per-run RunLogger."""

import json

import pytest

from src.training.logging import (
    CONFIG_FILENAME,
    DATASET_HASH_FILENAME,
    ENVIRONMENT_FILENAME,
    GIT_COMMIT_FILENAME,
    METRICS_FILENAME,
    RUN_LOG_FILENAME,
    RunLogger,
    generate_run_id,
)


def test_generate_run_id_contains_model_and_seed():
    run_id = generate_run_id("resnet50", 42)
    assert "resnet50" in run_id
    assert "seed42" in run_id


def test_generate_run_id_smoke_tag():
    run_id = generate_run_id("resnet50", 42, smoke_test=True)
    assert "smoke" in run_id


def test_generate_run_id_unique_even_within_same_second():
    ids = {generate_run_id("resnet50", 42) for _ in range(20)}
    assert len(ids) == 20  # no collisions from the random suffix


def test_run_logger_creates_directory_and_all_files(tmp_path):
    run_dir = tmp_path / "run1"
    logger = RunLogger(run_dir)
    logger.log("hello")
    logger.log_metrics({"epoch": 0, "loss": 1.0})
    logger.write_config({"a": 1})
    logger.write_environment({"python": "3.10"})
    logger.write_dataset_hash("abc123")
    logger.write_git_commit("deadbeef")
    logger.close()

    for filename in (RUN_LOG_FILENAME, METRICS_FILENAME, CONFIG_FILENAME, ENVIRONMENT_FILENAME, DATASET_HASH_FILENAME, GIT_COMMIT_FILENAME):
        assert (run_dir / filename).is_file(), f"missing {filename}"


def test_run_logger_refuses_to_overwrite_existing_dir(tmp_path):
    run_dir = tmp_path / "run1"
    RunLogger(run_dir).close()
    with pytest.raises(FileExistsError):
        RunLogger(run_dir)


def test_run_logger_log_writes_timestamped_line(tmp_path):
    run_dir = tmp_path / "run1"
    logger = RunLogger(run_dir)
    logger.log("a specific message", print_to_stdout=False)
    logger.close()
    content = (run_dir / RUN_LOG_FILENAME).read_text(encoding="utf-8")
    assert "a specific message" in content


def test_run_logger_metrics_jsonl_one_record_per_line(tmp_path):
    run_dir = tmp_path / "run1"
    logger = RunLogger(run_dir)
    logger.log_metrics({"epoch": 0, "loss": 1.0})
    logger.log_metrics({"epoch": 1, "loss": 0.5})
    logger.close()

    lines = (run_dir / METRICS_FILENAME).read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    records = [json.loads(line) for line in lines]
    assert records[0]["epoch"] == 0
    assert records[1]["loss"] == 0.5


def test_run_logger_write_config_roundtrips_as_yaml(tmp_path):
    import yaml

    run_dir = tmp_path / "run1"
    logger = RunLogger(run_dir)
    logger.write_config({"seed": 42, "nested": {"lr": 0.001}})
    logger.close()
    loaded = yaml.safe_load((run_dir / CONFIG_FILENAME).read_text(encoding="utf-8"))
    assert loaded == {"seed": 42, "nested": {"lr": 0.001}}


def test_run_logger_context_manager_closes_files(tmp_path):
    run_dir = tmp_path / "run1"
    with RunLogger(run_dir) as logger:
        logger.log("inside context")
    # File should be readable/closed cleanly after exiting the context.
    assert "inside context" in (run_dir / RUN_LOG_FILENAME).read_text(encoding="utf-8")

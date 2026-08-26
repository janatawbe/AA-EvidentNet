"""Tests for run_pipeline.py CLI parsing and dispatch behavior.

At this stage of the project, every command is recognized by the parser but
raises a clear "not implemented" error when dispatched. These tests check
that: (1) the parser accepts all required commands and options, (2) invalid
usage is rejected, and (3) dispatch fails loudly and informatively rather
than silently succeeding.
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import run_pipeline  # noqa: E402
from tests.conftest import make_image, make_invalid_image, write_min_dataset_config  # noqa: E402

# "audit" and "prepare_dataset" are implemented (see test_main_audit_command_*
# and test_main_prepare_dataset_command_* below); every other command is
# still recognized by the parser but not yet implemented.
ALL_COMMANDS = [
    "audit",
    "prepare_dataset",
    "ablation",
    "hard_pairs",
    "calibration",
    "selective",
    "gradcam",
    "robustness",
    "multi_seed",
    "publication",
    "final_test",
]

IMPLEMENTED_COMMANDS = {"audit", "prepare_dataset"}
UNIMPLEMENTED_COMMANDS = [c for c in ALL_COMMANDS if c not in IMPLEMENTED_COMMANDS]

MODEL_COMMANDS = ["baseline", "train"]


@pytest.mark.parametrize("command", ALL_COMMANDS)
def test_parser_accepts_bare_command(command):
    parser = run_pipeline.build_parser()
    args = parser.parse_args([command])
    assert args.command == command
    assert args.seed == run_pipeline.DEFAULT_SEED


@pytest.mark.parametrize("command", MODEL_COMMANDS)
def test_parser_requires_model_for_train_and_baseline(command):
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([command])  # missing required --model


@pytest.mark.parametrize("command", MODEL_COMMANDS)
def test_parser_accepts_model_argument(command):
    parser = run_pipeline.build_parser()
    args = parser.parse_args([command, "--model", "maxvit"])
    assert args.model == "maxvit"


def test_parser_common_options():
    parser = run_pipeline.build_parser()
    args = parser.parse_args(
        [
            "audit",
            "--config",
            "configs/dataset.yaml",
            "--seed",
            "123",
            "--device",
            "cpu",
            "--batch-size",
            "32",
            "--epochs",
            "5",
            "--num-workers",
            "2",
            "--smoke-test",
        ]
    )
    assert args.config == "configs/dataset.yaml"
    assert args.seed == 123
    assert args.device == "cpu"
    assert args.batch_size == 32
    assert args.epochs == 5
    assert args.num_workers == 2
    assert args.smoke_test is True


def test_parser_rejects_unknown_command():
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["not_a_real_command"])


def test_parser_rejects_invalid_device():
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["audit", "--device", "tpu"])


def test_parser_requires_a_command():
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


@pytest.mark.parametrize("command", UNIMPLEMENTED_COMMANDS)
def test_main_fails_clearly_for_unimplemented_commands(command, capsys):
    exit_code = run_pipeline.main([command])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT IMPLEMENTED" in captured.err
    assert command in captured.err


@pytest.mark.parametrize("command", MODEL_COMMANDS)
def test_main_fails_clearly_for_unimplemented_model_commands(command, capsys):
    exit_code = run_pipeline.main([command, "--model", "maxvit"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT IMPLEMENTED" in captured.err


# --- "audit" is implemented: exercise it end-to-end against a tiny fixture
# dataset (never the real data/raw/) via an explicit --config override. ---


def test_main_audit_command_passes_on_clean_fixture(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Beta" / "b1.jpg")
    config_path = write_min_dataset_config(
        tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir
    )

    exit_code = run_pipeline.main(["audit", "--config", str(config_path)])

    assert exit_code == 0
    assert (audit_dir / "dataset_inventory.csv").exists()


def test_main_audit_command_fails_clearly_on_policy_violation(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_invalid_image(raw_dir / "Alpha" / "broken.jpg")
    make_image(raw_dir / "Beta" / "b1.jpg")
    config_path = write_min_dataset_config(
        tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir
    )

    exit_code = run_pipeline.main(["audit", "--config", str(config_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "AUDIT FAILED" in captured.err


# --- "prepare_dataset" is implemented: exercise it end-to-end against a tiny
# fixture dataset (never the real data/raw/) via an explicit --config
# override, always after first running "audit" to produce the eligibility
# manifest it depends on. ---


def test_main_prepare_dataset_command_builds_split_after_audit(tmp_path):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    manifests_dir = tmp_path / "manifests"
    for i in range(10):
        make_image(raw_dir / "Alpha" / f"a{i}.jpg")
    for i in range(10):
        make_image(raw_dir / "Beta" / f"b{i}.jpg")
    config_path = write_min_dataset_config(
        tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir
    )

    assert run_pipeline.main(["audit", "--config", str(config_path)]) == 0

    exit_code = run_pipeline.main(["prepare_dataset", "--config", str(config_path)])

    assert exit_code == 0
    assert (manifests_dir / "train_original.csv").exists()
    assert (manifests_dir / "val_original.csv").exists()
    assert (manifests_dir / "test_original.csv").exists()
    assert (audit_dir / "split_leakage_report.csv").exists()
    assert (audit_dir / "split_metadata.json").exists()


def test_main_prepare_dataset_command_fails_clearly_without_audit_first(tmp_path, capsys):
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    make_image(raw_dir / "Alpha" / "a1.jpg")
    make_image(raw_dir / "Beta" / "b1.jpg")
    config_path = write_min_dataset_config(
        tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir
    )

    exit_code = run_pipeline.main(["prepare_dataset", "--config", str(config_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "SPLIT BUILD ERROR" in captured.err

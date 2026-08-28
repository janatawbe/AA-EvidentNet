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

# "audit", "prepare_dataset", and "model_check" are implemented (see
# test_main_audit_command_*, test_main_prepare_dataset_command_*, and
# test_main_model_check_command_* below); every other command is still
# recognized by the parser but not yet implemented.
ALL_COMMANDS = [
    "audit",
    "prepare_dataset",
    "model_check",
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

IMPLEMENTED_COMMANDS = {"audit", "prepare_dataset", "model_check"}
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


def test_main_train_command_aa_evidentnet_smoke_test_passes():
    # "train --model aa_evidentnet" is implemented (Task 7 completion:
    # src/training/run_aa_evidentnet.py, the combined classification +
    # CS-SupCon + EDL objective). Like "baseline", it is ONLY ever
    # exercised here with --smoke-test, never bare, since a bare
    # invocation would kick off a real, unbounded training run against
    # the real dataset.
    exit_code = run_pipeline.main(["train", "--model", "aa_evidentnet", "--smoke-test"])
    assert exit_code == 0


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


def test_main_prepare_dataset_command_runs_audit_internally(tmp_path):
    # prepare_dataset now runs the audit stage itself (see run_prepare_dataset
    # in run_pipeline.py) - it must succeed even without a prior standalone
    # `audit` invocation, producing the eligibility manifest along the way.
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

    exit_code = run_pipeline.main(["prepare_dataset", "--config", str(config_path)])

    assert exit_code == 0
    assert (audit_dir / "dataset_eligibility.csv").exists()
    assert (manifests_dir / "train_original.csv").exists()
    assert (manifests_dir / "train_balanced.csv").exists()


def test_main_prepare_dataset_command_fails_clearly_for_fatal_config_error(tmp_path, capsys):
    raw_dir = tmp_path / "does_not_exist"
    audit_dir = tmp_path / "audit"
    config_path = write_min_dataset_config(
        tmp_path, {"Alpha": "Alpha", "Beta": "Beta"}, raw_dir, audit_dir
    )

    exit_code = run_pipeline.main(["prepare_dataset", "--config", str(config_path)])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "AUDIT CONFIG ERROR" in captured.err


# --- "model_check" is implemented: offline, pretrained=False by default,
# fast, no dataset fixture needed (it only synthesizes tensors). ---


def test_main_model_check_command_passes_offline(tmp_path):
    output_csv = tmp_path / "model_parameters.csv"
    exit_code = run_pipeline.main(
        ["model_check", "--config", "configs/models.yaml", "--output-csv", str(output_csv)]
    )
    assert exit_code == 0
    assert output_csv.exists()


def test_main_model_check_command_default_is_offline_not_pretrained(tmp_path, capsys):
    output_csv = tmp_path / "model_parameters.csv"
    exit_code = run_pipeline.main(
        ["model_check", "--config", "configs/models.yaml", "--output-csv", str(output_csv)]
    )
    assert exit_code == 0
    captured = capsys.readouterr()
    assert "pretrained=False" in captured.out


# --- "baseline" is implemented: ONLY ever exercised here with
# --smoke-test, never bare, since a bare invocation would kick off a real,
# unbounded training run against the real dataset (dataset.yaml/training.yaml
# paths are not overridable via --config for this command - --config only
# points at models.yaml (use --dataset-config, tested separately below, to
# override configs/dataset.yaml) - so these tests unavoidably touch the
# real, but gitignored, results/logs, results/checkpoints, and
# experiments/registry.csv). ---


def test_main_baseline_command_smoke_test_passes():
    exit_code = run_pipeline.main(["baseline", "--model", "resnet50", "--smoke-test"])
    assert exit_code == 0


def test_main_baseline_command_invalid_model_fails_clearly(capsys):
    exit_code = run_pipeline.main(["baseline", "--model", "not_a_real_model", "--smoke-test"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "TRAINING SETUP ERROR" in captured.err


def test_main_train_command_non_aa_evidentnet_model_fails_clearly(capsys):
    exit_code = run_pipeline.main(["train", "--model", "resnet50"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "NOT IMPLEMENTED" in captured.err
    assert "baseline" in captured.err


# --- --dataset-config (Colab data-root override): baseline/train accept a
# custom dataset.yaml path without any source change (still --smoke-test
# only here, same rationale as above). ---


def test_parser_dataset_config_defaults_to_real_dataset_yaml():
    parser = run_pipeline.build_parser()
    for command in ("baseline", "train"):
        args = parser.parse_args([command, "--model", "resnet50"])
        assert args.dataset_config == "configs/dataset.yaml"


def test_parser_dataset_config_accepts_override():
    parser = run_pipeline.build_parser()
    args = parser.parse_args(["baseline", "--model", "resnet50", "--dataset-config", "configs/dataset.colab.yaml"])
    assert args.dataset_config == "configs/dataset.colab.yaml"


def test_main_baseline_command_honors_dataset_config_override(tmp_path):
    # A custom dataset.yaml (10 dummy classes, matching configs/models.yaml's
    # default num_classes=10) passed via --dataset-config must actually be
    # used - not silently ignored in favor of the real configs/dataset.yaml.
    from tests.conftest import write_min_dataset_config

    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)

    exit_code = run_pipeline.main(
        ["baseline", "--model", "resnet50", "--smoke-test", "--dataset-config", str(dataset_config_path)]
    )
    assert exit_code == 0

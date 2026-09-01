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
# test_main_model_check_command_* below); every other command in this list
# is still recognized by the parser but not yet implemented. "baseline",
# "train" (MODEL_COMMANDS below), "final_test", "robustness", and
# "ood_uncertainty" (their own dedicated tests below - all three require
# --model AND --checkpoint, so they don't fit this bare-command
# parametrization) are also implemented but excluded from this list for the
# same reason.
ALL_COMMANDS = [
    "audit",
    "prepare_dataset",
    "model_check",
    "ablation",
    "hard_pairs",
    "calibration",
    "selective",
    "gradcam",
    "multi_seed",
    "publication",
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


# --- "final_test" (Task 8) is implemented: requires --model AND
# --checkpoint (unlike every other command), so it needs its own parser
# tests rather than fitting ALL_COMMANDS/MODEL_COMMANDS above. Uses a tiny
# synthetic fixture dataset/checkpoint - never touches the real
# data/manifests/test_original.csv. There is no --registry-path CLI
# override, so this unavoidably does a best-effort lookup against the
# real experiments/registry.csv - but since this fixture's checkpoint
# directory name never matches a real experiment_id, that lookup finds
# nothing and the real registry is never modified (same reasoning as the
# baseline/train smoke tests above touching real results/logs/checkpoints). ---


def test_parser_final_test_requires_model_and_checkpoint():
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["final_test"])  # missing both --model and --checkpoint
    with pytest.raises(SystemExit):
        parser.parse_args(["final_test", "--model", "resnet50"])  # missing --checkpoint


def test_parser_final_test_config_defaults_to_models_yaml():
    parser = run_pipeline.build_parser()
    args = parser.parse_args(["final_test", "--model", "resnet50", "--checkpoint", "x.pt"])
    assert args.config == "configs/models.yaml"


def test_main_final_test_command_end_to_end(tmp_path):
    import yaml

    from src.models.factory import create_model
    from src.training.checkpointing import build_checkpoint, save_checkpoint
    from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

    canonical_classes = ["Alpha", "Beta", "Gamma"]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(tmp_path, num_classes=len(canonical_classes))

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in canonical_classes:
        make_image(raw_dir / name / f"{name}.jpg")
        rows.append(
            {
                "path": f"{name}/{name}.jpg",
                "class": name,
                "split": "test",
                "original_id": f"id_{name}",
                "parent_original_id": f"id_{name}",
                "is_original": "true",
                "augmentation_type": "original",
            }
        )
    import csv

    with open(manifests_dir / "test_original.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]
        )
        writer.writeheader()
        writer.writerows(rows)

    import torch

    models_config = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    model = create_model("resnet50", models_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, scheduler=None, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture=model.architecture,
        num_classes=len(canonical_classes), dataset_manifest_hash="h", git_commit="c",
    )
    checkpoint_dir = tmp_path / "checkpoints" / "cli_test_run"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)

    exit_code = run_pipeline.main(
        [
            "final_test",
            "--model", "resnet50",
            "--checkpoint", str(checkpoint_path),
            "--dataset-config", str(dataset_config_path),
            "--config", str(models_config_path),
            "--device", "cpu",
            "--num-workers", "0",
        ]
    )
    assert exit_code == 0


# --- "robustness" is implemented: requires --model AND --checkpoint (same
# shape as "final_test" above), so it also needs its own dedicated tests
# rather than fitting ALL_COMMANDS/MODEL_COMMANDS. Uses a tiny synthetic
# fixture dataset/checkpoint and a tmp_path-scoped evaluation.yaml (a
# reduced 1-condition degradation table, purely to keep this CLI smoke test
# fast) - never touches the real data/manifests/test_original.csv or the
# real configs/evaluation.yaml's full severity table. ---


def test_parser_robustness_requires_model_and_checkpoint():
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["robustness"])  # missing both --model and --checkpoint
    with pytest.raises(SystemExit):
        parser.parse_args(["robustness", "--model", "resnet50"])  # missing --checkpoint


def test_parser_robustness_config_defaults_to_models_yaml():
    parser = run_pipeline.build_parser()
    args = parser.parse_args(["robustness", "--model", "resnet50", "--checkpoint", "x.pt"])
    assert args.config == "configs/models.yaml"


def test_main_robustness_command_end_to_end(tmp_path):
    import yaml

    from src.models.factory import create_model
    from src.training.checkpointing import build_checkpoint, save_checkpoint
    from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

    canonical_classes = ["Alpha", "Beta", "Gamma"]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(tmp_path, num_classes=len(canonical_classes))

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in canonical_classes:
        make_image(raw_dir / name / f"{name}.jpg")
        rows.append(
            {
                "path": f"{name}/{name}.jpg",
                "class": name,
                "split": "test",
                "original_id": f"id_{name}",
                "parent_original_id": f"id_{name}",
                "is_original": "true",
                "augmentation_type": "original",
            }
        )
    import csv

    with open(manifests_dir / "test_original.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]
        )
        writer.writeheader()
        writer.writerows(rows)

    import torch

    models_config = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    model = create_model("resnet50", models_config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    checkpoint = build_checkpoint(
        model=model, optimizer=optimizer, scheduler=None, epoch=1, best_metric=0.5, monitor_metric="val_macro_f1",
        training_config={}, seed=42, model_name="resnet50", architecture=model.architecture,
        num_classes=len(canonical_classes), dataset_manifest_hash="h", git_commit="c",
    )
    checkpoint_dir = tmp_path / "checkpoints" / "cli_robustness_test_run"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)

    # run_robustness_evaluation's evaluation_config_path is not CLI-
    # overridable (same convention as final_test's evaluation_config_path),
    # so this end-to-end CLI invocation necessarily uses the real
    # configs/evaluation.yaml (full severity table) and writes into the
    # real results/robustness/<robustness_run_id>/ directory - a fresh,
    # uniquely-timestamped subdirectory that never collides with or
    # overwrites anything, exactly like test_main_final_test_command_end_to_end
    # above already does for results/tables/ and results/raw_predictions/.
    exit_code = run_pipeline.main(
        [
            "robustness",
            "--model", "resnet50",
            "--checkpoint", str(checkpoint_path),
            "--dataset-config", str(dataset_config_path),
            "--config", str(models_config_path),
            "--device", "cpu",
            "--num-workers", "0",
        ]
    )
    assert exit_code == 0


# --- "ood_uncertainty" is implemented: requires --model AND --checkpoint
# (same shape as "final_test"/"robustness" above), so it also needs its own
# dedicated tests. AA-EvidentNet only (baselines are explicitly rejected).
# Uses a tiny synthetic fixture with train_original.csv/val_original.csv/
# test_original.csv all present (calibration needs the first two;
# evaluation needs the third) - never touches the real
# data/manifests/*.csv. ---


def test_parser_ood_uncertainty_requires_model_and_checkpoint():
    parser = run_pipeline.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["ood_uncertainty"])  # missing both --model and --checkpoint
    with pytest.raises(SystemExit):
        parser.parse_args(["ood_uncertainty", "--model", "aa_evidentnet"])  # missing --checkpoint


def test_parser_ood_uncertainty_config_defaults_to_models_yaml():
    parser = run_pipeline.build_parser()
    args = parser.parse_args(["ood_uncertainty", "--model", "aa_evidentnet", "--checkpoint", "x.pt"])
    assert args.config == "configs/models.yaml"


def test_main_ood_uncertainty_rejects_non_aa_evidentnet_model(tmp_path):
    from src.models.factory import create_model
    from src.training.checkpointing import build_checkpoint, save_checkpoint
    from tests.conftest import make_image, write_min_dataset_config, write_min_models_config
    import csv
    import torch
    import yaml

    canonical_classes = ["Alpha", "Beta", "Gamma"]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(tmp_path, num_classes=len(canonical_classes))

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    columns = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]
    rows = []
    for name in canonical_classes:
        make_image(raw_dir / name / f"{name}.jpg")
        rows.append(
            {
                "path": f"{name}/{name}.jpg", "class": name, "split": "test",
                "original_id": f"id_{name}", "parent_original_id": f"id_{name}",
                "is_original": "true", "augmentation_type": "original",
            }
        )
    with open(manifests_dir / "test_original.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    models_config = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    model = create_model("resnet50", models_config)
    checkpoint = build_checkpoint(
        model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), scheduler=None,
        epoch=1, best_metric=0.5, monitor_metric="val_macro_f1", training_config={}, seed=42,
        model_name="resnet50", architecture=model.architecture, num_classes=len(canonical_classes),
        dataset_manifest_hash="h", git_commit="c",
    )
    checkpoint_dir = tmp_path / "checkpoints" / "cli_ood_reject_test_run"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)

    exit_code = run_pipeline.main(
        [
            "ood_uncertainty",
            "--model", "resnet50",
            "--checkpoint", str(checkpoint_path),
            "--dataset-config", str(dataset_config_path),
            "--config", str(models_config_path),
            "--device", "cpu",
            "--num-workers", "0",
        ]
    )
    assert exit_code == 1


def test_main_ood_uncertainty_command_end_to_end(tmp_path):
    import csv

    import torch
    import yaml

    from src.models.factory import create_model
    from src.training.checkpointing import build_checkpoint, save_checkpoint
    from tests.conftest import make_image, write_min_dataset_config, write_min_models_config

    canonical_classes = ["Alpha", "Beta", "Gamma"]
    raw_dir = tmp_path / "raw"
    audit_dir = tmp_path / "audit"
    mapping = {name: name for name in canonical_classes}
    dataset_config_path = write_min_dataset_config(tmp_path, mapping, raw_dir, audit_dir)
    models_config_path = write_min_models_config(tmp_path, num_classes=len(canonical_classes), include_aa_evidentnet=True)

    manifests_dir = tmp_path / "manifests"
    manifests_dir.mkdir(parents=True, exist_ok=True)
    columns = ["path", "class", "split", "original_id", "parent_original_id", "is_original", "augmentation_type"]

    def write_split(split, n_per_class):
        rows = []
        for name in canonical_classes:
            for i in range(n_per_class):
                filename = f"{split}_{name}_{i}.jpg"
                make_image(raw_dir / name / filename)
                rows.append(
                    {
                        "path": f"{name}/{filename}", "class": name, "split": split,
                        "original_id": f"id_{split}_{name}_{i}", "parent_original_id": f"id_{split}_{name}_{i}",
                        "is_original": "true", "augmentation_type": "original",
                    }
                )
        with open(manifests_dir / f"{split}_original.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    write_split("train", 3)
    write_split("val", 2)
    write_split("test", 1)

    models_config = yaml.safe_load(models_config_path.read_text(encoding="utf-8"))
    model = create_model("aa_evidentnet", models_config)
    checkpoint = build_checkpoint(
        model=model, optimizer=torch.optim.AdamW(model.parameters(), lr=1e-3), scheduler=None,
        epoch=1, best_metric=0.5, monitor_metric="val_macro_f1", training_config={}, seed=42,
        model_name="aa_evidentnet", architecture=model.architecture, num_classes=len(canonical_classes),
        dataset_manifest_hash="h", git_commit="c",
    )
    checkpoint_dir = tmp_path / "checkpoints" / "cli_ood_uncertainty_test_run"
    checkpoint_dir.mkdir(parents=True)
    checkpoint_path = checkpoint_dir / "best.pt"
    save_checkpoint(checkpoint, checkpoint_path)

    # run_ood_uncertainty_evaluation's evaluation_config_path is not CLI-
    # overridable (same convention as final_test's/robustness's), so this
    # end-to-end CLI invocation necessarily uses the real
    # configs/evaluation.yaml (full degradation table + weight grid) and
    # writes into the real results/ood_uncertainty/<run_id>/ directory - a
    # fresh, uniquely-timestamped subdirectory that never collides with or
    # overwrites anything, exactly like test_main_robustness_command_end_to_end
    # above already does for results/robustness/.
    exit_code = run_pipeline.main(
        [
            "ood_uncertainty",
            "--model", "aa_evidentnet",
            "--checkpoint", str(checkpoint_path),
            "--dataset-config", str(dataset_config_path),
            "--config", str(models_config_path),
            "--device", "cpu",
            "--num-workers", "0",
        ]
    )
    assert exit_code == 0

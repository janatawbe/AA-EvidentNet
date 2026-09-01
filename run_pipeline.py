#!/usr/bin/env python
"""AA-EvidentNet pipeline CLI.

This is the single entry point for the project's experiment pipeline.
`audit`, `prepare_dataset`, `model_check`, `baseline`, `train`
(AA-EvidentNet, Task 7 completion), `final_test` (Task 8), `robustness`,
and `ood_uncertainty` are implemented. Every other command (`ablation`,
`hard_pairs`, `calibration`, `selective`, `gradcam`, `multi_seed`,
`publication`) is recognized by the CLI but not yet implemented, and
fails with a clear NotImplementedError-derived message rather than doing
nothing silently.

Usage:
    python run_pipeline.py <command> [options]

Run `python run_pipeline.py --help` or `python run_pipeline.py <command>
--help` for details.
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT))

from src.utils.seeding import DEFAULT_SEED, set_seed  # noqa: E402
from src.data.audit_dataset import (  # noqa: E402
    AuditConfigError,
    AuditFailedError,
    run_dataset_audit,
)
from src.data.build_split import (  # noqa: E402
    SplitBuildError,
    SplitValidationError,
    run_build_split,
)
from src.data.duplicate_review import ReviewValidationError  # noqa: E402
from src.data.eligibility import EligibilityValidationError  # noqa: E402
from src.data.generate_balanced_dataset import (  # noqa: E402
    BalancedDatasetBuildError,
    BalancedDatasetValidationError,
    run_generate_balanced_dataset,
)
from src.evaluation.final_test import FinalTestError, run_final_test  # noqa: E402
from src.evaluation.metrics import MetricsInputError  # noqa: E402
from src.evaluation.ood_uncertainty import OODUncertaintyError, run_ood_uncertainty_evaluation  # noqa: E402
from src.evaluation.robustness import RobustnessError, run_robustness_evaluation  # noqa: E402
from src.losses.combined import CombinedObjectiveConfigError  # noqa: E402
from src.losses.cs_supcon import CSSupConConfigError  # noqa: E402
from src.losses.evidential import EvidentialConfigError  # noqa: E402
from src.models.model_check import ModelCheckError, run_model_check  # noqa: E402
from src.models.factory import ModelConfigError  # noqa: E402
from src.training.checkpointing import CheckpointIncompatibleError  # noqa: E402
from src.training.run_aa_evidentnet import RunAAEvidentNetError, run_aa_evidentnet_training  # noqa: E402
from src.training.run_baseline import RunBaselineError, run_baseline_training  # noqa: E402
from src.training.trainer import TrainerError  # noqa: E402


class PipelineNotImplementedError(NotImplementedError):
    """Raised by a command handler that is recognized but not yet built."""


DEFAULT_CONFIGS = {
    "audit": "configs/dataset.yaml",
    "prepare_dataset": "configs/dataset.yaml",
    "model_check": "configs/models.yaml",
    "baseline": "configs/models.yaml",
    "train": "configs/models.yaml",
    "ablation": "configs/experiments.yaml",
    "hard_pairs": "configs/evaluation.yaml",
    "calibration": "configs/evaluation.yaml",
    "selective": "configs/evaluation.yaml",
    "gradcam": "configs/evaluation.yaml",
    "multi_seed": "configs/experiments.yaml",
    "publication": "configs/experiments.yaml",
    # models.yaml, not evaluation.yaml: --config here selects the model
    # architecture config (same convention as baseline/train), consistent
    # with how run_final_test's `models_config_path` param is used.
    # configs/evaluation.yaml is loaded separately, internally, with its
    # own fixed default (src/evaluation/final_test.py:
    # evaluation_config_path) - it is not CLI-overridable, since Task 8
    # did not ask for that and the existing --config flag is already
    # spoken for by models.yaml here. robustness.py follows the identical
    # convention (see run_robustness_evaluation's models_config_path param).
    "final_test": "configs/models.yaml",
    "robustness": "configs/models.yaml",
    "ood_uncertainty": "configs/models.yaml",
}


def add_common_arguments(subparser: argparse.ArgumentParser, command: str) -> None:
    subparser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIGS.get(command),
        help="Path to a YAML config file (default: %(default)s).",
    )
    subparser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed (default: {DEFAULT_SEED}).",
    )
    subparser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Compute device (default: auto).",
    )
    subparser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the batch size from the config.",
    )
    subparser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of epochs from the config.",
    )
    subparser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Override the number of dataloader workers from the config.",
    )
    subparser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run a fast, reduced-scope version of this command for sanity checking.",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py",
        description="AA-EvidentNet experiment pipeline CLI.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    commands = [
        ("audit", "Run the dataset audit (class counts, duplicates, corruption, leakage checks)."),
        ("prepare_dataset", "Build processed dataset splits and manifests from data/raw."),
        ("model_check", "Instantiate baseline models with a synthetic batch and report shapes/param counts (no training)."),
        ("baseline", "Train/evaluate a baseline model (e.g. MaxViT)."),
        ("train", "Train the proposed AA-EvidentNet model."),
        ("ablation", "Run ablation studies over AA-EvidentNet components."),
        ("hard_pairs", "Analyze visually/clinically confusable class pairs."),
        ("calibration", "Evaluate calibration and uncertainty quantification."),
        ("selective", "Evaluate selective prediction / risk-coverage behavior."),
        ("gradcam", "Generate Grad-CAM and other interpretability visualizations."),
        ("robustness", "Evaluate robustness to input perturbations."),
        ("multi_seed", "Aggregate results across the project's multi-seed runs."),
        ("publication", "Assemble publication-ready tables and figures."),
        ("final_test", "Run the final held-out test evaluation."),
        ("ood_uncertainty", "Evaluate a feature-distance OOD score combined with EDL uncertainty (AA-EvidentNet only)."),
    ]

    for name, help_text in commands:
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        add_common_arguments(sub, name)
        if name in ("baseline", "train", "final_test", "robustness", "ood_uncertainty"):
            sub.add_argument(
                "--model",
                type=str,
                required=True,
                help="Model name (see configs/models.yaml registry, "
                "e.g. 'maxvit' or 'aa_evidentnet').",
            )
            sub.add_argument(
                "--dataset-config",
                type=str,
                default="configs/dataset.yaml",
                help="Path to a YAML dataset config (default: %(default)s). "
                "Override this to point `paths.raw_dir`/`paths.processed_dir`/"
                "`paths.manifests_dir` at an external location (e.g. a mounted "
                "Google Drive path on Colab) without editing configs/dataset.yaml "
                "or any source file - copy it and change only the `paths:` "
                "section; class names, split ratios, and augmentation policy "
                "must stay identical to the original.",
            )
        if name in ("baseline", "train"):
            sub.add_argument(
                "--resume",
                type=str,
                default=None,
                help="Path to a checkpoint (.pt) to resume training from.",
            )
        if name in ("final_test", "robustness", "ood_uncertainty"):
            sub.add_argument(
                "--checkpoint",
                type=str,
                required=True,
                help="Path to the FROZEN checkpoint (.pt) to evaluate on the held-out "
                "test set (data/manifests/test_original.csv). Weights are loaded and "
                "never modified - no training, no tuning, no optimizer/scheduler.",
            )
        if name == "model_check":
            sub.add_argument(
                "--pretrained",
                action="store_true",
                help="Also verify ImageNet-pretrained weights can be downloaded/instantiated "
                "for each baseline. Requires internet access. Default: offline (pretrained=False).",
            )
            sub.add_argument(
                "--output-csv",
                type=str,
                default="results/tables/model_parameters.csv",
                help="Where to write the parameter/feature report (default: %(default)s).",
            )

    return parser


def not_implemented(command: str, target_module: str) -> None:
    raise PipelineNotImplementedError(
        f"Command '{command}' is recognized but not yet implemented.\n"
        f"It will be implemented in {target_module}.\n"
        f"This is expected at the current stage of the project "
        f"(foundation setup only) - no dataset pipeline, model, or "
        f"training code exists yet."
    )


def run_audit(args: argparse.Namespace) -> None:
    run_dataset_audit(config_path=args.config)


def run_prepare_dataset(args: argparse.Namespace) -> None:
    """Full dataset-preparation pipeline: audit -> eligibility -> original
    70/20/10 split -> split validation -> balanced (2000/class) training
    set -> balanced manifest validation -> augmentation statistics ->
    hashes/provenance.

    A policy-driven AuditFailedError (e.g. the ever-present unresolved
    cross-class duplicate groups) is reported but does not abort this
    pipeline: the eligibility layer that the split/balance stages consume
    already excludes those images, which is the actual leakage-prevention
    mechanism. An AuditConfigError (a genuinely broken config/raw_dir) is
    NOT caught here and still aborts everything, as it should.
    """
    try:
        run_dataset_audit(config_path=args.config)
    except AuditFailedError as e:
        print(
            f"[run_pipeline] audit reported policy-'fail' issues (continuing - the "
            f"eligibility layer already excludes them from the split/balance below): {e}",
            file=sys.stderr,
        )
    run_build_split(config_path=args.config, seed=args.seed)
    run_generate_balanced_dataset(config_path=args.config, seed=args.seed)


def run_model_check_command(args: argparse.Namespace) -> None:
    run_model_check(config_path=args.config, check_pretrained=args.pretrained, output_csv=args.output_csv)


def run_baseline_command(args: argparse.Namespace) -> None:
    """python run_pipeline.py baseline --model {resnet50,efficientnetb0,maxvit}.

    Training always uses data/manifests/train_balanced.csv; validation
    always uses data/manifests/val_original.csv. The test set is never
    touched here.
    """
    summary = run_baseline_training(
        model_name=args.model,
        dataset_config_path=args.dataset_config,
        models_config_path=args.config,
        seed=args.seed,
        device_override=None if args.device == "auto" else args.device,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
        num_workers_override=args.num_workers,
        smoke_test=args.smoke_test,
        resume_from=args.resume,
    )
    print(
        f"[run_pipeline] run_id={summary.run_id} best_epoch={summary.fit_result.best_epoch} "
        f"best_metric={summary.fit_result.best_metric} checkpoint={summary.best_checkpoint_path}"
    )


def run_train_command(args: argparse.Namespace) -> None:
    """python run_pipeline.py train --model aa_evidentnet.

    Training always uses data/manifests/train_balanced.csv; validation
    always uses data/manifests/val_original.csv; the test set is never
    touched here (see src/training/run_aa_evidentnet.py). The training
    objective is the combined classification + CS-SupCon + EDL loss
    (src/losses/combined.py), driven by configs/losses.yaml.
    """
    if args.model != "aa_evidentnet":
        raise PipelineNotImplementedError(
            f"Command 'train --model {args.model}' is not recognized. "
            "'train' is reserved for the proposed model (aa_evidentnet). "
            "Use `python run_pipeline.py baseline --model <name>` for a baseline."
        )
    summary = run_aa_evidentnet_training(
        dataset_config_path=args.dataset_config,
        models_config_path=args.config,
        seed=args.seed,
        device_override=None if args.device == "auto" else args.device,
        batch_size_override=args.batch_size,
        epochs_override=args.epochs,
        num_workers_override=args.num_workers,
        smoke_test=args.smoke_test,
        resume_from=args.resume,
    )
    print(
        f"[run_pipeline] run_id={summary.run_id} best_epoch={summary.fit_result.best_epoch} "
        f"best_metric={summary.fit_result.best_metric} checkpoint={summary.best_checkpoint_path}"
    )


def run_final_test_command(args: argparse.Namespace) -> None:
    """python run_pipeline.py final_test --model <name> --checkpoint <path.pt>.

    Evaluates a single FROZEN, already-trained checkpoint on the held-out
    data/manifests/test_original.csv (src/evaluation/final_test.py). Never
    trains, never tunes, never modifies the checkpoint's weights - loads
    them once via the existing checkpoint utilities and runs inference
    only. Supports all four registered models (resnet50, efficientnetb0,
    maxvit, aa_evidentnet).
    """
    summary = run_final_test(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        models_config_path=args.config,
        seed=args.seed,
        device_override=None if args.device == "auto" else args.device,
        num_workers_override=args.num_workers,
        batch_size_override=args.batch_size,
    )
    print(
        f"[run_pipeline] eval_run_id={summary.eval_run_id} model={summary.model_name} "
        f"n={summary.num_samples} accuracy={summary.overall_metrics.get('accuracy')} "
        f"macro_f1={summary.overall_metrics.get('macro_f1')} "
        f"predictions={summary.predictions_path} registry_updated={summary.registry_updated}"
    )


def run_robustness_command(args: argparse.Namespace) -> None:
    """python run_pipeline.py robustness --model <name> --checkpoint <path.pt>.

    Evaluates a single FROZEN, already-finally-tested checkpoint
    (src/evaluation/final_test.py, Task 8) against fixed, predefined image
    degradations (src/evaluation/robustness.py) - a separate, additional
    test-time analysis. Never trains, never tunes, never modifies the
    checkpoint's weights, and never overwrites Task 8's clean final-test
    outputs (results/robustness/ is entirely separate from
    results/raw_predictions/ and results/tables/<final_test_run_id>/).
    """
    summary = run_robustness_evaluation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        models_config_path=args.config,
        seed=args.seed,
        device_override=None if args.device == "auto" else args.device,
        num_workers_override=args.num_workers,
        batch_size_override=args.batch_size,
    )
    print(
        f"[run_pipeline] robustness_run_id={summary.robustness_run_id} model={summary.model_name} "
        f"n={summary.num_samples} metrics={summary.metrics_path} metadata={summary.metadata_path}"
    )


def run_ood_uncertainty_command(args: argparse.Namespace) -> None:
    """python run_pipeline.py ood_uncertainty --model aa_evidentnet --checkpoint <path.pt>.

    Combines a feature-distance (cosine, to the nearest train_original.csv
    class prototype) OOD score with AA-EvidentNet's own EDL uncertainty
    (src/evaluation/ood_uncertainty.py) - AA-EvidentNet only (baselines have
    neither a fused embedding nor an evidential head). Calibrated entirely
    from train_original.csv/val_original.csv, never from
    test_original.csv; never trains, never tunes, never modifies the
    checkpoint's weights; never overwrites final_test's or robustness's
    outputs (results/ood_uncertainty/ is its own, separate directory).
    """
    summary = run_ood_uncertainty_evaluation(
        model_name=args.model,
        checkpoint_path=args.checkpoint,
        dataset_config_path=args.dataset_config,
        models_config_path=args.config,
        seed=args.seed,
        device_override=None if args.device == "auto" else args.device,
        num_workers_override=args.num_workers,
        batch_size_override=args.batch_size,
    )
    print(
        f"[run_pipeline] ood_uncertainty_run_id={summary.run_id} model={summary.model_name} "
        f"n={summary.num_test_samples} weight={summary.weight} metrics={summary.metrics_path} "
        f"figure={summary.figure_path}"
    )


def dispatch(args: argparse.Namespace) -> None:
    command = args.command

    handlers = {
        "audit": lambda: run_audit(args),
        "prepare_dataset": lambda: run_prepare_dataset(args),
        "model_check": lambda: run_model_check_command(args),
        "baseline": lambda: run_baseline_command(args),
        "train": lambda: run_train_command(args),
        "ablation": lambda: not_implemented(command, "src/training (ablation runner)"),
        "hard_pairs": lambda: not_implemented(command, "src/evaluation (hard pairs analysis)"),
        "calibration": lambda: not_implemented(command, "src/evaluation (calibration/uncertainty)"),
        "selective": lambda: not_implemented(command, "src/evaluation (selective prediction)"),
        "gradcam": lambda: not_implemented(command, "src/visualization (Grad-CAM)"),
        "robustness": lambda: run_robustness_command(args),
        "multi_seed": lambda: not_implemented(command, "src/statistics (multi-seed aggregation)"),
        "publication": lambda: not_implemented(command, "src/visualization + src/statistics (publication assets)"),
        "final_test": lambda: run_final_test_command(args),
        "ood_uncertainty": lambda: run_ood_uncertainty_command(args),
    }

    handlers[command]()


def main(argv=None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    set_seed(args.seed)

    try:
        dispatch(args)
    except PipelineNotImplementedError as e:
        print(f"[run_pipeline] NOT IMPLEMENTED: {e}", file=sys.stderr)
        return 1
    except AuditConfigError as e:
        print(f"[run_pipeline] AUDIT CONFIG ERROR: {e}", file=sys.stderr)
        return 1
    except AuditFailedError as e:
        print(f"[run_pipeline] AUDIT FAILED: {e}", file=sys.stderr)
        return 1
    except (SplitBuildError, ReviewValidationError, EligibilityValidationError) as e:
        print(f"[run_pipeline] SPLIT BUILD ERROR: {e}", file=sys.stderr)
        return 1
    except SplitValidationError as e:
        print(f"[run_pipeline] SPLIT VALIDATION FAILED: {e}", file=sys.stderr)
        return 1
    except BalancedDatasetBuildError as e:
        print(f"[run_pipeline] BALANCED DATASET BUILD ERROR: {e}", file=sys.stderr)
        return 1
    except BalancedDatasetValidationError as e:
        print(f"[run_pipeline] BALANCED DATASET VALIDATION FAILED: {e}", file=sys.stderr)
        return 1
    except ModelCheckError as e:
        print(f"[run_pipeline] MODEL CHECK FAILED: {e}", file=sys.stderr)
        return 1
    except (
        RunBaselineError,
        RunAAEvidentNetError,
        ModelConfigError,
        CheckpointIncompatibleError,
        CombinedObjectiveConfigError,
        CSSupConConfigError,
        EvidentialConfigError,
    ) as e:
        print(f"[run_pipeline] TRAINING SETUP ERROR: {e}", file=sys.stderr)
        return 1
    except TrainerError as e:
        print(f"[run_pipeline] TRAINER ERROR: {e}", file=sys.stderr)
        return 1
    except (FinalTestError, MetricsInputError) as e:
        print(f"[run_pipeline] FINAL TEST ERROR: {e}", file=sys.stderr)
        return 1
    except RobustnessError as e:
        print(f"[run_pipeline] ROBUSTNESS ERROR: {e}", file=sys.stderr)
        return 1
    except OODUncertaintyError as e:
        print(f"[run_pipeline] OOD UNCERTAINTY ERROR: {e}", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

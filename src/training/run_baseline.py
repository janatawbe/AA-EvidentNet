"""Orchestrates one baseline-model training run: loads config, builds the
dataset/dataloaders/model/optimizer/scheduler, drives src.training.trainer.Trainer,
and wires up checkpointing/logging/the experiment registry.

This is deliberately separate from Trainer itself: Trainer is pure
train/validate-loop mechanics reusable by any future model (AA-EvidentNet
included); this module is the CLI-facing glue specific to "run one of the
three registered baselines against train_balanced.csv / val_original.csv".

Test set lock: this module never imports, loads, or references
test_original.csv anywhere. Model selection (early stopping, "best"
checkpoint, LR scheduling) is driven exclusively by validation metrics.
"""

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from src.data.dataloaders import build_eval_dataloader, build_train_dataloader
from src.data.dataset import RetinalDataset
from src.data.transforms import build_transforms_from_config
from src.models.factory import create_model
from src.training.checkpointing import (
    assert_checkpoint_compatible,
    build_checkpoint,
    load_checkpoint,
    restore_training_state,
    save_checkpoint,
)
from src.training.logging import RunLogger, generate_run_id
from src.training.registry import (
    DEFAULT_REGISTRY_PATH,
    STATUS_COMPLETED,
    STATUS_FAILED,
    STATUS_RUNNING,
    register_run,
    update_run,
)
from src.training.trainer import FitResult, Trainer, TrainingConfig, build_optimizer, build_scheduler, resolve_device
from src.utils.config import hash_config, load_config
from src.utils.env_info import collect_environment_info
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file
from src.utils.seeding import DEFAULT_SEED, set_seed

BASELINE_MODEL_NAMES = ("resnet50", "efficientnetb0", "maxvit")

# Smoke-test defaults: small enough to run in seconds on CPU, but still
# exercise a REAL forward/backward/optimizer-step/validation cycle (not a
# no-op). batch_size=4 with 8 train / 4 val synthetic samples guarantees at
# least one full training batch even with drop_last=True - a larger
# default batch_size (e.g. the real config's 16) would silently produce
# ZERO training batches from only 8 samples.
SMOKE_TEST_BATCH_SIZE = 4
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_N_TRAIN = 8
SMOKE_TEST_N_VAL = 4


class RunBaselineError(Exception):
    """Fatal, unrecoverable problem starting/running a baseline training
    run (unknown model, missing manifests, incompatible resume checkpoint)."""


@dataclass
class BaselineRunSummary:
    run_id: str
    model_name: str
    architecture: str
    seed: int
    device: str
    amp_enabled: bool
    smoke_test: bool
    run_dir: Path
    checkpoint_dir: Path
    best_checkpoint_path: Optional[Path]
    latest_checkpoint_path: Optional[Path]
    fit_result: FitResult
    train_samples: int
    val_samples: int
    dataset_manifest_hash: str
    config_hash: str
    git_commit: Optional[str]


def _make_smoke_dataset(canonical_classes: List[str], raw_dir: Path, seed: int, n_train: int = 8, n_val: int = 4):
    """A tiny, fully self-contained synthetic dataset for the smoke test -
    never touches the real dataset, works in any environment. All rows are
    marked is_original=true referencing freshly-generated tiny JPEGs under
    raw_dir; the from-processed-dir code path is already covered by
    tests/test_dataset.py and is not re-verified here."""
    rng = random.Random(seed)
    classes = list(canonical_classes)

    def make_rows(n: int, split: str, prefix: str) -> List[Dict[str, Any]]:
        rows = []
        for i in range(n):
            canonical_class = classes[i % len(classes)]
            filename = f"{prefix}{i}.jpg"
            rel_path = f"{canonical_class}/{filename}"
            abs_path = raw_dir / rel_path
            abs_path.parent.mkdir(parents=True, exist_ok=True)
            color = (rng.randint(0, 255), rng.randint(0, 255), rng.randint(0, 255))
            Image.new("RGB", (64, 64), color).save(abs_path, format="JPEG")
            original_id = f"smoke_{prefix}_{i}"
            rows.append(
                {
                    "path": rel_path,
                    "class": canonical_class,
                    "split": split,
                    "original_id": original_id,
                    "parent_original_id": original_id,
                    "is_original": "true",
                    "augmentation_type": "original",
                }
            )
        return rows

    return make_rows(n_train, "train", "train"), make_rows(n_val, "val", "val")


def _resolve_num_workers(device, num_workers_override: Optional[int], config_num_workers: int) -> int:
    if num_workers_override is not None:
        return num_workers_override
    # Default to 0 on CPU: safer/faster for smoke tests and short sanity
    # runs than paying multiprocessing worker start-up cost for a handful
    # of batches; the configured value is still honored on CUDA (the
    # target RTX 3050 6GB deployment) or when explicitly overridden.
    if device.type == "cpu":
        return 0
    return config_num_workers


def run_baseline_training(
    model_name: str,
    dataset_config_path: Union[str, Path] = "configs/dataset.yaml",
    models_config_path: Union[str, Path] = "configs/models.yaml",
    training_config_path: Union[str, Path] = "configs/training.yaml",
    seed: int = DEFAULT_SEED,
    device_override: Optional[str] = None,
    batch_size_override: Optional[int] = None,
    epochs_override: Optional[int] = None,
    num_workers_override: Optional[int] = None,
    smoke_test: bool = False,
    resume_from: Optional[Union[str, Path]] = None,
    max_train_steps_per_epoch: Optional[int] = None,
    max_val_steps_per_epoch: Optional[int] = None,
    registry_path: Union[str, Path] = DEFAULT_REGISTRY_PATH,
    run_notes: str = "",
) -> BaselineRunSummary:
    if model_name not in BASELINE_MODEL_NAMES:
        raise RunBaselineError(f"Unknown baseline model '{model_name}'. Known baselines: {list(BASELINE_MODEL_NAMES)}")

    set_seed(seed)

    dataset_config = load_config(dataset_config_path)
    models_config = load_config(models_config_path)
    training_config_dict = load_config(training_config_path)

    if device_override:
        training_config_dict = {**training_config_dict, "device": device_override}
    if batch_size_override is not None:
        training_config_dict = {**training_config_dict, "batch_size": batch_size_override}
    elif smoke_test:
        training_config_dict = {**training_config_dict, "batch_size": SMOKE_TEST_BATCH_SIZE}
    if epochs_override is not None:
        training_config_dict = {**training_config_dict, "epochs": epochs_override}
    elif smoke_test:
        training_config_dict = {**training_config_dict, "epochs": SMOKE_TEST_EPOCHS}

    training_config = TrainingConfig.from_dict(training_config_dict)
    device = resolve_device(training_config_dict.get("device", "auto"))
    num_workers = _resolve_num_workers(device, num_workers_override, training_config.num_workers)

    canonical_classes = sorted(dataset_config["class_directory_mapping"].keys())
    num_classes = len(canonical_classes)
    train_transform, eval_transform = build_transforms_from_config(dataset_config)

    raw_dir = Path(dataset_config["paths"]["raw_dir"])
    processed_train_dir = Path(dataset_config["paths"]["processed_dir"]) / "train"
    manifests_dir = Path(dataset_config["paths"]["manifests_dir"])

    run_id = generate_run_id(model_name, seed, smoke_test=smoke_test)
    run_dir = Path(training_config_dict.get("logging", {}).get("log_dir", "results/logs")) / run_id
    checkpoint_dir = Path(training_config_dict.get("checkpointing", {}).get("save_dir", "results/checkpoints")) / run_id

    if smoke_test:
        # A real OS temp dir, deliberately NOT nested under run_dir:
        # RunLogger below requires run_dir to not exist yet (it refuses to
        # overwrite another run), and this scratch data is throwaway.
        smoke_raw_dir = Path(tempfile.mkdtemp(prefix=f"{run_id}_smoke_")) / "raw"
        train_rows, val_rows = _make_smoke_dataset(
            canonical_classes, smoke_raw_dir, seed, n_train=SMOKE_TEST_N_TRAIN, n_val=SMOKE_TEST_N_VAL
        )
        train_dataset = RetinalDataset(train_rows, canonical_classes, smoke_raw_dir, smoke_raw_dir, transform=train_transform)
        val_dataset = RetinalDataset(val_rows, canonical_classes, smoke_raw_dir, smoke_raw_dir, transform=eval_transform)
        dataset_manifest_hash = "smoke_test_synthetic_data_no_real_manifest"
        train_manifest_path_for_log = "<synthetic smoke-test data>"
    else:
        train_manifest_path = manifests_dir / "train_balanced.csv"
        val_manifest_path = manifests_dir / "val_original.csv"
        if not train_manifest_path.is_file():
            raise RunBaselineError(
                f"{train_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first."
            )
        train_dataset = RetinalDataset.from_manifest(
            train_manifest_path, canonical_classes, raw_dir, processed_train_dir, transform=train_transform, expected_split="train"
        )
        val_dataset = RetinalDataset.from_manifest(
            val_manifest_path,
            canonical_classes,
            raw_dir,
            processed_train_dir,
            transform=eval_transform,
            expected_split="val",
            require_all_original=True,
        )
        dataset_manifest_hash = hash_file(train_manifest_path)
        train_manifest_path_for_log = str(train_manifest_path)

    train_loader = build_train_dataloader(train_dataset, batch_size=training_config.batch_size, num_workers=num_workers, seed=seed)
    val_loader = build_eval_dataloader(val_dataset, batch_size=training_config.batch_size, num_workers=num_workers)

    model_config_entry = dict(models_config.get("baselines", {}).get(model_name, {}))
    if smoke_test:
        # Smoke tests must be fast, deterministic, and network-independent
        # (same philosophy as src/models/model_check.py) - never download
        # real ImageNet weights just to prove the training mechanics work.
        model_config_entry["pretrained"] = False
    effective_models_config = {**models_config, "baselines": {**models_config.get("baselines", {}), model_name: model_config_entry}}
    model = create_model(model_name, effective_models_config)
    architecture = model.architecture

    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    start_epoch = 0
    initial_best_metric = None
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        assert_checkpoint_compatible(checkpoint, model_name, num_classes)
        state = restore_training_state(checkpoint, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        initial_best_metric = state["best_metric"]

    git_commit = get_git_commit()
    config_hash = hash_config(
        {"dataset": dataset_config, "models": models_config, "training": training_config_dict}
    )

    logger = RunLogger(run_dir)
    logger.write_config(
        {
            "model_name": model_name,
            "architecture": architecture,
            "seed": seed,
            "smoke_test": smoke_test,
            "device": str(device),
            "training": training_config_dict,
            "model": model_config_entry,
        }
    )
    logger.write_environment(collect_environment_info())
    logger.write_dataset_hash(dataset_manifest_hash)
    logger.write_git_commit(git_commit)

    logger.log(f"run_id={run_id} model={model_name} architecture={architecture} seed={seed} smoke_test={smoke_test}")
    logger.log(f"device={device} cuda_available={torch.cuda.is_available()}")
    logger.log(f"train_manifest={train_manifest_path_for_log} train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    logger.log(
        f"mixed_precision_requested={training_config.mixed_precision} "
        f"amp_enabled={training_config.mixed_precision and device.type == 'cuda'} "
        f"(AMP is only ever active on CUDA)"
    )
    logger.log(f"gradient_clip_norm={training_config.gradient_clip_norm} gradient_accumulation_steps={training_config.gradient_accumulation_steps}")
    if resume_from is not None:
        logger.log(f"resumed from checkpoint: {resume_from} (start_epoch={start_epoch})")

    register_run(
        experiment_id=run_id,
        model=model_name,
        seed=seed,
        config=str(training_config_path),
        checkpoint="",
        test_result="",
        status=STATUS_RUNNING,
        notes="smoke_test" if smoke_test else run_notes,
        registry_path=registry_path,
    )

    best_checkpoint_path: Optional[Path] = None
    latest_checkpoint_path: Optional[Path] = None

    def on_epoch_end(epoch_result) -> None:
        nonlocal best_checkpoint_path, latest_checkpoint_path
        logger.log(
            f"epoch={epoch_result.epoch} "
            f"train_loss={epoch_result.train_metrics['loss']:.4f} "
            f"train_acc={epoch_result.train_metrics['accuracy']:.4f} "
            f"train_macro_f1={epoch_result.train_metrics['macro_f1']:.4f} "
            f"val_loss={epoch_result.val_metrics['loss']:.4f} "
            f"val_acc={epoch_result.val_metrics['accuracy']:.4f} "
            f"val_balanced_acc={epoch_result.val_metrics['balanced_accuracy']:.4f} "
            f"val_macro_f1={epoch_result.val_metrics['macro_f1']:.4f} "
            f"lr={epoch_result.lr:.6g} elapsed={epoch_result.elapsed_seconds:.1f}s"
        )
        logger.log_metrics(
            {
                "epoch": epoch_result.epoch,
                "train_loss": epoch_result.train_metrics["loss"],
                "train_accuracy": epoch_result.train_metrics["accuracy"],
                "train_macro_f1": epoch_result.train_metrics["macro_f1"],
                "val_loss": epoch_result.val_metrics["loss"],
                "val_accuracy": epoch_result.val_metrics["accuracy"],
                "val_balanced_accuracy": epoch_result.val_metrics["balanced_accuracy"],
                "val_macro_f1": epoch_result.val_metrics["macro_f1"],
                "lr": epoch_result.lr,
                "elapsed_seconds": epoch_result.elapsed_seconds,
                "is_best": epoch_result.is_best,
            }
        )

        checkpoint_kwargs = dict(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch_result.epoch,
            monitor_metric=training_config.monitor_metric,
            training_config=training_config_dict,
            seed=seed,
            model_name=model_name,
            architecture=architecture,
            num_classes=num_classes,
            dataset_manifest_hash=dataset_manifest_hash,
            git_commit=git_commit,
            scaler=trainer.scaler,
        )

        if epoch_result.is_best:
            best_checkpoint_path = checkpoint_dir / "best.pt"
            save_checkpoint(build_checkpoint(best_metric=epoch_result.monitor_value, **checkpoint_kwargs), best_checkpoint_path)
            logger.log(f"saved best checkpoint -> {best_checkpoint_path} ({training_config.monitor_metric}={epoch_result.monitor_value:.4f})")

        if training_config.checkpoint_frequency > 0 and (epoch_result.epoch + 1) % training_config.checkpoint_frequency == 0:
            latest_checkpoint_path = checkpoint_dir / "latest.pt"
            save_checkpoint(build_checkpoint(best_metric=epoch_result.monitor_value, **checkpoint_kwargs), latest_checkpoint_path)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        device=device,
        on_epoch_end=on_epoch_end,
        start_epoch=start_epoch,
        initial_best_metric=initial_best_metric,
    )

    # The scaler is only constructed inside Trainer.__init__, so its state
    # (relevant only for CUDA + mixed_precision; a no-op restore on CPU)
    # can only be restored after Trainer exists - unlike model/optimizer/
    # scheduler above, which are restored before Trainer wraps them.
    if resume_from is not None and checkpoint.get("scaler_state_dict") is not None:
        trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    try:
        fit_result = trainer.fit(
            max_train_steps_per_epoch=max_train_steps_per_epoch, max_val_steps_per_epoch=max_val_steps_per_epoch
        )
    except Exception as e:
        logger.log(f"RUN FAILED: {e}")
        update_run(run_id, registry_path=registry_path, status=STATUS_FAILED, notes=f"{run_notes} error={e}".strip())
        logger.close()
        raise

    logger.log(
        f"training finished: stopped_epoch={fit_result.stopped_epoch} best_epoch={fit_result.best_epoch} "
        f"best_metric={fit_result.best_metric} reason='{fit_result.stopping_reason}'"
    )
    update_run(
        run_id,
        registry_path=registry_path,
        status=STATUS_COMPLETED,
        checkpoint=str(best_checkpoint_path) if best_checkpoint_path else "",
        notes="smoke_test" if smoke_test else run_notes,
    )
    logger.close()

    return BaselineRunSummary(
        run_id=run_id,
        model_name=model_name,
        architecture=architecture,
        seed=seed,
        device=str(device),
        amp_enabled=trainer.amp_enabled,
        smoke_test=smoke_test,
        run_dir=run_dir,
        checkpoint_dir=checkpoint_dir,
        best_checkpoint_path=best_checkpoint_path,
        latest_checkpoint_path=latest_checkpoint_path,
        fit_result=fit_result,
        train_samples=len(train_dataset),
        val_samples=len(val_dataset),
        dataset_manifest_hash=dataset_manifest_hash,
        config_hash=config_hash,
        git_commit=git_commit,
    )

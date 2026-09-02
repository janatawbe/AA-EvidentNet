"""Orchestrates one AA-EvidentNet training run: loads config, builds the
dataset/dataloaders/model/optimizer/scheduler/combined-objective, drives
src.training.trainer.Trainer, and wires up checkpointing/logging/the
experiment registry - mirroring src.training.run_baseline.run_baseline_training
exactly (same manifests, same checkpoint/logging/registry infrastructure,
same test-set lock), with two differences specific to the proposed model:

  1. The model is AA-EvidentNet (src.models.aa_evidentnet, via
     create_model("aa_evidentnet", ...) reading configs/models.yaml:
     proposed.aa_evidentnet), not one of the three baselines.
  2. The criterion is the combined classification + CS-SupCon + EDL
     objective (src.losses.combined.CombinedAAEvidentNetLoss), which needs
     the model's full return_features=True output (logits, embedding,
     dirichlet_alpha) rather than logits alone. The only Trainer change
     this required was the `return_features` constructor flag added
     alongside this module - Trainer's train/validate loop, optimizer/
     scheduler/AMP/gradient-accumulation/clipping/early-stopping mechanics
     are entirely unmodified and unaware of AA-EvidentNet's existence.

Test set lock: this module never imports, loads, or references
test_original.csv anywhere - identical policy to run_baseline.py. Model
selection (early stopping, "best" checkpoint, LR scheduling) is driven
exclusively by validation metrics.

Learned class-level ambiguity (feature/learned-ambiguity, Phase 1,
src/losses/ambiguity.py + src/training/ambiguity_setup.py): when
configs/losses.yaml: cs_supcon.ambiguity_source="learned_class", this
module builds a frozen class-ambiguity matrix from an existing reference
checkpoint's embeddings over train_original.csv ONCE, before `optimizer`/
`scheduler`/`Trainer` are ever constructed, and installs it into
`criterion.cs_supcon_loss`. `Trainer` itself is not modified for this -
see build_learned_class_ambiguity's own docstring for the full method.
The default `ambiguity_source="fixed_pairs"` performs none of this and
requires no reference checkpoint - existing behavior is unchanged.
"""

import json
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from src.data.dataloaders import build_eval_dataloader, build_train_dataloader
from src.data.dataset import RetinalDataset
from src.data.transforms import build_transforms_from_config
from src.losses.ambiguity import load_ambiguity_settings
from src.losses.combined import CombinedAAEvidentNetLoss, build_combined_aa_evidentnet_loss
from src.models.factory import create_model
from src.training.ambiguity_setup import AmbiguitySetupError, build_learned_class_ambiguity
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

MODEL_NAME = "aa_evidentnet"

# Smoke-test defaults - identical rationale/values to run_baseline.py's
# (kept as a separate, small copy rather than importing a private helper
# from that module, so this file does not depend on run_baseline.py's
# internals; run_baseline.py itself is unmodified by Task 7 completion).
SMOKE_TEST_BATCH_SIZE = 4
SMOKE_TEST_EPOCHS = 2
SMOKE_TEST_N_TRAIN = 8
SMOKE_TEST_N_VAL = 4


class RunAAEvidentNetError(Exception):
    """Fatal, unrecoverable problem starting/running an AA-EvidentNet
    training run (missing manifests, incompatible resume checkpoint,
    invalid combined-objective configuration)."""


@dataclass
class AAEvidentNetRunSummary:
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
    cs_supcon_enabled: bool
    edl_enabled: bool
    ambiguity_source: str
    ambiguity_metadata_path: Optional[Path] = None


def _make_smoke_dataset(canonical_classes: List[str], raw_dir: Path, seed: int, n_train: int = 8, n_val: int = 4):
    """A tiny, fully self-contained synthetic dataset for the smoke test -
    never touches the real dataset. See run_baseline._make_smoke_dataset
    for the identical original; duplicated here (not imported) so this
    module has no dependency on run_baseline.py's internals."""
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
    if device.type == "cpu":
        return 0
    return config_num_workers


def run_aa_evidentnet_training(
    dataset_config_path: Union[str, Path] = "configs/dataset.yaml",
    models_config_path: Union[str, Path] = "configs/models.yaml",
    training_config_path: Union[str, Path] = "configs/training.yaml",
    losses_config_path: Union[str, Path] = "configs/losses.yaml",
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
) -> AAEvidentNetRunSummary:
    set_seed(seed)

    dataset_config = load_config(dataset_config_path)
    models_config = load_config(models_config_path)
    training_config_dict = load_config(training_config_path)
    losses_config = load_config(losses_config_path)

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

    run_id = generate_run_id(MODEL_NAME, seed, smoke_test=smoke_test)
    run_dir = Path(training_config_dict.get("logging", {}).get("log_dir", "results/logs")) / run_id
    checkpoint_dir = Path(training_config_dict.get("checkpointing", {}).get("save_dir", "results/checkpoints")) / run_id

    if smoke_test:
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
            raise RunAAEvidentNetError(
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

    model_config_entry = dict(models_config.get("proposed", {}).get(MODEL_NAME, {}))
    if not model_config_entry:
        raise RunAAEvidentNetError(f"configs/models.yaml: proposed.{MODEL_NAME} is missing or empty")
    if smoke_test:
        # Same offline/deterministic/no-network-access policy as
        # run_baseline.py's smoke test and src/models/model_check.py.
        model_config_entry = {**model_config_entry, "pretrained": False}
    effective_models_config = {**models_config, "proposed": {**models_config.get("proposed", {}), MODEL_NAME: model_config_entry}}
    model = create_model(MODEL_NAME, effective_models_config)
    architecture = model.architecture

    try:
        criterion = build_combined_aa_evidentnet_loss(losses_config, canonical_classes)
    except Exception as e:  # noqa: BLE001 - re-raise as a project-specific, clearer error
        raise RunAAEvidentNetError(f"Failed to build the combined AA-EvidentNet training objective: {e}") from e

    # --- learned class-level ambiguity (feature/learned-ambiguity, Phase
    # 1): only engaged when configs/losses.yaml: cs_supcon.ambiguity_source
    # is 'learned_class' AND CS-SupCon is actually enabled - otherwise this
    # is a complete no-op (no reference checkpoint required, no prototype
    # construction performed), preserving 'fixed_pairs' behavior exactly.
    # Runs entirely BEFORE the optimizer/scheduler/Trainer below are ever
    # constructed - Trainer itself is untouched by this mechanism. ---
    ambiguity_settings = load_ambiguity_settings(losses_config.get("cs_supcon", {}) or {})
    ambiguity_metadata: Optional[Dict[str, Any]] = None
    if criterion.cs_supcon_loss is not None and ambiguity_settings.ambiguity_source == "learned_class":
        if smoke_test:
            raise RunAAEvidentNetError(
                "cs_supcon.ambiguity_source='learned_class' is not supported with smoke_test=True - it "
                "requires a real train_original.csv manifest and a real reference checkpoint, neither of "
                "which the synthetic smoke-test dataset provides. Use smoke_test=False, or set "
                "ambiguity_source='fixed_pairs' for a smoke test."
            )
        train_original_manifest_path = manifests_dir / "train_original.csv"
        try:
            ambiguity_artifact = build_learned_class_ambiguity(
                reference_checkpoint_path=ambiguity_settings.reference_checkpoint_path,
                reference_model_name=ambiguity_settings.reference_model_name,
                models_config=models_config,
                dataset_config=dataset_config,
                canonical_classes=canonical_classes,
                raw_dir=raw_dir,
                processed_train_dir=processed_train_dir,
                train_manifest_path=train_original_manifest_path,
                device=device,
                batch_size=training_config.batch_size,
                num_workers=num_workers,
            )
        except Exception as e:  # noqa: BLE001 - re-raise as a project-specific, clearer error
            raise RunAAEvidentNetError(f"Failed to build the learned class-ambiguity matrix: {e}") from e

        criterion.cs_supcon_loss.set_learned_ambiguity_matrix(ambiguity_artifact.matrix_buffer)
        ambiguity_metadata = {
            "ambiguity_source": ambiguity_settings.ambiguity_source,
            "ambiguity_scale": ambiguity_settings.ambiguity_scale,
            "reference_checkpoint_path": ambiguity_artifact.reference_checkpoint_path,
            "reference_checkpoint_sha256": ambiguity_artifact.reference_checkpoint_sha256,
            "reference_model_name": ambiguity_artifact.reference_model_name,
            "reference_checkpoint_architecture": ambiguity_artifact.reference_checkpoint_architecture,
            "train_manifest_path": ambiguity_artifact.train_manifest_path,
            "train_manifest_sha256": ambiguity_artifact.train_manifest_sha256,
            "num_train_samples": ambiguity_artifact.num_train_samples,
            "class_sample_counts": ambiguity_artifact.class_sample_counts,
            "class_names": ambiguity_artifact.canonical_classes,
            "margin_normalization": {
                "margin_min": ambiguity_artifact.margin_normalization.margin_min,
                "margin_max": ambiguity_artifact.margin_normalization.margin_max,
                "fit_on": "train_original.csv",
            },
            "class_ambiguity_matrix": ambiguity_artifact.matrix_numpy.tolist(),
            "methodological_caveat": (
                "This reference checkpoint's embedding space was itself shaped by the EXISTING "
                "fixed-hard-pair CS-SupCon objective (ambiguity_weight on 3 clinician-picked pairs). The "
                "learned matrix therefore reflects class geometry AFTER that existing correction has "
                "already partially acted - it is not claimed to be a from-scratch, assumption-free "
                "measurement of natural class confusability, and is not claimed to be independent of the "
                "previous ambiguity mechanism."
            ),
        }

    optimizer = build_optimizer(model, training_config)
    scheduler = build_scheduler(optimizer, training_config)

    start_epoch = 0
    initial_best_metric = None
    if resume_from is not None:
        checkpoint = load_checkpoint(resume_from)
        assert_checkpoint_compatible(checkpoint, MODEL_NAME, num_classes)
        state = restore_training_state(checkpoint, model, optimizer, scheduler)
        start_epoch = state["epoch"] + 1
        initial_best_metric = state["best_metric"]

    git_commit = get_git_commit()
    config_hash = hash_config(
        {"dataset": dataset_config, "models": models_config, "training": training_config_dict, "losses": losses_config}
    )

    logger = RunLogger(run_dir)
    logger.write_config(
        {
            "model_name": MODEL_NAME,
            "architecture": architecture,
            "seed": seed,
            "smoke_test": smoke_test,
            "device": str(device),
            "training": training_config_dict,
            "model": model_config_entry,
            "losses": losses_config,
        }
    )
    logger.write_environment(collect_environment_info())
    logger.write_dataset_hash(dataset_manifest_hash)
    logger.write_git_commit(git_commit)

    logger.log(f"run_id={run_id} model={MODEL_NAME} architecture={architecture} seed={seed} smoke_test={smoke_test}")
    logger.log(f"device={device} cuda_available={torch.cuda.is_available()}")
    logger.log(f"train_manifest={train_manifest_path_for_log} train_samples={len(train_dataset)} val_samples={len(val_dataset)}")
    logger.log(
        f"mixed_precision_requested={training_config.mixed_precision} "
        f"amp_enabled={training_config.mixed_precision and device.type == 'cuda'} "
        f"(AMP is only ever active on CUDA)"
    )
    logger.log(f"gradient_clip_norm={training_config.gradient_clip_norm} gradient_accumulation_steps={training_config.gradient_accumulation_steps}")
    logger.log(
        f"combined_objective: cs_supcon_enabled={criterion.cs_supcon_loss is not None} "
        f"cs_supcon_weight={criterion.cs_supcon_weight} "
        f"edl_enabled={criterion.edl_loss_module is not None} edl_weight={criterion.edl_weight}"
    )
    ambiguity_metadata_path: Optional[Path] = None
    logger.log(f"ambiguity_source={ambiguity_settings.ambiguity_source}")
    if ambiguity_metadata is not None:
        logger.log(
            f"learned class-ambiguity matrix built from reference_checkpoint="
            f"{ambiguity_metadata['reference_checkpoint_path']} "
            f"(sha256={ambiguity_metadata['reference_checkpoint_sha256'][:12]}...) over "
            f"train_original.csv ({ambiguity_metadata['num_train_samples']} samples)"
        )
        ambiguity_metadata_path = run_dir / "ambiguity_metadata.json"
        with open(ambiguity_metadata_path, "w", encoding="utf-8") as f:
            json.dump(ambiguity_metadata, f, indent=2, sort_keys=True)
    if resume_from is not None:
        logger.log(f"resumed from checkpoint: {resume_from} (start_epoch={start_epoch})")

    register_run(
        experiment_id=run_id,
        model=MODEL_NAME,
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
            f"lr={epoch_result.lr:.6g} elapsed={epoch_result.elapsed_seconds:.1f}s "
            f"loss_components={criterion.last_components}"
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
                "loss_components": criterion.last_components,
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
            model_name=MODEL_NAME,
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
        criterion=criterion,
        return_features=True,
        on_epoch_end=on_epoch_end,
        start_epoch=start_epoch,
        initial_best_metric=initial_best_metric,
    )

    # The scaler is only constructed inside Trainer.__init__, so its state
    # (relevant only for CUDA + mixed_precision; a no-op restore on CPU)
    # can only be restored after Trainer exists.
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

    return AAEvidentNetRunSummary(
        run_id=run_id,
        model_name=MODEL_NAME,
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
        cs_supcon_enabled=criterion.cs_supcon_loss is not None,
        edl_enabled=criterion.edl_loss_module is not None,
        ambiguity_source=ambiguity_settings.ambiguity_source,
        ambiguity_metadata_path=ambiguity_metadata_path,
    )

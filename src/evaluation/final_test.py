"""Final held-out test evaluation for a single FROZEN, already-trained
checkpoint (Task 8).

This is the ONLY place in the codebase (besides its own regression test)
that is allowed to load data/manifests/test_original.csv at all --
src/training/run_baseline.py and src/training/run_aa_evidentnet.py never
reference it (see their own module docstrings, and the model-selection
policy note below).

Reuses, unmodified: the model factory (src/models/factory.py), checkpoint
loading/compatibility utilities (src/training/checkpointing.py),
RetinalDataset's test-manifest safeguards (src/data/dataset.py:
expected_split="test", require_all_original=True), and the multiclass
metrics in src/evaluation/metrics.py. Never retrains, never tunes never
touches the checkpoint's weights beyond loading them, and never uses the
test set for training/model-selection/hyperparameter/calibration/
threshold decisions -- this module only ever reads a manifest and a
checkpoint and writes results out.

No training engine (src/training/trainer.py: Trainer) is used here: it
exists specifically to drive an optimizer/scheduler through train+validate
epochs, which is the opposite of what a frozen-checkpoint evaluation
needs (no optimizer, no backward pass, no scheduler stepping ever
constructed anywhere in this module). Inference runs once, in plain
fp32 (deliberately no AMP/autocast, even on CUDA) under
`torch.inference_mode()`, so the exported logits/probabilities are the
same regardless of device -- appropriate for a "final, locked" result
that later calibration/statistical analyses will depend on.
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.records import write_csv
from src.data.transforms import build_transforms_from_config
from src.evaluation.metrics import EvaluationResult, evaluate_predictions
from src.models.factory import MODEL_NAMES, PROPOSED_MODEL_NAMES, create_model
from src.training.checkpointing import (
    assert_checkpoint_compatible,
    load_checkpoint,
    restore_training_state,
)
from src.training.logging import generate_run_id
from src.training.registry import DEFAULT_REGISTRY_PATH, load_registry, update_run
from src.training.trainer import resolve_device
from src.utils.config import hash_config, load_config
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file
from src.utils.seeding import DEFAULT_SEED, set_seed
from datetime import datetime, timezone

ALL_MODEL_NAMES = tuple(MODEL_NAMES) + tuple(PROPOSED_MODEL_NAMES)

PREDICTION_BASE_COLUMNS = [
    "sample_id",
    "image_path",
    "true_class_index",
    "true_class_name",
    "predicted_class_index",
    "predicted_class_name",
    "correct",
    "max_probability",
]

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 2


class FinalTestError(Exception):
    """Raised for a final-test-specific problem: an unknown model name, a
    missing test manifest, or an internal invariant violation (e.g. an
    exported prediction count that does not match the loaded manifest
    count). Checkpoint incompatibility is raised by the reused
    `assert_checkpoint_compatible` as `CheckpointIncompatibleError`, not
    wrapped here."""


@dataclass
class FinalTestSummary:
    eval_run_id: str
    model_name: str
    checkpoint_path: str
    checkpoint_hash: str
    training_run_id: Optional[str]
    device: str
    num_samples: int
    class_names: List[str]
    predictions_path: str
    overall_metrics_path: str
    per_class_metrics_path: str
    confusion_matrix_path: str
    metadata_path: str
    overall_metrics: Dict[str, Any]
    registry_updated: bool


def _class_indexed_columns(prefix: str, num_classes: int) -> List[str]:
    return [f"{prefix}_{i}" for i in range(num_classes)]


def _resolve_num_workers(device: torch.device, num_workers_override: Optional[int], config_num_workers: int) -> int:
    if num_workers_override is not None:
        return num_workers_override
    # Same policy as run_baseline.py/run_aa_evidentnet.py: 0 on CPU
    # (avoids paying multiprocessing worker start-up cost), configured
    # value on CUDA.
    if device.type == "cpu":
        return 0
    return config_num_workers


def _effective_model_config(model_name: str, models_config: Dict[str, Any]) -> Dict[str, Any]:
    """Same models.yaml the checkpoint was trained under, with
    pretrained forced to False -- final_test immediately overwrites every
    weight via restore_training_state(), so downloading real ImageNet
    weights first would be pure wasted network I/O (same policy as the
    existing --smoke-test paths and model_check.py)."""
    if model_name in PROPOSED_MODEL_NAMES:
        section = "proposed"
    else:
        section = "baselines"
    entry = dict(models_config.get(section, {}).get(model_name, {}))
    entry["pretrained"] = False
    return {**models_config, section: {**models_config.get(section, {}), model_name: entry}}


def run_final_test(
    model_name: str,
    checkpoint_path: Union[str, Path],
    dataset_config_path: Union[str, Path] = "configs/dataset.yaml",
    models_config_path: Union[str, Path] = "configs/models.yaml",
    evaluation_config_path: Union[str, Path] = "configs/evaluation.yaml",
    seed: int = DEFAULT_SEED,
    device_override: Optional[str] = None,
    num_workers_override: Optional[int] = None,
    batch_size_override: Optional[int] = None,
    registry_path: Union[str, Path] = DEFAULT_REGISTRY_PATH,
) -> FinalTestSummary:
    if model_name not in ALL_MODEL_NAMES:
        raise FinalTestError(f"Unknown model '{model_name}'. Known models: {list(ALL_MODEL_NAMES)}")

    set_seed(seed)

    dataset_config = load_config(dataset_config_path)
    models_config = load_config(models_config_path)
    evaluation_config = load_config(evaluation_config_path)

    final_test_cfg = evaluation_config.get("final_test", {}) or {}

    device = resolve_device(device_override or "auto")
    num_workers = _resolve_num_workers(
        device, num_workers_override, final_test_cfg.get("num_workers", DEFAULT_NUM_WORKERS)
    )

    canonical_classes = sorted(dataset_config["class_directory_mapping"].keys())
    num_classes = len(canonical_classes)
    _, eval_transform = build_transforms_from_config(dataset_config)

    raw_dir = Path(dataset_config["paths"]["raw_dir"])
    processed_train_dir = Path(dataset_config["paths"]["processed_dir"]) / "train"
    manifests_dir = Path(dataset_config["paths"]["manifests_dir"])
    test_manifest_path = manifests_dir / "test_original.csv"
    if not test_manifest_path.is_file():
        raise FinalTestError(f"{test_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first.")

    checkpoint_path = Path(checkpoint_path)
    checkpoint = load_checkpoint(checkpoint_path)
    assert_checkpoint_compatible(checkpoint, model_name, num_classes)
    checkpoint_hash = hash_file(checkpoint_path)
    checkpoint_metadata = checkpoint.get("metadata", {}) or {}
    # Convention, not a guaranteed field inside the checkpoint itself:
    # every checkpoint this project writes lives at
    # results/checkpoints/<run_id>/{best,latest}.pt (src/training/run_baseline.py,
    # run_aa_evidentnet.py) - the parent directory name IS the training
    # run_id. Recorded as "inferred" in metadata so this is never
    # mistaken for an authoritative field stored inside the checkpoint.
    training_run_id = checkpoint_path.parent.name

    effective_models_config = _effective_model_config(model_name, models_config)
    model = create_model(model_name, effective_models_config)
    restore_training_state(checkpoint, model)  # weights only - no optimizer/scheduler/scaler needed for eval
    model.to(device)
    model.eval()

    test_dataset = RetinalDataset.from_manifest(
        test_manifest_path,
        canonical_classes,
        raw_dir,
        processed_train_dir,
        transform=eval_transform,
        expected_split="test",
        require_all_original=True,
    )
    test_manifest_hash = hash_file(test_manifest_path)

    batch_size = batch_size_override if batch_size_override is not None else final_test_cfg.get("batch_size", DEFAULT_BATCH_SIZE)
    loader = build_eval_dataloader(test_dataset, batch_size=batch_size, num_workers=num_workers)

    is_aa_evidentnet = model_name == "aa_evidentnet"

    sample_rows: List[Dict[str, Any]] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    probabilities_all: List[List[float]] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]

            if is_aa_evidentnet:
                output = model(images, return_features=True)
                logits = output.logits
            else:
                logits = model(images)

            probabilities = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            logits_np = logits.detach().cpu().numpy()
            probs_np = probabilities.detach().cpu().numpy()
            preds_np = preds.detach().cpu().numpy()
            labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else labels

            if is_aa_evidentnet:
                evidence_np = output.evidence.detach().cpu().numpy()
                alpha_np = output.dirichlet_alpha.detach().cpu().numpy()
                evidential_prob_np = output.probabilities.detach().cpu().numpy()
                uncertainty_np = output.uncertainty.detach().cpu().numpy()

            for i in range(images.size(0)):
                true_idx = int(labels_np[i])
                pred_idx = int(preds_np[i])
                row: Dict[str, Any] = {
                    "sample_id": batch["original_id"][i],
                    "image_path": batch["image_path"][i],
                    "true_class_index": true_idx,
                    "true_class_name": canonical_classes[true_idx],
                    "predicted_class_index": pred_idx,
                    "predicted_class_name": canonical_classes[pred_idx],
                    "correct": true_idx == pred_idx,
                    "max_probability": float(probs_np[i].max()),
                }
                for k in range(num_classes):
                    row[f"logit_{k}"] = float(logits_np[i, k])
                for k in range(num_classes):
                    row[f"prob_{k}"] = float(probs_np[i, k])
                if is_aa_evidentnet:
                    for k in range(num_classes):
                        row[f"evidence_{k}"] = float(evidence_np[i, k])
                    for k in range(num_classes):
                        row[f"dirichlet_alpha_{k}"] = float(alpha_np[i, k])
                    for k in range(num_classes):
                        row[f"evidential_prob_{k}"] = float(evidential_prob_np[i, k])
                    row["uncertainty"] = float(uncertainty_np[i])

                sample_rows.append(row)
                y_true.append(true_idx)
                y_pred.append(pred_idx)
                probabilities_all.append(probs_np[i].tolist())

    # --- integrity checks (Task 8 item 5): every sample exactly once,
    # unique IDs, exported count matches the loaded manifest exactly. ---
    if len(sample_rows) != len(test_dataset):
        raise FinalTestError(
            f"exported prediction count ({len(sample_rows)}) != loaded test manifest count ({len(test_dataset)})"
        )
    sample_ids = [row["sample_id"] for row in sample_rows]
    if len(set(sample_ids)) != len(sample_ids):
        raise FinalTestError("duplicate sample_id values found among exported predictions - refusing to write output")

    evaluation_result: EvaluationResult = evaluate_predictions(y_true, y_pred, probabilities_all, num_classes)

    # --- outputs ---
    eval_run_id = generate_run_id(f"finaltest_{model_name}", seed, smoke_test=False)
    output_paths_cfg = evaluation_config.get("output_paths", {}) or {}
    raw_predictions_dir = Path(output_paths_cfg.get("raw_predictions_dir", "results/raw_predictions")) / eval_run_id
    tables_dir = Path(output_paths_cfg.get("tables_dir", "results/tables")) / eval_run_id

    predictions_path = raw_predictions_dir / "predictions.csv"
    prediction_columns = list(PREDICTION_BASE_COLUMNS)
    prediction_columns += _class_indexed_columns("logit", num_classes)
    prediction_columns += _class_indexed_columns("prob", num_classes)
    if is_aa_evidentnet:
        prediction_columns += _class_indexed_columns("evidence", num_classes)
        prediction_columns += _class_indexed_columns("dirichlet_alpha", num_classes)
        prediction_columns += _class_indexed_columns("evidential_prob", num_classes)
        prediction_columns += ["uncertainty"]
    write_csv(sample_rows, prediction_columns, predictions_path)

    overall_metrics_path = tables_dir / "overall_metrics.json"
    overall_metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(overall_metrics_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_result.overall, f, indent=2, sort_keys=True)

    per_class_metrics_path = tables_dir / "per_class_metrics.csv"
    per_class_rows = [{"class_name": canonical_classes[row["class_index"]], **row} for row in evaluation_result.per_class]
    per_class_columns = ["class_name"] + list(evaluation_result.per_class[0].keys())
    write_csv(per_class_rows, per_class_columns, per_class_metrics_path)

    confusion_matrix_path = tables_dir / "confusion_matrix.csv"
    cm = evaluation_result.confusion_matrix
    cm_columns = ["true_class_index", "true_class_name"] + [f"pred_{name}" for name in canonical_classes]
    cm_rows = []
    for k in range(num_classes):
        cm_row = {"true_class_index": k, "true_class_name": canonical_classes[k]}
        for j in range(num_classes):
            cm_row[f"pred_{canonical_classes[j]}"] = int(cm[k, j])
        cm_rows.append(cm_row)
    write_csv(cm_rows, cm_columns, confusion_matrix_path)

    config_hash = hash_config({"dataset": dataset_config, "models": models_config, "evaluation": evaluation_config})
    metadata = {
        "eval_run_id": eval_run_id,
        "model_name": model_name,
        "model_architecture": checkpoint_metadata.get("architecture"),
        "training_run_id_inferred_from_checkpoint_path": training_run_id,
        "training_seed": checkpoint_metadata.get("seed"),
        "eval_invocation_seed": seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_best_epoch": checkpoint_metadata.get("epoch"),
        "checkpoint_monitor_metric": checkpoint_metadata.get("monitor_metric"),
        "checkpoint_best_metric": checkpoint_metadata.get("best_metric"),
        "test_manifest_path": str(test_manifest_path),
        "test_manifest_sha256": test_manifest_hash,
        "class_names": canonical_classes,
        "num_classes": num_classes,
        "num_evaluated_samples": len(sample_rows),
        "dataset_config_path": str(dataset_config_path),
        "models_config_path": str(models_config_path),
        "evaluation_config_path": str(evaluation_config_path),
        "config_hash": config_hash,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "predictions_path": str(predictions_path),
        "overall_metrics_path": str(overall_metrics_path),
        "per_class_metrics_path": str(per_class_metrics_path),
        "confusion_matrix_path": str(confusion_matrix_path),
    }
    metadata_path = tables_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    # --- registry: best-effort update of the ORIGINAL TRAINING run's row
    # only (never creates a new row - see module docstring / Task 8 item
    # 11). Only reached after every output above was written successfully. ---
    registry_updated = False
    existing_rows = load_registry(registry_path)
    if any(r.get("experiment_id") == training_run_id for r in existing_rows):
        macro_f1 = evaluation_result.overall.get("macro_f1")
        accuracy = evaluation_result.overall.get("accuracy")
        summary_str = (
            f"macro_f1={macro_f1};accuracy={accuracy};n={evaluation_result.overall.get('num_samples')};"
            f"predictions={predictions_path}"
        )
        update_run(training_run_id, registry_path=registry_path, test_result=summary_str)
        registry_updated = True

    return FinalTestSummary(
        eval_run_id=eval_run_id,
        model_name=model_name,
        checkpoint_path=str(checkpoint_path),
        checkpoint_hash=checkpoint_hash,
        training_run_id=training_run_id,
        device=str(device),
        num_samples=len(sample_rows),
        class_names=canonical_classes,
        predictions_path=str(predictions_path),
        overall_metrics_path=str(overall_metrics_path),
        per_class_metrics_path=str(per_class_metrics_path),
        confusion_matrix_path=str(confusion_matrix_path),
        metadata_path=str(metadata_path),
        overall_metrics=evaluation_result.overall,
        registry_updated=registry_updated,
    )

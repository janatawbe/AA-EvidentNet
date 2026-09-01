"""Validation-only analysis of the learned sample-level ambiguity signal
(Phase 1: analysis only - see src/losses/ambiguity.py and
src/training/ambiguity_setup.py).

Answers the question "is sample ambiguity actually meaningful?" using
ONLY data/manifests/val_original.csv and the SAME frozen reference
representation (prototypes, class-ambiguity matrix, margin normalization)
already built by src.training.ambiguity_setup.build_learned_class_ambiguity
from train_original.csv. This module never reads, requires, or depends on
data/manifests/test_original.csv in any way - it does not even import
anything that could load it.

Seven analyses (see REPRODUCIBILITY.md for the full discussion of each):

  1. Error-detection AUROC/AUPRC of `is_error` (val prediction wrong) vs.
     sample ambiguity - reuses src.evaluation.ood_uncertainty's own
     `_error_detection_auroc`/`_error_detection_auprc` (same-package
     cross-import, same convention final_test.py/robustness.py already
     use for each other).
  2. Competing-class hit rate: among misclassified val samples, how often
     does `competing_class` (the sample's 2nd-nearest prototype) equal the
     model's actual (wrong) predicted class?
  3. Ambiguity distribution (mean/median) for correct vs. incorrect val
     predictions.
  4. Ambiguity distribution (mean/median) for val samples whose TRUE class
     is one of the existing fixed hard-pair classes vs. all other classes.
  5. Spearman correlation between sample ambiguity and the model's own EDL
     uncertainty on the same val samples.
  6. A 2x2 quadrant breakdown (median-split ambiguity x median-split EDL
     uncertainty) reporting the val error rate within each quadrant -
     operationalizing "low/high ambiguity x low/high uncertainty."
  7. Ranking comparison ONLY (never a construction input) between the
     class-ambiguity matrix's off-diagonal entries and an ordinary
     validation confusion matrix's off-diagonal entries (symmetrized) -
     the confusion matrix is used here strictly to sanity-check agreement
     with the learned matrix; the learned matrix itself is never built
     from it (see src/losses/ambiguity.py's module docstring).

Outputs are written only under results/ambiguity/<run_id>/ - a separate
directory from results/raw_predictions/, results/tables/, results/
robustness/, and results/ood_uncertainty/.
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from src.data.dataloaders import build_eval_dataloader
from src.data.dataset import RetinalDataset
from src.data.records import write_csv
from src.data.transforms import build_transforms_from_config
from src.evaluation.metrics import compute_confusion_matrix
from src.evaluation.ood_uncertainty import _error_detection_auprc, _error_detection_auroc, _spearman
from src.losses.ambiguity import compute_sample_ambiguity
from src.losses.cs_supcon import resolve_ambiguity_pairs
from src.models.factory import create_model
from src.training.ambiguity_setup import LearnedAmbiguityArtifact
from src.training.checkpointing import assert_checkpoint_compatible, load_checkpoint, restore_training_state
from src.training.logging import generate_run_id
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0


class AmbiguityValidationError(Exception):
    """Raised for a problem running validation-only ambiguity analysis: a
    missing val_original.csv, or an internal invariant violation."""


@dataclass
class AmbiguityValidationSummary:
    run_id: str
    num_val_samples: int
    error_detection_auroc: Optional[float]
    error_detection_auprc: Optional[float]
    competing_class_hit_rate_among_errors: Optional[float]
    num_errors: int
    ambiguity_mean_correct: Optional[float]
    ambiguity_median_correct: Optional[float]
    ambiguity_mean_incorrect: Optional[float]
    ambiguity_median_incorrect: Optional[float]
    ambiguity_mean_hard_pair_classes: Optional[float]
    ambiguity_median_hard_pair_classes: Optional[float]
    ambiguity_mean_other_classes: Optional[float]
    ambiguity_median_other_classes: Optional[float]
    spearman_ambiguity_vs_edl_uncertainty: Optional[float]
    quadrant_error_rates: Dict[str, Optional[float]]
    quadrant_counts: Dict[str, int]
    class_matrix_vs_confusion_spearman: Optional[float]
    top5_learned_ambiguity_pairs: List[Dict[str, Any]]
    top5_confusion_pairs: List[Dict[str, Any]]
    matrix_path: str
    metrics_path: str
    metadata_path: str


def _median_or_none(values: np.ndarray) -> Optional[float]:
    return float(np.median(values)) if values.size > 0 else None


def _mean_or_none(values: np.ndarray) -> Optional[float]:
    return float(np.mean(values)) if values.size > 0 else None


def _offdiagonal_pairs(matrix: np.ndarray, canonical_classes: Sequence[str]) -> List[Tuple[int, int, float]]:
    """(a, b, value) for every a < b, from a symmetric [K, K] matrix."""
    num_classes = matrix.shape[0]
    pairs = []
    for a in range(num_classes):
        for b in range(a + 1, num_classes):
            pairs.append((a, b, float(matrix[a, b])))
    return pairs


def _top_n_pairs(pairs: List[Tuple[int, int, float]], canonical_classes: Sequence[str], n: int = 5) -> List[Dict[str, Any]]:
    ranked = sorted(pairs, key=lambda p: p[2], reverse=True)[:n]
    return [
        {"class_a": canonical_classes[a], "class_b": canonical_classes[b], "value": value}
        for a, b, value in ranked
    ]


def run_ambiguity_validation(
    artifact: LearnedAmbiguityArtifact,
    val_manifest_path: Union[str, Path],
    dataset_config: Dict[str, Any],
    models_config: Dict[str, Any],
    raw_dir: Union[str, Path],
    processed_train_dir: Union[str, Path],
    device: torch.device,
    hard_pair_class_names: Optional[Sequence[Sequence[str]]] = None,
    output_dir: Union[str, Path] = "results/ambiguity",
    batch_size: int = DEFAULT_BATCH_SIZE,
    num_workers: int = DEFAULT_NUM_WORKERS,
) -> AmbiguityValidationSummary:
    """Run all 7 validation-only analyses against `val_manifest_path`
    (data/manifests/val_original.csv) using the SAME reference checkpoint
    and frozen prototypes/matrix/margin-normalization already built by
    `src.training.ambiguity_setup.build_learned_class_ambiguity` (passed
    in as `artifact` - this function does not recompute them, and does
    not re-touch train_original.csv). `hard_pair_class_names` defaults to
    the existing fixed hard pairs (Healthy/Glaucoma, Disc Edema/Glaucoma,
    Diabetic Retinopathy/Central Serous Chorioretinopathy) if not given.

    Never reads, requires, or references data/manifests/test_original.csv.
    """
    canonical_classes = artifact.canonical_classes
    num_classes = len(canonical_classes)
    val_manifest_path = Path(val_manifest_path)
    if not val_manifest_path.is_file():
        raise AmbiguityValidationError(
            f"{val_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first."
        )

    if hard_pair_class_names is None:
        hard_pair_class_names = [
            ["Healthy", "Glaucoma"],
            ["Disc Edema", "Glaucoma"],
            ["Diabetic Retinopathy", "Central Serous Chorioretinopathy"],
        ]
    hard_pairs = resolve_ambiguity_pairs(
        [p for p in hard_pair_class_names if p[0] in canonical_classes and p[1] in canonical_classes],
        canonical_classes,
    )
    hard_pair_class_indices = set()
    for pair in hard_pairs.pairs:
        hard_pair_class_indices.update(pair)

    # Reload the SAME reference checkpoint into its own, throwaway model
    # instance - never the model being trained, read-only, no gradients.
    checkpoint = load_checkpoint(artifact.reference_checkpoint_path)
    assert_checkpoint_compatible(checkpoint, artifact.reference_model_name, num_classes)
    reference_model = create_model(artifact.reference_model_name, models_config)
    restore_training_state(checkpoint, reference_model)
    reference_model.to(device)
    reference_model.eval()

    _, eval_transform = build_transforms_from_config(dataset_config)
    val_dataset = RetinalDataset.from_manifest(
        val_manifest_path,
        canonical_classes,
        raw_dir,
        processed_train_dir,
        transform=eval_transform,
        expected_split="val",
        require_all_original=True,
    )
    loader = build_eval_dataloader(val_dataset, batch_size=batch_size, num_workers=num_workers)

    embeddings_chunks: List[np.ndarray] = []
    uncertainty_chunks: List[np.ndarray] = []
    y_true: List[int] = []
    y_pred: List[int] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"]
            labels_np = labels.detach().cpu().numpy() if torch.is_tensor(labels) else np.asarray(labels)

            output = reference_model(images, return_features=True)
            preds = torch.argmax(output.logits, dim=1).detach().cpu().numpy()

            embeddings_chunks.append(output.embedding.detach().cpu().numpy())
            uncertainty_chunks.append(output.uncertainty.detach().cpu().numpy())
            y_true.extend(int(v) for v in labels_np)
            y_pred.extend(int(v) for v in preds)

    embeddings = np.concatenate(embeddings_chunks, axis=0)
    edl_uncertainty = np.concatenate(uncertainty_chunks, axis=0)
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    is_error = (y_true_arr != y_pred_arr).astype(np.int64)

    sample_result = compute_sample_ambiguity(embeddings, artifact.prototypes, artifact.margin_normalization)
    ambiguity = sample_result.ambiguity
    competing_class = sample_result.competing_class

    # --- 1. error-detection AUROC/AUPRC ---
    auroc = _error_detection_auroc(is_error, ambiguity)
    auprc = _error_detection_auprc(is_error, ambiguity)

    # --- 2. competing-class hit rate among errors ---
    error_mask = is_error.astype(bool)
    num_errors = int(error_mask.sum())
    if num_errors > 0:
        hit_rate = float((competing_class[error_mask] == y_pred_arr[error_mask]).mean())
    else:
        hit_rate = None

    # --- 3. ambiguity: correct vs incorrect ---
    correct_ambiguity = ambiguity[~error_mask]
    incorrect_ambiguity = ambiguity[error_mask]

    # --- 4. ambiguity: hard-pair classes vs other classes (by TRUE class) ---
    hard_pair_mask = np.array([label in hard_pair_class_indices for label in y_true_arr])
    hard_pair_ambiguity = ambiguity[hard_pair_mask]
    other_ambiguity = ambiguity[~hard_pair_mask]

    # --- 5. Spearman(ambiguity, EDL uncertainty) ---
    spearman_edl = _spearman(ambiguity.tolist(), edl_uncertainty.tolist())

    # --- 6. quadrants: median-split ambiguity x median-split EDL uncertainty ---
    ambiguity_median = float(np.median(ambiguity)) if ambiguity.size > 0 else 0.0
    uncertainty_median = float(np.median(edl_uncertainty)) if edl_uncertainty.size > 0 else 0.0
    quadrant_error_rates: Dict[str, Optional[float]] = {}
    quadrant_counts: Dict[str, int] = {}
    for amb_label, amb_mask in (("low_ambiguity", ambiguity <= ambiguity_median), ("high_ambiguity", ambiguity > ambiguity_median)):
        for unc_label, unc_mask in (
            ("low_uncertainty", edl_uncertainty <= uncertainty_median),
            ("high_uncertainty", edl_uncertainty > uncertainty_median),
        ):
            quadrant_mask = amb_mask & unc_mask
            key = f"{amb_label}_{unc_label}"
            count = int(quadrant_mask.sum())
            quadrant_counts[key] = count
            quadrant_error_rates[key] = float(is_error[quadrant_mask].mean()) if count > 0 else None

    # --- 7. class matrix vs. observed val confusion (comparison only) ---
    confusion = compute_confusion_matrix(y_true_arr.tolist(), y_pred_arr.tolist(), num_classes)
    confusion_symmetric = confusion.astype(np.float64) + confusion.astype(np.float64).T
    np.fill_diagonal(confusion_symmetric, 0.0)

    ambiguity_pairs_ranked = _offdiagonal_pairs(artifact.matrix_numpy, canonical_classes)
    confusion_pairs_ranked = _offdiagonal_pairs(confusion_symmetric, canonical_classes)
    ambiguity_values = [p[2] for p in ambiguity_pairs_ranked]
    confusion_values = [p[2] for p in confusion_pairs_ranked]
    class_matrix_vs_confusion_spearman = _spearman(ambiguity_values, confusion_values)

    top5_ambiguity = _top_n_pairs(ambiguity_pairs_ranked, canonical_classes)
    top5_confusion = _top_n_pairs(confusion_pairs_ranked, canonical_classes)

    # --- outputs ---
    run_id = generate_run_id("ambiguity_validation", seed=0, smoke_test=False)
    base_dir = Path(output_dir) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = base_dir / "class_ambiguity_matrix.csv"
    matrix_columns = ["class_name"] + list(canonical_classes)
    matrix_rows = []
    for i, name in enumerate(canonical_classes):
        row = {"class_name": name}
        for j, other_name in enumerate(canonical_classes):
            row[other_name] = float(artifact.matrix_numpy[i, j])
        matrix_rows.append(row)
    write_csv(matrix_rows, matrix_columns, matrix_path)

    summary = AmbiguityValidationSummary(
        run_id=run_id,
        num_val_samples=len(val_dataset),
        error_detection_auroc=auroc,
        error_detection_auprc=auprc,
        competing_class_hit_rate_among_errors=hit_rate,
        num_errors=num_errors,
        ambiguity_mean_correct=_mean_or_none(correct_ambiguity),
        ambiguity_median_correct=_median_or_none(correct_ambiguity),
        ambiguity_mean_incorrect=_mean_or_none(incorrect_ambiguity),
        ambiguity_median_incorrect=_median_or_none(incorrect_ambiguity),
        ambiguity_mean_hard_pair_classes=_mean_or_none(hard_pair_ambiguity),
        ambiguity_median_hard_pair_classes=_median_or_none(hard_pair_ambiguity),
        ambiguity_mean_other_classes=_mean_or_none(other_ambiguity),
        ambiguity_median_other_classes=_median_or_none(other_ambiguity),
        spearman_ambiguity_vs_edl_uncertainty=spearman_edl,
        quadrant_error_rates=quadrant_error_rates,
        quadrant_counts=quadrant_counts,
        class_matrix_vs_confusion_spearman=class_matrix_vs_confusion_spearman,
        top5_learned_ambiguity_pairs=top5_ambiguity,
        top5_confusion_pairs=top5_confusion,
        matrix_path=str(matrix_path),
        metrics_path=str(base_dir / "validation_metrics.json"),
        metadata_path=str(base_dir / "metadata.json"),
    )

    with open(summary.metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, sort_keys=True)

    metadata = {
        "run_id": run_id,
        "reference_checkpoint_path": artifact.reference_checkpoint_path,
        "reference_checkpoint_sha256": artifact.reference_checkpoint_sha256,
        "reference_model_name": artifact.reference_model_name,
        "class_names": canonical_classes,
        "num_classes": num_classes,
        "class_sample_counts_train": artifact.class_sample_counts,
        "num_train_samples": artifact.num_train_samples,
        "train_manifest_path": artifact.train_manifest_path,
        "train_manifest_sha256": artifact.train_manifest_sha256,
        "val_manifest_path": str(val_manifest_path),
        "val_manifest_sha256": hash_file(val_manifest_path),
        "num_val_samples": len(val_dataset),
        "margin_normalization": {
            "margin_min": artifact.margin_normalization.margin_min,
            "margin_max": artifact.margin_normalization.margin_max,
            "fit_on": "train_original.csv",
        },
        "hard_pair_class_names": hard_pair_class_names,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "metrics_path": summary.metrics_path,
    }
    with open(summary.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return summary

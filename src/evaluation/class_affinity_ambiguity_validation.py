"""Phase 3: validation-only analysis of the class-affinity ambiguity
signal, and a three-way comparison against Phase 1's prototype-based and
Phase 2's neighborhood-based ambiguity (both left completely unmodified
and intact).

Answers the SAME question Phase 1's and Phase 2's validation modules ask
- "is this ambiguity signal actually meaningful?" - using ONLY
data/manifests/val_original.csv and the SAME frozen reference
representation (train embeddings/labels, class-affinity matrix,
margin_scale, boundary_gap_scale) already built by
src.training.class_affinity_ambiguity_setup.build_class_affinity_ambiguity
from train_original.csv. This module never reads, requires, or depends on
data/manifests/test_original.csv in any way - its public function has no
test-manifest parameter at all.

Analyses (paralleling Phase 1's/Phase 2's own validation modules):

  1. Error-detection AUROC/AUPRC of `is_error` (val prediction wrong) vs.
     the PRIMARY class-affinity ambiguity score.
  2. Ambiguity mean/median for correct vs. incorrect val predictions.
  3. Competing-class hit rate: among misclassified val samples, how often
     does `top2_class` (the sample's second-highest class affinity) equal
     the model's actual (wrong) predicted class? (the exact rule Phase
     1/2 already use, applied to this phase's own competing-class field).
  4. Ambiguity for the existing fixed hard-pair classes vs. all others -
     explicitly NOT claimed as validated just because they were manually
     defined in the original CS-SupCon configuration.
  5. Spearman correlation between ambiguity and the model's own EDL
     uncertainty.
  6. A 2x2 quadrant breakdown (median-split ambiguity x median-split EDL
     uncertainty) reporting the val error rate within each quadrant.
  7. Ranking comparison ONLY (never a construction input) between the
     class-affinity matrix's off-diagonal entries and an ordinary
     validation confusion matrix's off-diagonal entries.
  8. Explicit rank/value of the three existing fixed hard pairs in the
     class-affinity matrix.
  9. Top-5 learned class-affinity pairs and top-5 validation confusion
     pairs.

`build_three_phase_comparison` reads Phase 1's and Phase 2's OWN saved
`*_metrics.json` files (produced by prior, separate validation runs -
never recomputed here) and this phase's own summary, and reports all
three methods' key numbers side by side, including where each of the
three existing fixed hard pairs ranks in EACH of the three matrices. It
makes NO claim about which method is "better" - it only reports numbers.
"""

import csv
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
from src.evaluation.ambiguity_validation import _mean_or_none, _median_or_none, _offdiagonal_pairs, _top_n_pairs
from src.evaluation.metrics import compute_confusion_matrix
from src.evaluation.ood_uncertainty import _error_detection_auprc, _error_detection_auroc, _spearman
from src.losses.class_affinity_ambiguity import compute_class_affinities, compute_sample_class_affinity_result
from src.losses.cs_supcon import resolve_ambiguity_pairs
from src.models.factory import create_model
from src.models.prototypes import extract_embeddings
from src.training.checkpointing import assert_checkpoint_compatible, load_checkpoint, restore_training_state
from src.training.class_affinity_ambiguity_setup import ClassAffinityAmbiguityArtifact
from src.training.logging import generate_run_id
from src.utils.git_info import get_git_commit
from src.utils.hashing import hash_file

DEFAULT_BATCH_SIZE = 16
DEFAULT_NUM_WORKERS = 0

DEFAULT_HARD_PAIR_CLASS_NAMES = [
    ["Healthy", "Glaucoma"],
    ["Disc Edema", "Glaucoma"],
    ["Diabetic Retinopathy", "Central Serous Chorioretinopathy"],
]


class ClassAffinityAmbiguityValidationError(Exception):
    """Raised for a problem running validation-only class-affinity
    ambiguity analysis: a missing val_original.csv, or an internal
    invariant violation."""


@dataclass
class ClassAffinityAmbiguityValidationSummary:
    run_id: str
    num_val_samples: int
    m: int
    temperature: float
    margin_scale: float
    boundary_gap_scale: float
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
    spearman_entropy_vs_edl_uncertainty: Optional[float]
    quadrant_error_rates: Dict[str, Optional[float]]
    quadrant_counts: Dict[str, int]
    class_matrix_vs_confusion_spearman: Optional[float]
    top5_learned_ambiguity_pairs: List[Dict[str, Any]]
    top5_confusion_pairs: List[Dict[str, Any]]
    hard_pair_ranks: List[Dict[str, Any]]
    matrix_path: str
    metrics_path: str
    metadata_path: str


def _pair_rank_and_value(matrix: np.ndarray, canonical_classes: Sequence[str], class_a: str, class_b: str) -> Tuple[Optional[int], Optional[float]]:
    if class_a not in canonical_classes or class_b not in canonical_classes:
        return None, None
    pairs = _offdiagonal_pairs(matrix, canonical_classes)
    ranked = sorted(pairs, key=lambda p: p[2], reverse=True)
    a_idx, b_idx = canonical_classes.index(class_a), canonical_classes.index(class_b)
    for rank, (i, j, value) in enumerate(ranked, start=1):
        if {i, j} == {a_idx, b_idx}:
            return rank, value
    return None, None


def run_class_affinity_ambiguity_validation(
    artifact: ClassAffinityAmbiguityArtifact,
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
) -> ClassAffinityAmbiguityValidationSummary:
    """Run the class-affinity ambiguity validation analyses against
    `val_manifest_path` (data/manifests/val_original.csv) using the SAME
    reference checkpoint and frozen train embeddings/matrix/scales
    already built by
    `src.training.class_affinity_ambiguity_setup.build_class_affinity_ambiguity`
    (passed in as `artifact`). Never reads, requires, or references
    data/manifests/test_original.csv.
    """
    canonical_classes = artifact.canonical_classes
    num_classes = len(canonical_classes)
    val_manifest_path = Path(val_manifest_path)
    if not val_manifest_path.is_file():
        raise ClassAffinityAmbiguityValidationError(
            f"{val_manifest_path} not found. Run `python run_pipeline.py prepare_dataset` first."
        )

    if hard_pair_class_names is None:
        hard_pair_class_names = DEFAULT_HARD_PAIR_CLASS_NAMES
    hard_pairs = resolve_ambiguity_pairs(
        [p for p in hard_pair_class_names if p[0] in canonical_classes and p[1] in canonical_classes],
        canonical_classes,
    )
    hard_pair_class_indices = set()
    for pair in hard_pairs.pairs:
        hard_pair_class_indices.update(pair)

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
    extracted = extract_embeddings(reference_model, loader, device)

    y_true_arr = extracted.labels
    y_pred_arr = extracted.predictions
    edl_uncertainty = extracted.uncertainty
    is_error = (y_true_arr != y_pred_arr).astype(np.int64)

    val_affinities = compute_class_affinities(
        extracted.embeddings, artifact.train_embeddings, artifact.train_labels, num_classes, m=artifact.m, exclude_self=False
    )
    result = compute_sample_class_affinity_result(
        val_affinities,
        margin_scale=artifact.margin_scale,
        temperature=artifact.temperature,
        true_labels=y_true_arr,
        boundary_gap_scale=artifact.boundary_gap_scale,
    )
    ambiguity = result.ambiguity
    competing_class = result.top2_class  # rule: 2nd-highest class affinity, see module docstring #3

    # --- 1. error-detection AUROC/AUPRC ---
    auroc = _error_detection_auroc(is_error, ambiguity)
    auprc = _error_detection_auprc(is_error, ambiguity)

    # --- 3. competing-class hit rate among errors ---
    error_mask = is_error.astype(bool)
    num_errors = int(error_mask.sum())
    hit_rate = float((competing_class[error_mask] == y_pred_arr[error_mask]).mean()) if num_errors > 0 else None

    # --- 2. ambiguity: correct vs incorrect ---
    correct_ambiguity = ambiguity[~error_mask]
    incorrect_ambiguity = ambiguity[error_mask]

    # --- 4. ambiguity: hard-pair classes vs other classes (by TRUE class) ---
    hard_pair_mask = np.array([label in hard_pair_class_indices for label in y_true_arr])
    hard_pair_ambiguity = ambiguity[hard_pair_mask]
    other_ambiguity = ambiguity[~hard_pair_mask]

    # --- 5. Spearman(ambiguity, EDL uncertainty) - both diagnostics reported ---
    spearman_margin = _spearman(ambiguity.tolist(), edl_uncertainty.tolist())
    spearman_entropy = _spearman(result.entropy.tolist(), edl_uncertainty.tolist())

    # --- 6. quadrants ---
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

    # --- 7. matrix vs. observed val confusion (comparison only) ---
    confusion = compute_confusion_matrix(y_true_arr.tolist(), y_pred_arr.tolist(), num_classes)
    confusion_symmetric = confusion.astype(np.float64) + confusion.astype(np.float64).T
    np.fill_diagonal(confusion_symmetric, 0.0)

    ambiguity_pairs_ranked = _offdiagonal_pairs(artifact.matrix_numpy, canonical_classes)
    confusion_pairs_ranked = _offdiagonal_pairs(confusion_symmetric, canonical_classes)
    class_matrix_vs_confusion_spearman = _spearman(
        [p[2] for p in ambiguity_pairs_ranked], [p[2] for p in confusion_pairs_ranked]
    )

    # --- 8. named hard-pair ranks ---
    hard_pair_ranks = []
    for class_a, class_b in DEFAULT_HARD_PAIR_CLASS_NAMES:
        rank, value = _pair_rank_and_value(artifact.matrix_numpy, canonical_classes, class_a, class_b)
        hard_pair_ranks.append({"class_a": class_a, "class_b": class_b, "rank": rank, "value": value})

    # --- 9. top pairs ---
    top5_ambiguity = _top_n_pairs(ambiguity_pairs_ranked, canonical_classes)
    top5_confusion = _top_n_pairs(confusion_pairs_ranked, canonical_classes)

    # --- outputs ---
    run_id = generate_run_id("class_affinity_ambiguity_validation", seed=0, smoke_test=False)
    base_dir = Path(output_dir) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)

    matrix_path = base_dir / "class_affinity_ambiguity_matrix.csv"
    matrix_columns = ["class_name"] + list(canonical_classes)
    matrix_rows = []
    for i, name in enumerate(canonical_classes):
        row = {"class_name": name}
        for j, other_name in enumerate(canonical_classes):
            row[other_name] = float(artifact.matrix_numpy[i, j])
        matrix_rows.append(row)
    write_csv(matrix_rows, matrix_columns, matrix_path)

    summary = ClassAffinityAmbiguityValidationSummary(
        run_id=run_id,
        num_val_samples=len(val_dataset),
        m=artifact.m,
        temperature=artifact.temperature,
        margin_scale=artifact.margin_scale,
        boundary_gap_scale=artifact.boundary_gap_scale,
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
        spearman_ambiguity_vs_edl_uncertainty=spearman_margin,
        spearman_entropy_vs_edl_uncertainty=spearman_entropy,
        quadrant_error_rates=quadrant_error_rates,
        quadrant_counts=quadrant_counts,
        class_matrix_vs_confusion_spearman=class_matrix_vs_confusion_spearman,
        top5_learned_ambiguity_pairs=top5_ambiguity,
        top5_confusion_pairs=top5_confusion,
        hard_pair_ranks=hard_pair_ranks,
        matrix_path=str(matrix_path),
        metrics_path=str(base_dir / "class_affinity_validation_metrics.json"),
        metadata_path=str(base_dir / "class_affinity_metadata.json"),
    )

    with open(summary.metrics_path, "w", encoding="utf-8") as f:
        json.dump(asdict(summary), f, indent=2, sort_keys=True)

    metadata = {
        "run_id": run_id,
        "method": "class_affinity_ambiguity (Phase 3) - analysis only, not wired into training",
        "m": artifact.m,
        "temperature": artifact.temperature,
        "scale_percentile": artifact.scale_percentile,
        "margin_scale": artifact.margin_scale,
        "boundary_gap_scale": artifact.boundary_gap_scale,
        "primary_score_formula": (
            "a_i,c = mean(top-m cosine similarities to class c's train embeddings); "
            "margin_i = top1_affinity - top2_affinity; "
            "ambiguity_i (PRIMARY) = 1 - clip(margin_i / margin_scale, 0, 1); "
            "margin_scale = 95th percentile of TRAIN samples' own self-excluded top1-top2 margins"
        ),
        "secondary_diagnostics_formula": "normalized entropy of softmax(affinities / temperature)",
        "label_aware_formula_analysis_only": (
            "boundary_gap_i = affinity to TRUE class - max affinity to any other class; "
            "label_aware_ambiguity_i = 1 - clip(boundary_gap_i / boundary_gap_scale, 0, 1); "
            "boundary_gap_scale = 95th percentile of TRAIN samples' own boundary gaps. "
            "NEVER an inference-time score - requires the ground-truth label."
        ),
        "competing_class_hit_rate_rule": (
            "Among misclassified validation samples, the fraction where top2_class "
            "(the sample's second-highest class affinity) equals the model's actual (wrong) predicted class."
        ),
        "matrix_formula": (
            "directed(a->b) = mean over train samples in class a of their (self-excluded) affinity to class b; "
            "A_sym = (directed(a->b)+directed(b->a))/2, diagonal=0; "
            "A = min-max rescale of A_sym's off-diagonal entries to [0,1] (train-derived only)"
        ),
        "reference_checkpoint_path": artifact.reference_checkpoint_path,
        "reference_checkpoint_sha256": artifact.reference_checkpoint_sha256,
        "reference_model_name": artifact.reference_model_name,
        "class_names": canonical_classes,
        "num_classes": num_classes,
        "num_train_samples": artifact.num_train_samples,
        "train_manifest_path": artifact.train_manifest_path,
        "train_manifest_sha256": artifact.train_manifest_sha256,
        "val_manifest_path": str(val_manifest_path),
        "val_manifest_sha256": hash_file(val_manifest_path),
        "num_val_samples": len(val_dataset),
        "hard_pair_class_names": hard_pair_class_names,
        "hard_pair_ranks": hard_pair_ranks,
        "test_data_used": False,
        "test_data_confirmation": "data/manifests/test_original.csv was not read, referenced, or used anywhere in this computation.",
        "training_confirmation": "No training or fine-tuning occurred - the reference checkpoint's weights were only read, never updated.",
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "matrix_path": str(matrix_path),
        "metrics_path": summary.metrics_path,
    }
    with open(summary.metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)

    return summary


def _load_matrix_csv(path: Union[str, Path], canonical_classes: Sequence[str]) -> np.ndarray:
    num_classes = len(canonical_classes)
    matrix = np.zeros((num_classes, num_classes), dtype=np.float64)
    index_of = {name: i for i, name in enumerate(canonical_classes)}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            i = index_of[row["class_name"]]
            for name, j in index_of.items():
                matrix[i, j] = float(row[name])
    return matrix


def build_three_phase_comparison(
    phase1_metrics_path: Union[str, Path],
    phase1_matrix_path: Union[str, Path],
    phase2_metrics_path: Union[str, Path],
    phase2_matrix_path: Union[str, Path],
    phase3_summary: ClassAffinityAmbiguityValidationSummary,
    phase3_matrix: np.ndarray,
    canonical_classes: Sequence[str],
    output_path: Union[str, Path],
    named_pairs: Optional[Sequence[Sequence[str]]] = None,
) -> Dict[str, Any]:
    """Read Phase 1's and Phase 2's OWN saved artifacts (produced by
    prior, separate validation runs - never recomputed here) and build a
    three-way side-by-side comparison against `phase3_summary`/
    `phase3_matrix`. Makes NO claim about which method is "better" -
    reports the numbers only.
    """
    canonical_classes = list(canonical_classes)
    with open(phase1_metrics_path, encoding="utf-8") as f:
        phase1_metrics = json.load(f)
    with open(phase2_metrics_path, encoding="utf-8") as f:
        phase2_metrics = json.load(f)
    phase1_matrix = _load_matrix_csv(phase1_matrix_path, canonical_classes)
    phase2_matrix = _load_matrix_csv(phase2_matrix_path, canonical_classes)

    if named_pairs is None:
        named_pairs = DEFAULT_HARD_PAIR_CLASS_NAMES

    named_pair_comparison = []
    for class_a, class_b in named_pairs:
        p1_rank, p1_value = _pair_rank_and_value(phase1_matrix, canonical_classes, class_a, class_b)
        p2_rank, p2_value = _pair_rank_and_value(phase2_matrix, canonical_classes, class_a, class_b)
        p3_rank, p3_value = _pair_rank_and_value(phase3_matrix, canonical_classes, class_a, class_b)
        named_pair_comparison.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "phase1_prototype_rank": p1_rank,
                "phase1_prototype_value": p1_value,
                "phase2_neighborhood_rank": p2_rank,
                "phase2_neighborhood_value": p2_value,
                "phase3_class_affinity_rank": p3_rank,
                "phase3_class_affinity_value": p3_value,
            }
        )

    def _extract(metrics: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "error_detection_auroc": metrics.get("error_detection_auroc"),
            "error_detection_auprc": metrics.get("error_detection_auprc"),
            "class_matrix_vs_confusion_spearman": metrics.get("class_matrix_vs_confusion_spearman"),
            "competing_class_hit_rate_among_errors": metrics.get("competing_class_hit_rate_among_errors"),
        }

    comparison = {
        "note": (
            "This is a factual side-by-side comparison only. It does NOT claim any method is better or worse "
            "than another - that judgment requires reviewing the actual numbers below, all computed on "
            "val_original.csv only (never test_original.csv)."
        ),
        "phase1_prototype": _extract(phase1_metrics),
        "phase2_neighborhood": _extract(phase2_metrics),
        "phase3_class_affinity": _extract(asdict(phase3_summary)),
        "named_pair_comparison": named_pair_comparison,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, sort_keys=True)

    return comparison

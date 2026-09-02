"""Phase 3b: does Phase 3's class-affinity ambiguity signal carry error-
detection information beyond the model's own EDL uncertainty?

This module performs NO inference and loads NO model/checkpoint. It reads
a per-sample CSV already produced by
`src.evaluation.class_affinity_ambiguity_validation.run_class_affinity_ambiguity_validation(
..., save_per_sample_csv=True)` - i.e. it reuses Phase 3's own validation
run rather than duplicating the forward pass over val_original.csv. Its
public function has no test-manifest parameter and never reads
data/manifests/test_original.csv.

Both input signals are already normalized to a bounded range by
construction, so NO additional normalization fitting is performed here
(there is nothing left to fit "using train data only" - it was already
done, using TRAIN data only, by the modules that produced these two
numbers):
  - Phase 3's `ambiguity` = 1 - clip(margin / margin_scale, 0, 1), where
    margin_scale is a 95th-percentile constant fitted on TRAIN samples
    only (src/losses/class_affinity_ambiguity.py). Already in [0, 1].
  - AA-EvidentNet's own EDL `uncertainty` = K / sum(dirichlet_alpha),
    mathematically guaranteed to lie in (0, 1] by construction
    (src/losses/evidential.py) - no empirical fitting is possible or
    needed for a quantity that is already bounded by its own formula.

The combination weight is NOT searched: it is fixed, in code, at
0.5/0.5 (see COMBINATION_EQUATION below), applied identically to every
run. No optimizer, no backward pass, and no validation-based weight
selection occurs anywhere in this module.

This module makes NO claim that ambiguity is or is not "complementary" to
EDL uncertainty. It only computes and reports numbers; the interpretation
is left to the reader (see `NOTE` in the output JSON).
"""

import csv
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

from src.evaluation.ood_uncertainty import _error_detection_auprc, _error_detection_auroc, _spearman
from src.training.logging import generate_run_id
from src.utils.git_info import get_git_commit

AMBIGUITY_WEIGHT = 0.5
EDL_WEIGHT = 0.5
COMBINATION_EQUATION = "combined_score = 0.5 * normalized_ambiguity + 0.5 * normalized_edl_uncertainty"

DEFAULT_BOOTSTRAP_RESAMPLES = 10000
DEFAULT_BOOTSTRAP_SEED = 0

NOTE = (
    "This is a factual report of computed metrics only. It does NOT claim that "
    "class-affinity ambiguity is or is not complementary to EDL uncertainty - that "
    "determination requires reviewing the numbers above (in particular whether the "
    "combined score detects errors the two individual signals individually miss, and "
    "whether the AUROC/AUPRC differences are large and stable across the bootstrap "
    "confidence interval). A high Spearman correlation alone does not prove redundancy, "
    "and a combined-score improvement alone does not prove statistical significance."
)


class AmbiguityComplementarityError(Exception):
    """Raised for a problem running the ambiguity/EDL complementarity
    analysis: a missing/malformed per-sample CSV, or an internal
    invariant violation."""


@dataclass
class PerSampleData:
    sample_id: np.ndarray
    is_error: np.ndarray  # [N] int64, 1 = model prediction wrong
    ambiguity: np.ndarray  # [N] float64, Phase 3's primary score, already in [0, 1]
    edl_uncertainty: np.ndarray  # [N] float64, already in (0, 1]


def load_per_sample_csv(path: Union[str, Path]) -> PerSampleData:
    """Read the per-sample CSV written by
    `run_class_affinity_ambiguity_validation(..., save_per_sample_csv=True)`.
    Requires at least the columns: sample_id, correct, ambiguity,
    edl_uncertainty. Never reads test_original.csv or any other manifest -
    this function only ever opens the single CSV path it is given."""
    path = Path(path)
    if not path.is_file():
        raise AmbiguityComplementarityError(f"{path} not found")

    sample_ids: List[str] = []
    is_error: List[int] = []
    ambiguity: List[float] = []
    edl_uncertainty: List[float] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"sample_id", "correct", "ambiguity", "edl_uncertainty"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise AmbiguityComplementarityError(f"{path} is missing required column(s): {sorted(missing)}")
        for row in reader:
            sample_ids.append(row["sample_id"])
            is_error.append(1 - int(row["correct"]))
            ambiguity.append(float(row["ambiguity"]))
            edl_uncertainty.append(float(row["edl_uncertainty"]))

    if not sample_ids:
        raise AmbiguityComplementarityError(f"{path} contains zero rows - nothing to analyze")

    return PerSampleData(
        sample_id=np.asarray(sample_ids, dtype=object),
        is_error=np.asarray(is_error, dtype=np.int64),
        ambiguity=np.asarray(ambiguity, dtype=np.float64),
        edl_uncertainty=np.asarray(edl_uncertainty, dtype=np.float64),
    )


def compute_combined_score(ambiguity: np.ndarray, edl_uncertainty: np.ndarray) -> np.ndarray:
    """`AMBIGUITY_WEIGHT * ambiguity + EDL_WEIGHT * edl_uncertainty` -
    a fixed, predetermined 0.5/0.5 combination (see COMBINATION_EQUATION).
    Neither input is re-normalized here: both already lie in a known,
    bounded range by construction (see module docstring)."""
    return AMBIGUITY_WEIGHT * np.asarray(ambiguity, dtype=np.float64) + EDL_WEIGHT * np.asarray(
        edl_uncertainty, dtype=np.float64
    )


def _quadrant_counts(flag_a: np.ndarray, flag_b: np.ndarray, mask: np.ndarray) -> Dict[str, int]:
    subset_a = flag_a[mask]
    subset_b = flag_b[mask]
    return {
        "both": int(np.sum(subset_a & subset_b)),
        "a_only": int(np.sum(subset_a & ~subset_b)),
        "b_only": int(np.sum(~subset_a & subset_b)),
        "neither": int(np.sum(~subset_a & ~subset_b)),
    }


def _percentages(counts: Dict[str, int], denominator: int) -> Dict[str, Optional[float]]:
    if denominator == 0:
        return {k: None for k in counts}
    return {k: float(v) / float(denominator) for k, v in counts.items()}


def _discordant_group(mask: np.ndarray, is_error: np.ndarray) -> Dict[str, Any]:
    count = int(mask.sum())
    error_count = int(is_error[mask].sum()) if count > 0 else 0
    error_rate = float(error_count) / float(count) if count > 0 else None
    return {"count": count, "error_count": error_count, "error_rate": error_rate}


def _bootstrap_auroc_diff_ci(
    is_error: np.ndarray,
    ambiguity: np.ndarray,
    edl_uncertainty: np.ndarray,
    combined: np.ndarray,
    num_resamples: int,
    seed: int,
) -> Dict[str, Any]:
    """Paired bootstrap: the SAME resampled indices are applied to all
    three scores within each iteration, so the resulting distribution is
    of the metric DIFFERENCE on matched resamples (not of two independently
    resampled AUROCs). A resample is skipped (and counted as degenerate,
    not as a zero difference) whenever `is_error` in that resample is
    single-class (all-correct or all-error), since error-detection AUROC
    is undefined there - `_error_detection_auroc` already returns None in
    that case, which is what is checked here."""
    rng = np.random.default_rng(seed)
    n = len(is_error)
    combined_minus_edl: List[float] = []
    combined_minus_ambiguity: List[float] = []
    num_valid = 0
    num_degenerate = 0
    for _ in range(num_resamples):
        idx = rng.integers(0, n, size=n)
        resampled_is_error = is_error[idx]
        auroc_combined = _error_detection_auroc(resampled_is_error, combined[idx])
        auroc_edl = _error_detection_auroc(resampled_is_error, edl_uncertainty[idx])
        auroc_ambiguity = _error_detection_auroc(resampled_is_error, ambiguity[idx])
        if auroc_combined is None or auroc_edl is None or auroc_ambiguity is None:
            num_degenerate += 1
            continue
        num_valid += 1
        combined_minus_edl.append(auroc_combined - auroc_edl)
        combined_minus_ambiguity.append(auroc_combined - auroc_ambiguity)

    def _ci(values: List[float]) -> Optional[List[float]]:
        if not values:
            return None
        arr = np.asarray(values, dtype=np.float64)
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {
        "num_resamples_requested": num_resamples,
        "num_valid_resamples": num_valid,
        "num_degenerate_resamples_skipped": num_degenerate,
        "seed": seed,
        "combined_minus_edl_auroc_95ci": _ci(combined_minus_edl),
        "combined_minus_ambiguity_auroc_95ci": _ci(combined_minus_ambiguity),
    }


def run_ambiguity_edl_complementarity_analysis(
    per_sample_csv_path: Union[str, Path],
    output_dir: Union[str, Path] = "results/ambiguity",
    bootstrap_resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    run_bootstrap: bool = True,
) -> Dict[str, Any]:
    """Analysis-only comparison of EDL-alone, ambiguity-alone, and a fixed
    0.5/0.5 combined score's error detection, on the SAME validation
    samples/errors Phase 3 already evaluated (read from
    `per_sample_csv_path`). No model, checkpoint, optimizer, or manifest is
    touched by this function - it is pure post-hoc arithmetic over already-
    computed numbers. Writes `ambiguity_edl_complementarity.json` under
    `output_dir/<run_id>/` and returns the same dict that was written.
    """
    data = load_per_sample_csv(per_sample_csv_path)
    n = len(data.is_error)
    num_errors = int(data.is_error.sum())

    combined = compute_combined_score(data.ambiguity, data.edl_uncertainty)

    edl_auroc = _error_detection_auroc(data.is_error, data.edl_uncertainty)
    edl_auprc = _error_detection_auprc(data.is_error, data.edl_uncertainty)
    ambiguity_auroc = _error_detection_auroc(data.is_error, data.ambiguity)
    ambiguity_auprc = _error_detection_auprc(data.is_error, data.ambiguity)
    combined_auroc = _error_detection_auroc(data.is_error, combined)
    combined_auprc = _error_detection_auprc(data.is_error, combined)

    def _diff(a: Optional[float], b: Optional[float]) -> Optional[float]:
        return None if a is None or b is None else float(a - b)

    # --- median-split thresholds, computed on this same validation set,
    # used only for the discrete overlap/false-alarm/discordant analyses
    # below - never used to fit or select anything. ---
    ambiguity_median = float(np.median(data.ambiguity))
    edl_median = float(np.median(data.edl_uncertainty))
    flag_ambiguity = data.ambiguity > ambiguity_median
    flag_edl = data.edl_uncertainty > edl_median

    error_mask = data.is_error.astype(bool)
    correct_mask = ~error_mask

    # --- A. error overlap (among the num_errors misclassified samples) ---
    error_overlap_counts_raw = _quadrant_counts(flag_ambiguity, flag_edl, error_mask)
    error_overlap = {
        "both": error_overlap_counts_raw["both"],
        "ambiguity_only": error_overlap_counts_raw["a_only"],
        "edl_only": error_overlap_counts_raw["b_only"],
        "neither": error_overlap_counts_raw["neither"],
    }
    error_overlap_pct = {
        "both": None,
        "ambiguity_only": None,
        "edl_only": None,
        "neither": None,
    }
    if num_errors > 0:
        for key in error_overlap_pct:
            error_overlap_pct[key] = float(error_overlap[key]) / float(num_errors)

    # --- B. correct-sample false alarms ---
    num_correct = n - num_errors
    correct_counts_raw = _quadrant_counts(flag_ambiguity, flag_edl, correct_mask)
    correct_false_alarms = {
        "both": correct_counts_raw["both"],
        "ambiguity_only": correct_counts_raw["a_only"],
        "edl_only": correct_counts_raw["b_only"],
        "neither": correct_counts_raw["neither"],
    }
    correct_false_alarms_pct = {"both": None, "ambiguity_only": None, "edl_only": None, "neither": None}
    if num_correct > 0:
        for key in correct_false_alarms_pct:
            correct_false_alarms_pct[key] = float(correct_false_alarms[key]) / float(num_correct)

    # --- C. discordant cases (recomputed from the actual per-sample data
    # loaded above - never hardcoded). ---
    high_ambiguity_low_edl_mask = flag_ambiguity & ~flag_edl
    low_ambiguity_high_edl_mask = ~flag_ambiguity & flag_edl
    discordant = {
        "high_ambiguity_low_edl": _discordant_group(high_ambiguity_low_edl_mask, data.is_error),
        "low_ambiguity_high_edl": _discordant_group(low_ambiguity_high_edl_mask, data.is_error),
    }

    # --- D. Spearman(ambiguity, EDL uncertainty) ---
    spearman = _spearman(data.ambiguity.tolist(), data.edl_uncertainty.tolist())

    # --- E. bootstrap 95% CI for the AUROC differences ---
    bootstrap: Optional[Dict[str, Any]] = None
    if run_bootstrap:
        bootstrap = _bootstrap_auroc_diff_ci(
            data.is_error, data.ambiguity, data.edl_uncertainty, combined, bootstrap_resamples, bootstrap_seed
        )

    run_id = generate_run_id("ambiguity_edl_complementarity", seed=0, smoke_test=False)
    base_dir = Path(output_dir) / run_id
    base_dir.mkdir(parents=True, exist_ok=True)
    output_path = base_dir / "ambiguity_edl_complementarity.json"

    result: Dict[str, Any] = {
        "run_id": run_id,
        "source_per_sample_csv": str(Path(per_sample_csv_path)),
        "num_samples": n,
        "num_errors": num_errors,
        "num_correct": num_correct,
        "edl_error_detection_auroc": edl_auroc,
        "edl_error_detection_auprc": edl_auprc,
        "ambiguity_error_detection_auroc": ambiguity_auroc,
        "ambiguity_error_detection_auprc": ambiguity_auprc,
        "combined_error_detection_auroc": combined_auroc,
        "combined_error_detection_auprc": combined_auprc,
        "combined_minus_edl_auroc": _diff(combined_auroc, edl_auroc),
        "combined_minus_edl_auprc": _diff(combined_auprc, edl_auprc),
        "combined_minus_ambiguity_auroc": _diff(combined_auroc, ambiguity_auroc),
        "combined_minus_ambiguity_auprc": _diff(combined_auprc, ambiguity_auprc),
        "median_thresholds": {"ambiguity_median": ambiguity_median, "edl_uncertainty_median": edl_median},
        "error_overlap_counts": error_overlap,
        "error_overlap_percentages_of_errors": error_overlap_pct,
        "correct_false_alarm_counts": correct_false_alarms,
        "correct_false_alarm_percentages_of_correct": correct_false_alarms_pct,
        "discordant_cases": discordant,
        "spearman_ambiguity_vs_edl_uncertainty": spearman,
        "bootstrap": bootstrap,
        "combination_equation": COMBINATION_EQUATION,
        "ambiguity_weight": AMBIGUITY_WEIGHT,
        "edl_weight": EDL_WEIGHT,
        "weights_predetermined_confirmation": (
            "The 0.5/0.5 combination weights are fixed constants in source code "
            "(AMBIGUITY_WEIGHT, EDL_WEIGHT in src/evaluation/ambiguity_complementarity.py), "
            "chosen before this analysis was run, not fitted or selected from validation data."
        ),
        "no_weight_search_confirmation": "No alternative weight was tried, computed, or compared - only 0.5/0.5.",
        "no_additional_normalization_confirmation": (
            "Phase 3's ambiguity score is already in [0, 1] via its own TRAIN-derived margin_scale; "
            "EDL uncertainty (K / sum(dirichlet_alpha)) is already mathematically bounded in (0, 1] by "
            "construction. Neither input was re-normalized before combination."
        ),
        "no_training_confirmation": (
            "No model was loaded, no checkpoint was read, and no optimizer/backward pass occurred in this "
            "module - it only reads a pre-computed per-sample CSV and performs arithmetic on it."
        ),
        "test_data_used": False,
        "test_data_confirmation": (
            "data/manifests/test_original.csv was not read, referenced, or used anywhere in this computation. "
            "This module's public function has no test-manifest parameter."
        ),
        "same_validation_samples_confirmation": (
            "EDL-alone, ambiguity-alone, and combined scores were all computed on the identical "
            f"{n} validation samples and {num_errors} model errors read from a single per-sample CSV."
        ),
        "note": NOTE,
        "git_commit": get_git_commit(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "output_path": str(output_path),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    return result

"""Tests for src.evaluation.ambiguity_complementarity (Phase 3b) - the
ambiguity/EDL-uncertainty complementarity analysis.

CRITICAL: this module performs no inference and touches no model,
checkpoint, or manifest at all - it only reads a per-sample CSV. Every
test here uses a synthetic, hand-constructed CSV; none reads
test_original.csv, val_original.csv, or any real dataset artifact.
"""

import csv
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.ambiguity_complementarity import (
    AMBIGUITY_WEIGHT,
    COMBINATION_EQUATION,
    EDL_WEIGHT,
    AmbiguityComplementarityError,
    compute_combined_score,
    load_per_sample_csv,
    run_ambiguity_edl_complementarity_analysis,
)

PER_SAMPLE_COLUMNS = [
    "sample_id", "image_path", "true_class_name", "predicted_class_name",
    "correct", "ambiguity", "edl_uncertainty",
]


def _write_per_sample_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=PER_SAMPLE_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _row(i, correct, ambiguity, edl_uncertainty, true_class="A", pred_class=None):
    pred_class = true_class if correct else "B"
    return {
        "sample_id": f"s{i}",
        "image_path": f"/data/s{i}.jpg",
        "true_class_name": true_class,
        "predicted_class_name": pred_class,
        "correct": int(correct),
        "ambiguity": ambiguity,
        "edl_uncertainty": edl_uncertainty,
    }


def _synthetic_rows(seed=0, n=200):
    """A mix where ambiguity and EDL uncertainty are correlated but not
    identical, and errors are more likely (not guaranteed) at higher
    values of both - enough structure for AUROC/AUPRC to be well-defined
    and non-trivial, without hand-picking a specific numeric answer."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n):
        base = rng.random()
        ambiguity = float(np.clip(base + rng.normal(0, 0.1), 0, 1))
        edl_uncertainty = float(np.clip(base + rng.normal(0, 0.1), 0.01, 1))
        error_prob = 0.05 + 0.5 * base
        correct = rng.random() > error_prob
        rows.append(_row(i, correct, ambiguity, edl_uncertainty))
    return rows


# --- CSV loading ---


def test_load_per_sample_csv_raises_on_missing_file(tmp_path):
    with pytest.raises(AmbiguityComplementarityError):
        load_per_sample_csv(tmp_path / "does_not_exist.csv")


def test_load_per_sample_csv_raises_on_missing_columns(tmp_path):
    path = tmp_path / "bad.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "correct"])
        writer.writeheader()
        writer.writerow({"sample_id": "s0", "correct": 1})
    with pytest.raises(AmbiguityComplementarityError, match="missing required column"):
        load_per_sample_csv(path)


def test_load_per_sample_csv_raises_on_zero_rows(tmp_path):
    path = tmp_path / "empty.csv"
    _write_per_sample_csv(path, [])
    with pytest.raises(AmbiguityComplementarityError, match="zero rows"):
        load_per_sample_csv(path)


def test_load_per_sample_csv_parses_fields_correctly(tmp_path):
    path = tmp_path / "sample.csv"
    rows = [_row(0, True, 0.1, 0.2), _row(1, False, 0.9, 0.8)]
    _write_per_sample_csv(path, rows)

    data = load_per_sample_csv(path)
    assert list(data.is_error) == [0, 1]
    assert data.ambiguity == pytest.approx([0.1, 0.9])
    assert data.edl_uncertainty == pytest.approx([0.2, 0.8])


# --- combination formula ---


def test_combination_weights_are_fixed_at_half_and_half():
    assert AMBIGUITY_WEIGHT == pytest.approx(0.5)
    assert EDL_WEIGHT == pytest.approx(0.5)
    assert AMBIGUITY_WEIGHT + EDL_WEIGHT == pytest.approx(1.0)
    assert "0.5" in COMBINATION_EQUATION


def test_compute_combined_score_matches_predetermined_formula():
    ambiguity = np.array([0.0, 0.4, 1.0])
    edl_uncertainty = np.array([1.0, 0.4, 0.0])
    combined = compute_combined_score(ambiguity, edl_uncertainty)
    expected = 0.5 * ambiguity + 0.5 * edl_uncertainty
    assert combined == pytest.approx(expected)


def test_no_alternative_weight_is_ever_computed_in_source():
    import src.evaluation.ambiguity_complementarity as module

    source = inspect.getsource(module)
    # The only numeric literals used as combination weights anywhere in
    # this module must be the fixed 0.5/0.5 constants - no grid, no list
    # of candidate weights, no argmax over weight choices.
    assert "weight_grid" not in source.lower()
    assert "argmax" not in source.lower()


# --- end-to-end analysis ---


def test_run_analysis_end_to_end_produces_expected_json(tmp_path):
    csv_path = tmp_path / "class_affinity_per_sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=1, n=300))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path,
        output_dir=tmp_path / "ambiguity_out",
        bootstrap_resamples=200,
        bootstrap_seed=0,
    )

    output_path = Path(result["output_path"])
    assert output_path.is_file()
    with open(output_path, encoding="utf-8") as f:
        on_disk = json.load(f)
    assert on_disk == result

    assert result["num_samples"] == 300
    assert result["num_errors"] + result["num_correct"] == 300
    for key in (
        "edl_error_detection_auroc", "edl_error_detection_auprc",
        "ambiguity_error_detection_auroc", "ambiguity_error_detection_auprc",
        "combined_error_detection_auroc", "combined_error_detection_auprc",
    ):
        assert result[key] is None or 0.0 <= result[key] <= 1.0

    assert result["combined_minus_edl_auroc"] == pytest.approx(
        result["combined_error_detection_auroc"] - result["edl_error_detection_auroc"]
    )
    assert result["combined_minus_ambiguity_auprc"] == pytest.approx(
        result["combined_error_detection_auprc"] - result["ambiguity_error_detection_auprc"]
    )
    assert result["combination_equation"] == COMBINATION_EQUATION
    assert result["ambiguity_weight"] == pytest.approx(0.5)
    assert result["edl_weight"] == pytest.approx(0.5)
    assert result["test_data_used"] is False
    assert "note" in result
    assert "not claim" in result["note"].lower() or "does not claim" in result["note"].lower()


def test_error_overlap_counts_sum_to_num_errors(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=2, n=250))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )

    overlap = result["error_overlap_counts"]
    assert sum(overlap.values()) == result["num_errors"]


def test_correct_false_alarm_counts_sum_to_num_correct(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=3, n=250))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )

    counts = result["correct_false_alarm_counts"]
    assert sum(counts.values()) == result["num_correct"]


def test_discordant_groups_recomputed_not_hardcoded(tmp_path):
    """Construct a fixture where the discordant-quadrant counts are known
    by hand construction (not by asserting a specific literal previously
    seen elsewhere) and confirm the module reproduces those exact counts
    from the actual per-sample data."""
    rows = []
    # 4 samples: ambiguity/edl split exactly at the median of [0.2, 0.4, 0.6, 0.8].
    # median ambiguity = 0.5, median edl = 0.5.
    # s0: amb=0.1 (low), edl=0.1 (low) -> neither flagged
    # s1: amb=0.9 (high), edl=0.1 (low) -> high_ambiguity_low_edl, error
    # s2: amb=0.1 (low), edl=0.9 (high) -> low_ambiguity_high_edl, correct
    # s3: amb=0.9 (high), edl=0.9 (high) -> both flagged
    rows.append(_row(0, True, 0.1, 0.1))
    rows.append(_row(1, False, 0.9, 0.1))
    rows.append(_row(2, True, 0.1, 0.9))
    rows.append(_row(3, True, 0.9, 0.9))

    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, rows)

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )

    high_amb_low_edl = result["discordant_cases"]["high_ambiguity_low_edl"]
    low_amb_high_edl = result["discordant_cases"]["low_ambiguity_high_edl"]
    assert high_amb_low_edl == {"count": 1, "error_count": 1, "error_rate": pytest.approx(1.0)}
    assert low_amb_high_edl == {"count": 1, "error_count": 0, "error_rate": pytest.approx(0.0)}


def test_spearman_matches_reused_helper(tmp_path):
    from src.evaluation.ood_uncertainty import _spearman

    rows = _synthetic_rows(seed=4, n=150)
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, rows)

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )

    ambiguity = [r["ambiguity"] for r in rows]
    edl_uncertainty = [r["edl_uncertainty"] for r in rows]
    expected = _spearman(ambiguity, edl_uncertainty)
    assert result["spearman_ambiguity_vs_edl_uncertainty"] == pytest.approx(expected)


# --- bootstrap ---


def test_bootstrap_ci_reported_with_valid_and_degenerate_counts(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=5, n=100))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out",
        bootstrap_resamples=500, bootstrap_seed=42,
    )

    bootstrap = result["bootstrap"]
    assert bootstrap["num_resamples_requested"] == 500
    assert bootstrap["num_valid_resamples"] + bootstrap["num_degenerate_resamples_skipped"] == 500
    assert bootstrap["seed"] == 42
    for key in ("combined_minus_edl_auroc_95ci", "combined_minus_ambiguity_auroc_95ci"):
        ci = bootstrap[key]
        if ci is not None:
            assert len(ci) == 2
            assert ci[0] <= ci[1]


def test_bootstrap_skipped_when_run_bootstrap_false(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=6, n=80))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )
    assert result["bootstrap"] is None


def test_bootstrap_is_deterministic_for_fixed_seed(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=7, n=120))

    result_a = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out_a",
        bootstrap_resamples=300, bootstrap_seed=99,
    )
    result_b = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out_b",
        bootstrap_resamples=300, bootstrap_seed=99,
    )
    assert result_a["bootstrap"] == result_b["bootstrap"]


def test_all_error_detection_metrics_deterministic_for_fixed_input(tmp_path):
    csv_path = tmp_path / "sample.csv"
    rows = _synthetic_rows(seed=8, n=100)
    _write_per_sample_csv(csv_path, rows)

    result_a = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out_a", run_bootstrap=False,
    )
    result_b = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out_b", run_bootstrap=False,
    )

    for key in (
        "edl_error_detection_auroc", "ambiguity_error_detection_auroc", "combined_error_detection_auroc",
        "error_overlap_counts", "correct_false_alarm_counts", "discordant_cases",
        "spearman_ambiguity_vs_edl_uncertainty",
    ):
        assert result_a[key] == result_b[key]


# --- leakage safeguards ---


def test_no_test_manifest_parameter_exists():
    signature = inspect.signature(run_ambiguity_edl_complementarity_analysis)
    param_names = set(signature.parameters.keys())
    assert not any("test" in name.lower() for name in param_names)


def test_module_never_imports_torch_or_touches_a_checkpoint():
    """The docstring is allowed to reference "checkpoint"/"optimizer" in
    prose (explaining what this module deliberately does NOT do) - what
    must never appear is actual code that imports torch, loads a
    checkpoint, constructs an optimizer, or calls .backward()."""
    import src.evaluation.ambiguity_complementarity as module

    source = inspect.getsource(module)
    assert "import torch" not in source
    assert "load_checkpoint" not in source
    assert "restore_training_state" not in source
    assert "torch.optim" not in source
    assert ".backward(" not in source


def test_module_never_reads_test_original_csv_by_name():
    """The docstring is allowed to mention "test_original.csv" in prose
    (explaining that it is never read) - what must never appear is a
    literal path construction/open of that file."""
    import src.evaluation.ambiguity_complementarity as module

    source = inspect.getsource(module)
    assert '"test_original.csv"' not in source
    assert "'test_original.csv'" not in source


def test_same_validation_samples_used_for_all_three_scores(tmp_path):
    csv_path = tmp_path / "sample.csv"
    rows = _synthetic_rows(seed=9, n=90)
    _write_per_sample_csv(csv_path, rows)

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )
    assert result["num_samples"] == 90
    assert str(result["num_samples"]) in result["same_validation_samples_confirmation"]


def test_result_contains_all_required_confirmation_statements(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=10, n=60))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )
    for key in (
        "weights_predetermined_confirmation",
        "no_weight_search_confirmation",
        "no_additional_normalization_confirmation",
        "no_training_confirmation",
        "test_data_confirmation",
        "same_validation_samples_confirmation",
    ):
        assert key in result
        assert isinstance(result[key], str) and len(result[key]) > 0
    assert result["test_data_used"] is False


def test_result_never_claims_ambiguity_is_complementary(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=11, n=60))

    result = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )
    serialized = json.dumps(result).lower()
    assert "ambiguity is complementary" not in serialized
    assert "proves" not in serialized
    assert "ambiguity is superior" not in serialized
    assert "ambiguity outperforms" not in serialized


def test_two_invocations_get_distinct_run_ids(tmp_path):
    csv_path = tmp_path / "sample.csv"
    _write_per_sample_csv(csv_path, _synthetic_rows(seed=12, n=60))

    result_a = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )
    result_b = run_ambiguity_edl_complementarity_analysis(
        per_sample_csv_path=csv_path, output_dir=tmp_path / "out", run_bootstrap=False,
    )
    assert result_a["run_id"] != result_b["run_id"]
    assert Path(result_a["output_path"]).is_file()
    assert Path(result_b["output_path"]).is_file()

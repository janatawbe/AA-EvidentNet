# Experiment Logging Policy

This document defines how every experiment run under `run_pipeline.py`
should be logged once the corresponding pipeline stages are implemented.
It exists now, at the foundation stage, so that logging is designed in from
the start rather than bolted on later.

## Core rule: no fabricated results

**Never** write numbers, tables, or figures into `results/`, `paper/`, or
any report unless they were produced by actually executing the pipeline
against real data. If a stage is not implemented yet, the CLI must fail
loudly (see `run_pipeline.py`), not emit placeholder numbers.

## What every run must record

Each invocation of a pipeline stage that produces results should write a
run record containing, at minimum:

- `command`: the CLI command and full argument list used
- `config_path` and `config_hash` (`src/utils/config.hash_config`)
- `seed` (one of 42, 123, 456, 789, 2026 for official runs)
- `git_commit` and `git_status` (`src/utils/git_info`) — a "dirty" status
  (uncommitted changes) should be flagged in the run record, not hidden
- `environment_info` (`src/utils/env_info.collect_environment_info`)
- `dataset_manifest_hash` (`src/utils/hashing.hash_manifest`) — so results
  can be tied to the exact set of files/labels used
- start/end timestamps
- output artifact paths (checkpoints, logs, tables, figures)

## Directory conventions

- `results/logs/<experiment>/<run_id>/` — training/eval logs, run record
  (e.g. `run_record.json`), console output
- `results/checkpoints/<experiment>/<run_id>/` — model weights
- `results/raw_predictions/<experiment>/<run_id>/` — per-sample predictions,
  logits/evidential parameters, uncertainty scores (needed for later
  calibration/selective-prediction/statistics stages to recompute metrics
  without re-running inference)
- `results/tables/` — aggregated, human-readable tables (CSV/Markdown)
- `results/figures/` — generated plots

`run_id` should be unique and traceable, e.g.
`{experiment}_seed{seed}_{git_commit_short}_{utc_timestamp}`.

## Experiment directories (`experiments/`)

Each stage has a dedicated directory (see `configs/experiments.yaml` for the
authoritative mapping from CLI command to directory):

- `00_data_audit`, `01_baselines`, `02_proposed_model`, `03_ablation`,
  `04_hard_pairs`, `05_calibration_uncertainty`, `06_selective_prediction`,
  `07_interpretability`, `08_robustness`

These directories hold experiment-specific scripts/notebooks/configs as
they're implemented; raw outputs still go under `results/`, not here.

## Multi-seed and statistical reporting

Any comparative claim (e.g. "AA-EvidentNet has lower ECE than MaxViT")
must be backed by results aggregated across all seeds in
`configs/experiments.yaml: default_seeds`, with variance/confidence
intervals reported (`src/statistics`), not a single seed's number.

## Status of this document

This is a policy document written before any experiment has been run. It
will need small revisions once `src/training` and `src/evaluation` exist and
their actual output formats are known — update it in place rather than
leaving it aspirational and wrong.

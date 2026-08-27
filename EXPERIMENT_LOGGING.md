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

**Implemented as of Task 6** for baseline training runs
(`src/training/run_baseline.py` + `src/training/logging.py`). Every run
writes, to its own `results/logs/<run_id>/`:

- `run.log` — human-readable, timestamped (UTC ISO8601) line per event
  (run start, per-epoch summary, checkpoint saves, completion/failure)
- `metrics.jsonl` — one JSON record per epoch: `epoch`, `train_loss`,
  `train_accuracy`, `train_macro_f1`, `val_loss`, `val_accuracy`,
  `val_balanced_accuracy`, `val_macro_f1`, `lr`, `elapsed_seconds`,
  `is_best`
- `config.yaml` — the full effective config (model name/architecture,
  seed, device, smoke_test flag, the resolved training config, the
  resolved model config)
- `environment.txt` — `src/utils/env_info.collect_environment_info()`
  output (Python/torch/torchvision/timm versions, CUDA availability, GPU
  name if any, OS/platform)
- `dataset_hash.txt` — SHA-256 of the manifest actually used for training
  (`train_balanced.csv`; a fixed sentinel string for smoke tests, which
  use synthetic data with no real manifest)
- `git_commit.txt` — the commit this run executed at

This directly reuses `src/utils/config.hash_config`, `src/utils/hashing.hash_file`,
and `src/utils/git_info.get_git_commit` rather than reimplementing any of
them.

## Directory conventions

- `results/logs/<run_id>/` — per-run logs, as listed above (flat, not
  nested by experiment/stage name — `run_id` already encodes the model)
- `results/checkpoints/<run_id>/` — `best.pt` (saved whenever
  `monitor_metric` improves) and `latest.pt` (saved every
  `checkpoint_frequency` epochs); each is a self-describing bundle of
  model/optimizer/scheduler state + metadata (see REPRODUCIBILITY.md)
- `results/raw_predictions/<experiment>/<run_id>/` — per-sample
  predictions, logits/evidential parameters, uncertainty scores (a later
  task; not produced by Task 6's training engine, which only trains/
  validates, never predicts-and-dumps)
- `results/tables/` — aggregated, human-readable tables (CSV/Markdown),
  e.g. `model_parameters.csv`
- `results/figures/` — generated plots

Both `results/logs/*` and `results/checkpoints/*` are gitignored (regenerable,
and too large/numerous to version) — `RunLogger` refuses to create a run
directory that already exists, so no run's logs/checkpoints are ever
silently overwritten by another.

`run_id` format (`src/training/logging.generate_run_id`):
`YYYYMMDD_HHMMSS_<model>_seed<seed>[_smoke]_<6-hex>` — e.g.
`20260827_093914_maxvit_seed42_65f791`, or
`20260827_092925_maxvit_seed42_smoke_a0ee86` for a smoke test. The
trailing 6-hex random suffix (not just the timestamp) is what actually
guarantees uniqueness for same-second runs.

## Experiment registry (`experiments/registry.csv`)

**Implemented as of Task 6** (`src/training/registry.py`). Unlike
`results/logs`/`results/checkpoints`, this file IS committed to git — it
is a small, human-readable ledger of every run, never the heavy
artifacts themselves. Columns: `experiment_id, model, seed, config,
checkpoint, test_result, status, notes`.

- `register_run()` appends a new row with `status=running` at the start
  of a run; duplicate `experiment_id`s are rejected (`RegistryError`).
- `update_run()` rewrites that one row in place — to `status=completed`
  (with the best checkpoint's path) on success, or `status=failed` (with
  the exception message in `notes`) if training raises. Every other run's
  row is left untouched.
- `test_result` is **always empty** for a training run — it is only ever
  populated by a future, separate test-evaluation task (`final_test`, not
  yet implemented). A training run must never write to this column.
- `notes` distinguishes run kinds: `smoke_test` for smoke-test runs, or a
  free-text note (e.g. a sanity-check run is tagged something like
  `REAL_DATA_SANITY_CHECK_ONLY_not_a_completed_baseline...`) so a reader
  scanning the registry can never mistake a mechanical check for a real
  result.

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

Training-run logging and the experiment registry (above) are implemented
and match this document exactly (updated in place after Task 6, per this
document's own original instruction not to leave it aspirational once the
real formats were known). `src/training/run_aa_evidentnet.py` (Task 7
completion) follows this exact same logging/registry policy — same
`results/logs/<run_id>/` layout, same `experiments/registry.csv` schema,
same `notes` convention for tagging smoke tests and sanity checks (e.g.
`AA_EVIDENTNET_REAL_DATA_SANITY_CHECK_ONLY_...`) — nothing about this
policy changed to accommodate the proposed model. `results/raw_predictions/`
and the multi-seed/statistical-reporting sections below remain
aspirational — they describe `src/evaluation`/`src/statistics`, neither
of which exists yet.

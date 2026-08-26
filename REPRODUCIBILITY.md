# Reproducibility

## Seeds

The default seed for any single run is:

```text
42
```

Official multi-seed robustness/statistics reporting uses exactly these five
seeds (see `configs/experiments.yaml: default_seeds` and
`src/utils/seeding.SUPPORTED_SEEDS`):

```text
42, 123, 456, 789, 2026
```

`src/utils/seeding.set_seed(seed)` seeds Python's `random`, NumPy, and
PyTorch (CPU and, if available, CUDA), and sets `PYTHONHASHSEED`.

## Deterministic behavior

`set_seed(seed, deterministic=True)` (the default) additionally:

- calls `torch.use_deterministic_algorithms(True, warn_only=True)`
- sets `torch.backends.cudnn.deterministic = True` and
  `torch.backends.cudnn.benchmark = False`
- sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` (required by cuBLAS for
  deterministic GEMM on CUDA)

`warn_only=True` is used deliberately: some ops (especially in future
interpolation/augmentation code) may not have deterministic CUDA kernels.
Rather than crash, we warn and proceed — but any such warning should be
investigated before trusting exact bit-for-bit reproducibility of a given
run.

## Reference environment

Verified in this project's development environment (2026-08-26):

- OS: Windows 11
- Python: 3.10.11
- torch: 2.10.0+**cpu** — no CUDA GPU was available in this environment
  (`torch.cuda.is_available()` returns `False`). If you run on a machine
  with an NVIDIA GPU, install a matching CUDA build of torch instead (see
  https://pytorch.org/get-started/locally/) and expect some numerical
  differences vs. CPU-only runs.
- numpy 2.2.6, pandas 2.3.3, pillow 12.1.1, PyYAML 6.0.3,
  scikit-learn 1.7.2, pytest 9.1.1

Exact pins live in `requirements.txt` / `environment.yml`. Every run should
also capture live environment info via
`src/utils/env_info.collect_environment_info()`, since pinned files describe
intent, not necessarily what was actually installed at run time.

## Configuration hashing

All hyperparameters live in `configs/*.yaml`, not hardcoded in source.
`src/utils/config.load_config` loads a YAML file; `hash_config` produces a
deterministic SHA-256 digest of its content (independent of key order).
Every experiment run should log its resolved config's hash so that any two
runs can be checked for having used identical settings.

## Dataset and manifest hashing

`src/utils/hashing.hash_file` computes a SHA-256 digest of a single file's
bytes; `hash_string` hashes an arbitrary string (used for
`build_split.compute_original_id`). `hash_manifest` combines a list of
`(relative_path, file_hash)` pairs (order-independent) into one digest
representing an entire dataset manifest.

`prepare_dataset` (`src/data/build_split.py`) now produces exactly this
provenance for the original 70/20/10 split: `data/audit/{train,val,test}_manifest_hash.txt`
each hold the SHA-256 of their manifest file, and `data/audit/split_metadata.json`
records the seed, split ratios, `dataset_eligibility.csv`'s hash, the
config hash, git commit, full environment info, and per-manifest hashes
together — so any split can be traced back to the exact eligibility state,
config, and code commit that produced it. The same eligibility manifest +
config + seed always reproduces byte-identical `*_original.csv` files; a
different seed may produce a different (still valid, still leakage-free)
allocation.

## Git provenance

`src/utils/git_info.get_git_commit()` and `get_git_status_summary()` report
the current commit SHA and whether the working tree is clean. These degrade
gracefully (returning `None` / `"not_a_git_repository"`) if run outside a
git repository. Every run record (see `EXPERIMENT_LOGGING.md`) should
include both, and a "dirty" tree should be flagged rather than hidden —
results from an uncommitted, modified codebase are not fully reproducible
until that code is committed.

## What is NOT yet guaranteed

- The original train/val/test split (`prepare_dataset`) is reproducible
  and leakage-checked, but it is built on an eligibility state where 464
  cross-class duplicate groups remain UNRESOLVED (see README.md) — it will
  change as those get human-reviewed, so treat it as provisional, not final.
- Balancing to `target_train_samples_per_class` and augmentation are not
  yet implemented, so there is no reproducibility guarantee for those
  stages yet.
- No model or training loop exists yet, so weight initialization and
  training dynamics reproducibility is not yet applicable.
- CPU-only determinism has been configured defensively but not empirically
  verified end-to-end, since there is no training loop yet to test it
  against.

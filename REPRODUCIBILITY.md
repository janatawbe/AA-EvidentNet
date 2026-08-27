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
- torchvision: 0.25.0+cpu, timm: 1.0.28 (added for Task 5 — model
  infrastructure). **Pinned deliberately**: an unpinned
  `pip install torchvision` resolves the latest torchvision (0.28.0 at
  time of writing), which in turn upgrades torch itself (observed: torch
  2.10.0 -> 2.13.0). To add torchvision without upgrading torch, always
  install both together with torch pinned:
  `pip install torch==2.10.0 torchvision==0.25.0` (0.25.0 is the
  torchvision release that pairs with torch 2.10.0).
- numpy 2.2.6, pandas 2.3.3, pillow 12.1.1, PyYAML 6.0.3,
  scikit-learn 1.7.2, pytest 9.1.1

Exact pins live in `requirements.txt` / `environment.yml`. Every run should
also capture live environment info via
`src/utils/env_info.collect_environment_info()`, since pinned files describe
intent, not necessarily what was actually installed at run time.

## Model reproducibility (Task 5)

- Baseline model architectures (`src/models/factory.py: create_model`) are
  entirely config-driven (`configs/models.yaml: baselines.*` — backbone
  name, `pretrained`, `num_classes`, `dropout`); no architecture constant
  is duplicated in source.
- `python run_pipeline.py model_check` (and the pytest suite) always use
  `pretrained=False` by default — fully offline, deterministic given a
  fixed torch/timm version, no network access or weight download
  required. `feature_dim` and parameter counts in
  `results/tables/model_parameters.csv` are read from the actually
  instantiated model, never hand-typed. The MaxViT variant
  (`maxvit_tiny_tf_224`, pretrained tag `.in1k`) was checked against the
  installed `timm==1.0.28` (`timm.list_models(..., pretrained=True)`)
  rather than assumed to exist.
- `--pretrained` on `model_check` additionally downloads real
  ImageNet-pretrained weights — one of only two places in this project
  that require internet access (the other being a real, non-smoke-test
  `baseline` run, since `configs/models.yaml` defaults to
  `pretrained: true`); neither is ever invoked by the default `pytest`
  suite or by `prepare_dataset`.

## Training reproducibility (Task 6)

- `src/training/trainer.py: Trainer` is the one training engine every
  model uses; nothing architecture-specific lives there. Every
  hyperparameter (`configs/training.yaml`) is config-driven — optimizer
  (AdamW: `lr`, `weight_decay`, `betas`, `eps`), scheduler
  (`ReduceLROnPlateau`: `factor`, `patience`, `min_lr`), early stopping
  (`patience`), `gradient_clip_norm`, `gradient_accumulation_steps`,
  `mixed_precision`, `checkpoint_frequency`, and `monitor_metric`/`mode`
  (used identically for early stopping, LR scheduling, AND "best"
  checkpoint selection — always validation-only, never test).
- **Every checkpoint is self-describing**: model/optimizer/scheduler
  state plus metadata (`model_name`, `architecture`, `num_classes`,
  `seed`, `epoch`, `best_metric`, the full training config, the
  *training* dataset manifest's SHA-256, and the git commit) — see
  `src/training/checkpointing.py: build_checkpoint`. `assert_checkpoint_compatible()`
  raises `CheckpointIncompatibleError` (never silently coerces) if a
  resume target's `model_name`/`num_classes` don't match.
- **Every run gets its own directory** —
  `results/logs/<run_id>/` and `results/checkpoints/<run_id>/` — and
  `RunLogger` refuses to reuse an existing `run_id` directory, so no run's
  logs/checkpoints can ever silently overwrite another's.
  `run_id = YYYYMMDD_HHMMSS_<model>_seed<seed>[_smoke]_<6-hex>`; the
  trailing random suffix (not just the timestamp) is what actually
  guarantees uniqueness under same-second collisions.
- `experiments/registry.csv` (committed to git, unlike `results/logs`/
  `results/checkpoints`) is the append-only ledger of every run — status
  `running` → `completed`/`failed`, `test_result` always empty until a
  future, separate test-evaluation task populates it.
- AMP (`torch.amp`) is only ever actually active on CUDA — requesting it
  on CPU is downgraded to disabled and logged as such (`Trainer.amp_enabled`),
  never falsely reported as active. Reproducibility of the numeric
  training trajectory itself (not just the mechanics) has not been
  empirically verified end-to-end on CUDA, since this development
  environment has none.
- **No baseline has completed a full training run** in this environment
  (CPU-only) — see README.md "The real baseline run" for exactly what was
  and wasn't done (a full smoke test for all three baselines, a real-data
  loading/forward sanity check, and one real-data training run capped at
  3 batches/epoch to prove the pipeline works, explicitly logged and
  registered as a sanity check, not a result). No baseline accuracy,
  F1, or other performance number exists anywhere in this repository.

## AA-EvidentNet architecture reproducibility (Task 7)

- `src/models/aa_evidentnet.py: AAEvidentNet` is entirely config-driven
  (`configs/models.yaml: proposed.aa_evidentnet` — `global_backbone`,
  `embedding_dim`, `local_feature_dim`, `pretrained`, `dropout`,
  `num_classes`); no architecture constant is duplicated in source. The
  global branch's native feature width is always read from the
  instantiated backbone (`global_backbone.num_features`), never assumed.
- `create_model("aa_evidentnet", config)` goes through the exact same
  factory as the three baselines (`src/models/factory.py`); unit tests use
  `pretrained=False` exclusively, so they never require internet access or
  a weight download (same policy as `model_check` and the baseline smoke
  tests).
- The adaptive-fusion gate (`alpha = sigmoid(a single nn.Parameter)`) is a
  real trainable parameter — included in `state_dict()`, and verified to
  actually change value when the model is trained for even one tiny
  synthetic epoch through the unmodified `Trainer`. Given the same random
  seed, two freshly-constructed models produce numerically identical
  outputs (`torch.manual_seed` before construction), consistent with every
  other model in this project.
- **Verified compatible, unmodified, with all Task 6 infrastructure**:
  `Trainer.fit()` (forward/backward/optimizer step/validation), and
  `build_checkpoint`/`save_checkpoint`/`load_checkpoint`/
  `assert_checkpoint_compatible`/`restore_training_state` (checkpoint
  round-trip) — no special-casing for AA-EvidentNet was needed anywhere in
  `src/training/`.
- **No training run — smoke, sanity-check, or real — has been performed
  for AA-EvidentNet.** Only architecture construction, forward-pass
  correctness (shapes, finiteness, fusion arithmetic), and infrastructure
  compatibility have been verified, all with `pretrained=False` and
  synthetic tensors. The CS-SupCon and EDL training objectives do not
  exist yet, so there is nothing to train against beyond plain
  cross-entropy (used only in the `Trainer`-compatibility test, not as a
  declared training methodology for this model).

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

`prepare_dataset` (`src/data/build_split.py`) produces exactly this
provenance for the original 70/20/10 split: `data/audit/{train,val,test}_manifest_hash.txt`
each hold the SHA-256 of their manifest file, and `data/audit/split_metadata.json`
records the seed, split ratios, `dataset_eligibility.csv`'s hash, the
config hash, git commit, full environment info, and per-manifest hashes
together — so any split can be traced back to the exact eligibility state,
config, and code commit that produced it. The same eligibility manifest +
config + seed always reproduces byte-identical `*_original.csv` files; a
different seed may produce a different (still valid, still leakage-free)
allocation.

The same `prepare_dataset` run then produces the balanced training set
(`src/data/generate_balanced_dataset.py`): `data/audit/train_balanced_manifest_hash.txt`
holds `train_balanced.csv`'s SHA-256, and `data/audit/balanced_dataset_metadata.json`
records the seed, target samples/class, `train_original.csv`'s hash, the
config hash (including a dedicated hash of just the augmentation
sub-config), git commit, and environment info. Every generated sample's ID
and its augmentation parameters are derived deterministically from
`(parent_original_id, canonical_class, augmentation_index, seed,
augmentation_config_hash)` — see `compute_generated_id` and `_sample_rng`
in `src/data/generate_balanced_dataset.py` — so generation is independent
of processing order and the same inputs always reproduce byte-identical
generated images and manifest. A different seed may produce different
(still valid) augmented images.

## Git provenance

`src/utils/git_info.get_git_commit()` and `get_git_status_summary()` report
the current commit SHA and whether the working tree is clean. These degrade
gracefully (returning `None` / `"not_a_git_repository"`) if run outside a
git repository. Every run record (see `EXPERIMENT_LOGGING.md`) should
include both, and a "dirty" tree should be flagged rather than hidden —
results from an uncommitted, modified codebase are not fully reproducible
until that code is committed.

## What is NOT yet guaranteed

- The original train/val/test split AND the balanced training set
  (`prepare_dataset`) are both reproducible and leakage-checked, but both
  are built on an eligibility state where 464 cross-class duplicate groups
  remain UNRESOLVED (see README.md) — both will change as those get
  human-reviewed, so treat them as provisional, not final.
- The training engine and checkpoint/logging/registry mechanics are
  verified end-to-end (smoke tests + a capped real-data sanity check),
  but no baseline has completed a full training run, so weight-trajectory
  reproducibility (does training produce the same result run-to-run on
  real data, over real epochs) has not been empirically verified — only
  mechanics have.
- CPU-only determinism has been configured defensively (deterministic
  runtime preprocessing, seeded dataloader shuffling, seeded per-sample
  augmentation) but full training-loop determinism on CUDA (where AMP
  actually activates) has not been tested, since this environment has no
  GPU.

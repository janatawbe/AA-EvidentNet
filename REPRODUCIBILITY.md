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
- **Since Task 7's completion**, a real-data (capped) sanity-check
  training run has been performed for AA-EvidentNet through the combined
  classification + CS-SupCon + EDL objective — see "AA-EvidentNet
  training orchestration reproducibility (Task 7 completion)" below. No
  full training run has been performed; only architecture construction,
  forward-pass correctness, infrastructure compatibility, and this one
  capped real-data run have been verified.
- **Since Task 9**, `AAEvidentNet` also attaches an `EvidentialHead`
  (`src/losses/evidential.py`) — a separate `Linear(embedding_dim,
  num_classes)` on the same fused embedding, entirely independent of the
  pre-existing classifier/`logits`. Verified real parameter count
  (`pretrained=False`, default config): **30,814,557** total (was
  30,811,987 before Task 9; the 2,570-parameter difference is exactly the
  new `Linear(256, 10)` evidential head). `AAEvidentNetOutput`'s new
  fields (`evidential_raw`, `evidence`, `dirichlet_alpha`, `probabilities`,
  `uncertainty`) are populated only when `return_features=True`, so the
  plain `model(images)` path the `Trainer` uses for its forward/backward
  step is unaffected — re-verified after this change:
  `test_aa_evidentnet_trains_via_existing_trainer_without_modification`
  and the checkpoint round-trip test both still pass unmodified.

## CS-SupCon reproducibility (Task 8)

- `src/losses/cs_supcon.py: CSSupConLoss` is entirely config-driven
  (`configs/losses.yaml: cs_supcon` — `enabled`, `temperature`,
  `loss_weight`, `ambiguity_weight`, `ambiguity_pairs`); no ambiguity
  relationship or hyperparameter is hard-coded in the loss implementation.
  Ambiguity pairs are configured by canonical class NAME and resolved to
  indices via the project's single canonical ordering
  (`src/data/dataset.py: build_class_to_idx`, alphabetical sort of
  `configs/dataset.yaml: class_names`) — there is no second, independently
  maintained class-to-index mapping.
- `resolve_ambiguity_pairs`/`load_cs_supcon_settings` validate strictly and
  fail with `CSSupConConfigError` (never a silent default or best-effort
  correction) on: an unknown class name, a class paired with itself, a
  duplicate/conflicting pair (order-independent), or a non-positive
  `temperature`/`ambiguity_weight`/negative `loss_weight`.
- Numerical stability uses the standard log-sum-exp trick (per-row max
  similarity subtracted, detached, before exponentiating); an anchor with
  no same-class positive in its batch (e.g. a batch with one sample per
  class) contributes a finite zero rather than `NaN`/`Inf` from an
  empty-set division.
- **46/46 new tests pass** (`tests/test_cs_supcon.py`), covering: scalar/
  finite output, gradient propagation (including through an upstream
  `nn.Linear`), same-class positive-pair recognition, self-pair exclusion
  from the denominator, degenerate batches (no positive for an anchor,
  single-class batch, batch size 1), arbitrary batch sizes, pre-normalized
  vs. raw embeddings, extreme-scale numerical stability, ambiguity
  weighting actually changing the loss (and leaving it unchanged when no
  configured ambiguous class is present in the batch), temperature and
  ambiguity-weight sensitivity, determinism across repeated/independent
  calls, and config/input validation failures (invalid class names,
  self-paired/duplicate ambiguity pairs, invalid temperature/weights,
  out-of-range labels, malformed embedding shapes). All tests use tiny
  synthetic tensors — no real dataset, `data/raw/`, or `test_original.csv`
  access anywhere in this test module.
- **Since Task 7's completion**, CS-SupCon is also used (unmodified) as
  one term of the combined AA-EvidentNet training objective
  (`src/losses/combined.py`) and has been exercised on real data via a
  capped sanity-check run — see "AA-EvidentNet training orchestration
  reproducibility (Task 7 completion)" below. `temperature=0.1`,
  `ambiguity_weight=2.0`, and `loss_weight=1.0` in `configs/losses.yaml`
  remain PROVISIONAL and have not been experimentally tuned.

## EDL reproducibility (Task 9)

- `src/losses/evidential.py` implements the Dirichlet-based Evidential
  Deep Learning formulation of Sensoy, Kaplan, and Kandemir (2018) — one
  of several published EDL formulations, not claimed to be universally
  optimal. Pipeline: `evidence = softplus(raw_output)` (>= 0), `alpha =
  evidence + 1` (>= 1, Dirichlet parameters), `S = sum(alpha)`,
  `probabilities = alpha / S`, `uncertainty = num_classes / S` (always in
  `(0, 1]`). Because `alpha >= 1` always, every `lgamma`/`digamma`
  evaluation in the loss (both the direct term and the KL regularizer) is
  guaranteed to operate on inputs `>= 1`, where both functions are smooth
  and well-behaved — this is a property of the construction, not a
  defensive epsilon patch.
- Entirely config-driven (`configs/losses.yaml: edl` — `enabled`,
  `loss_weight`, `kl_annealing_epochs`, `kl_weight_max`, `epsilon`); no
  hyperparameter is hard-coded in the loss implementation.
  `load_edl_settings` validates strictly and fails with
  `EvidentialConfigError` (never a silent default or best-effort
  correction) on a non-positive `kl_annealing_epochs`/`epsilon`, or a
  negative `loss_weight`/`kl_weight_max`. `edl_loss`/`EDLLoss` separately
  validate their tensor inputs (`ValueError` for out-of-range labels,
  a batch-size mismatch, a malformed `alpha` shape, or an `alpha` value
  below 1).
- The evidential head (`EvidentialHead`, a `Linear(embedding_dim,
  num_classes)`) is attached to `AAEvidentNet` on the same fused embedding
  the existing classifier uses, but is a **separate** layer — the ordinary
  `logits` output is bit-for-bit unaffected (verified directly:
  `test_ordinary_logits_unaffected_by_evidential_head`), and the
  evidential outputs are populated only when `return_features=True`.
- **67/67 new tests pass** (`tests/test_evidential.py`), covering: head
  construction and output shapes, evidence non-negativity, `alpha =
  evidence + 1`, probabilities summing to 1, uncertainty shape/finiteness/
  valid range (`(0, 1]` for `K=10`), the qualitative more-evidence-means-
  less-uncertainty relationship (including near-zero and very large
  evidence), extreme raw-output values (`+-1e6`) remaining finite,
  arbitrary and degenerate batches (batch size 1, single-class-label
  batches, up to 33 samples), loss scalar/finiteness/differentiability
  (including through an upstream `nn.Linear`), the loss penalizing
  confident-but-incorrect evidence more than confident-and-correct
  evidence, the KL term penalizing unjustified wrong-class evidence,
  annealing behavior (loss changes with epoch until `kl_annealing_epochs`,
  then plateaus), determinism, and config/input validation failures. 14
  further integration tests were added to `tests/test_aa_evidentnet.py`
  covering the evidential head as attached to the full model (shapes,
  non-negativity, `alpha=evidence+1`, probabilities-sum-to-1,
  uncertainty range, no NaN/Inf, gradient propagation through the whole
  model, state_dict round-trip, and that ordinary logits are unaffected).
  All tests use tiny synthetic tensors — no real dataset, `data/raw/`, or
  `test_original.csv` access anywhere in either test module.
- **Since Task 7's completion**, EDL is also used (unmodified) as one term
  of the combined AA-EvidentNet training objective
  (`src/losses/combined.py`) and has been exercised on real data via a
  capped sanity-check run — see below. `kl_annealing_epochs=10`,
  `kl_weight_max=1.0`, `loss_weight=1.0`, and `epsilon=1e-8` in
  `configs/losses.yaml` remain PROVISIONAL and have not been
  experimentally tuned.

## AA-EvidentNet training orchestration reproducibility (Task 7 completion)

- **Combined objective** (`src/losses/combined.py:
  CombinedAAEvidentNetLoss`): `L_total = L_classification +
  cs_supcon_weight * L_CS-SupCon + edl_weight * L_EDL`, where
  `cs_supcon_weight`/`edl_weight` are read directly from the pre-existing
  `configs/losses.yaml: cs_supcon.loss_weight` / `edl.loss_weight` fields
  (no new weight parameters were invented). `configs/losses.yaml:
  baseline.label_smoothing` is now consumed by the classification term;
  `baseline.class_weighting` values other than `none` raise
  `CombinedObjectiveConfigError` rather than being silently ignored, since
  that scheme is not implemented. Either loss term can be disabled
  entirely via its own `enabled: false` (the module is not constructed and
  the weight is forced to `0.0`, not merely left at a nonzero-but-unused
  value).
- **Trainer change**: `Trainer.__init__` gained one new optional
  constructor argument, `return_features: bool = False`. When `True`
  (used only by `run_aa_evidentnet_training`), each forward pass calls
  `model(images, return_features=True)` and passes the *full* output to
  `criterion(output, labels)` instead of `criterion(logits, labels)`;
  `Trainer.fit()` additionally calls `criterion.set_epoch(epoch)` once per
  epoch if the criterion defines that method (used for EDL's KL-annealing
  coefficient). Every baseline (`return_features=False` by default, and
  `nn.CrossEntropyLoss` has no `set_epoch`) is provably unaffected — the
  full pre-existing `tests/test_trainer.py` (29 tests) and
  `tests/test_run_baseline.py` (7 tests) suites still pass unmodified, and
  5 new tests were added directly verifying the default/opt-in behavior.
- **`src/training/run_aa_evidentnet.py: run_aa_evidentnet_training`**
  mirrors `run_baseline.py` exactly (same `train_balanced.csv`/
  `val_original.csv` manifests, same checkpointing/logging/registry/
  resume/smoke-test conventions, same test-set lock — verified by a
  dedicated regression test that no path under a completed run's
  directories mentions `test_original`) but constructs
  `create_model("aa_evidentnet", ...)` and the combined-objective
  criterion instead. Wired into the CLI as `python run_pipeline.py train
  --model aa_evidentnet` (`--smoke-test`/`--resume`/`--device`/
  `--batch-size`/`--epochs`/`--num-workers` all supported identically to
  `baseline`).
- **43 new tests** covering this work: `tests/test_combined_loss.py` (27:
  settings loading/validation, scalar/finite output, each term actually
  contributing, weight/epoch sensitivity, gradients propagating through
  all three components including into a real `AAEvidentNet`'s fusion gate
  `alpha`, missing-field validation, determinism, ambiguity-pair
  end-to-end effect), `tests/test_trainer.py` (+5: `return_features`
  plumbing, full-output criterion, `set_epoch` hook), and
  `tests/test_run_aa_evidentnet.py` (11: end-to-end smoke test, real
  checkpoint/state-dict contents, per-term enable/disable ablation via
  config, invalid-config handling, resume with a compatible/incompatible
  checkpoint, failure-path registry updates, test-set lock).
- **Real-data sanity check performed** (not a completed training run):
  `run_id=20260827_152101_aa_evidentnet_seed42_c4f592`, seed 42, real
  `train_balanced.csv` (20,000 rows, hash
  `ae79e693a2b1ef0632f71f05c1ff927ad9d3059784057d9c1ecef529c1847002`) /
  `val_original.csv` (880 rows), capped at 3 training + 3 validation
  batches, 1 epoch, git commit `e5129b570cfbb76de24b1d1c6055d838ce471d80`,
  config hash
  `18bafee80830487bc0aff104085a7a0165f88707d1a31ebae84ab336e3159e88`. All
  three loss components were finite and non-zero
  (`classification=2.336, cs_supcon=3.048, edl=2.633, total=8.017`),
  proving the combined objective genuinely backpropagates through the
  real model on real images, not just in synthetic unit tests. Registered
  in `experiments/registry.csv` with notes explicitly reading
  `AA_EVIDENTNET_REAL_DATA_SANITY_CHECK_ONLY_...` so it is never mistaken
  for a completed run. An earlier mis-configured attempt at this check
  (`run_id=20260827_150813_aa_evidentnet_seed42_393841`, which omitted the
  epoch cap and began running the full 50-epoch configured budget) was
  manually stopped and its registry row corrected to `status=failed` with
  an explanatory note, rather than deleted or left showing a stale
  `running` status.
- **No full AA-EvidentNet training run was attempted.** Extrapolating
  from the sanity check (~95s wall-clock for 3 train + 3 validation
  batches combined, CPU-only, `batch_size=16`, 4 physical cores,
  `torch.cuda.is_available() == False`): a full epoch over
  `train_balanced.csv` (1,250 batches) plus `val_original.csv` (55
  batches) is on the order of several hours, against a 50-epoch
  configured budget with early-stopping patience 10 — at least as
  expensive as the baseline MaxViT case (`configs/training.yaml`), and
  more so given AA-EvidentNet's additional local branch, evidential head,
  and CS-SupCon/EDL loss computation every batch. No full run was started
  in this CPU-only environment. **No AA-EvidentNet performance number
  exists anywhere in this repository.**

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
- The AA-EvidentNet training orchestration and combined objective (Task 7
  completion) are verified end-to-end mechanically (unit tests + one
  capped real-data sanity check), but no full AA-EvidentNet training run
  has been performed, so neither its weight-trajectory reproducibility
  nor any performance/calibration/uncertainty-quality claim can be made —
  only "the combined objective backpropagates correctly through the real
  model on real data" has been verified.

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

## Colab/GPU preparation (no training performed)

See `docs/COLAB_SETUP.md` for the full guide. Summary of what changed and
what was verified, all **preparation only** — no model was trained as
part of this work, on CPU or GPU:

- **Device selection and mixed precision were already fully correct** for
  CUDA before this work (verified by static code review, since this
  development machine's torch build has no CUDA support at all —
  `model.to(torch.device("cuda"))` itself raises `AssertionError: Torch
  not compiled with CUDA enabled`, so genuine CUDA execution is not
  possible here): `resolve_device("auto")` already picks CUDA when
  available and falls back to CPU otherwise with no code path assuming a
  specific GPU model; `Trainer.amp_enabled = bool(config.mixed_precision)
  and self.device.type == "cuda"` already only activates AMP on CUDA;
  `torch.amp.autocast(device_type=self.device.type, ...)` already tracks
  whichever device is actually in use. `--device {auto,cpu,cuda}`
  continues to work unchanged. New tests
  (`tests/test_trainer.py::test_resolve_device_auto_picks_cuda_when_available`
  and related) verify this decision logic via monkeypatching
  `torch.cuda.is_available`, since real CUDA execution cannot be tested on
  this machine — actual GPU execution must happen on Colab.
- **Fixed**: checkpoints did not save the AMP `GradScaler`'s state.
  `build_checkpoint`/`restore_training_state`
  (`src/training/checkpointing.py`) now accept an optional `scaler`
  parameter; `run_baseline.py`/`run_aa_evidentnet.py` pass `trainer.scaler`
  through. This only matters for CUDA + `mixed_precision: true` (on CPU
  the scaler is always disabled and its state is trivial); without it, a
  CUDA run resumed from a checkpoint would silently restart AMP's
  adaptive loss scale from its default rather than where it left off.
  Fully backward compatible: a checkpoint written before this change has
  no `scaler_state_dict` key at all, and `restore_training_state` treats
  that exactly like an explicit `None` (skips the restore, no error).
- **Added**: `--dataset-config` (`run_pipeline.py: baseline`/`train`,
  default `configs/dataset.yaml`) — lets a Colab run point `paths.raw_dir`/
  `paths.processed_dir`/`paths.manifests_dir`/`paths.audit_dir` at an
  externally-mounted location (e.g. Google Drive) via a copy of
  `configs/dataset.yaml` with only its `paths:` section changed, with no
  source-code edit required. `class_names`, `class_directory_mapping`,
  `split:` ratios, `target_train_samples_per_class`, and `augmentation:`
  must remain byte-identical between the two files — this flag changes
  *where the data is read from*, never *what the data or methodology is*.
  Manifests (`data/manifests/*.csv`) already store paths relative to
  `raw_dir`/`processed_dir`, so they remain valid unchanged regardless of
  where those roots point.
- **No dataset, split, class definition, augmentation methodology, or
  loss/architecture hyperparameter was changed.** `data/raw/` (5,335
  files) was not touched. `test_original.csv` continues to never be
  referenced by any training code path — unchanged by this work.
- The already-stopped ResNet50 run
  (`20260827_153917_resnet50_seed42_5f8341`, `results/checkpoints/.../best.pt`,
  epoch 2, `val_macro_f1=0.8811`) was verified intact and untouched
  throughout this preparation work, and remains resumable exactly as
  before (its checkpoint predates the `scaler_state_dict` field, so
  resuming it will simply skip restoring scaler state, per the backward
  compatibility described above).

## Final test evaluation reproducibility (Task 8)

**Naming note**: this is the *original* project task-numbering's Task 8
(the final held-out test evaluation pipeline) — distinct from the
"CS-SupCon reproducibility (Task 8)" section earlier in this document,
which used this project's own internal session numbering (established
before a later audit identified the mismatch between the two numbering
schemes; see the Tasks 1-9 completion audit). No section heading above
was renamed to avoid rewriting already-accurate history — read heading
numbers in context, not as a single global sequence.

- `src/evaluation/final_test.py: run_final_test` (CLI: `python
  run_pipeline.py final_test --model <name> --checkpoint <path>`)
  evaluates a single frozen checkpoint for any of the four registered
  models. Reuses, unmodified: `src/models/factory.py: create_model`,
  `src/training/checkpointing.py: load_checkpoint` /
  `assert_checkpoint_compatible` / `restore_training_state`,
  `src/data/dataset.py: RetinalDataset.from_manifest(...,
  expected_split="test", require_all_original=True)`, and
  `src/data/dataloaders.py: build_eval_dataloader`. No training engine
  (`Trainer`) is used — there is no optimizer, no scheduler, and no
  backward pass anywhere in this module; inference runs once under
  `torch.inference_mode()` in plain fp32 (deliberately no AMP, even on
  CUDA), and `model.eval()` is called before any forward pass.
- **Test-set discipline**: this module is the only place in the codebase
  (besides its own test) that loads `data/manifests/test_original.csv`.
  It is never used for training, model selection, hyperparameter tuning,
  calibration fitting, or threshold selection — `final_test` only ever
  reads a manifest and a checkpoint and writes results out.
- **Core metrics** (`src/evaluation/metrics.py`, reusing scikit-learn
  throughout, no metric math reimplemented): overall accuracy, balanced
  accuracy, macro precision/recall/F1 (sklearn's `zero_division=0`
  convention, matching `src/training/metrics.py`), macro ROC-AUC, macro
  PR-AUC; per-class precision/recall/specificity/F1/ROC-AUC/PR-AUC; a
  full K×K confusion matrix. Every sklearn call is passed
  `labels=range(num_classes)` explicitly, so a class absent from a given
  evaluation never silently reshapes a result. Undefined per-class
  metrics (zero positive or zero negative samples) are reported as
  `None` with an explicit reason string, never coerced to 0.0.
- **Raw per-sample export**: `results/raw_predictions/<eval_run_id>/predictions.csv`
  — `sample_id` is the manifest's `original_id` (SHA-256-derived,
  content-stable — identical across all four models' evaluations of the
  same physical image, enabling exact sample-by-sample alignment for
  later paired statistical tests), plus true/predicted class, `correct`,
  `logit_0..9`, `prob_0..9` (softmax, identical formulation for all four
  models), and for AA-EvidentNet additionally `evidence_0..9`,
  `dirichlet_alpha_0..9`, `evidential_prob_0..9`, `uncertainty` — all
  read directly from `AAEvidentNetOutput`, never recomputed with a
  different formulation. Before writing, `final_test` verifies the
  exported row count equals the loaded manifest's row count and every
  `sample_id` is unique, raising `FinalTestError` (refusing to write
  anything) if either check fails.
- **Class ordering**: `0..9` always means
  `sorted(configs/dataset.yaml: class_directory_mapping.keys())` — the
  same ordering `src.data.dataset.build_class_to_idx` uses everywhere
  else. Recorded once, explicitly, in every run's `metadata.json:
  class_names` rather than repeated (and risking drift) in column
  headers.
- **Provenance metadata** (`results/tables/<eval_run_id>/metadata.json`):
  model name, the checkpoint's inferred training `run_id` (from its
  parent directory name — a path convention, not a field stored inside
  the checkpoint itself, and labeled as such), training seed (from
  checkpoint metadata) and the eval invocation's own seed, checkpoint
  path + SHA-256, checkpoint's saved epoch/monitor-metric/best-metric,
  test manifest path + SHA-256, exact class-name ordering, sample count,
  dataset/models/evaluation config paths + a combined config hash, git
  commit, timestamp, device, and every output path.
- **Outputs never collide**: each invocation gets a fresh `eval_run_id`
  (`finaltest_<model>_seed<seed>_<timestamp>_<6-hex>`, the same
  `generate_run_id` mechanism training runs already use) — two
  evaluations of the same checkpoint produce two independent,
  non-overwriting output directories.
- **Registry**: on success, if the checkpoint's inferred training
  `run_id` matches an existing `experiments/registry.csv` row, that
  row's `test_result` column is updated via the existing `update_run()`
  with a concise summary string. `register_run()` (which creates a new
  row) is never called by this module — an unmatched `run_id` is
  reported (`FinalTestSummary.registry_updated=False`), not guessed at
  or fabricated. On any failure, the registry is never touched.
- **83 new tests** (`tests/test_evaluation_metrics.py`: 26,
  `tests/test_final_test.py`: 21, plus CLI dispatch tests in
  `tests/test_cli.py`), covering hand-verified metric correctness
  (confusion matrix, accuracy, balanced accuracy, macro
  precision/recall/F1, per-class precision/recall/specificity/F1,
  perfect-separation and undefined-case ROC-AUC/PR-AUC), the full
  `run_final_test` orchestration (schema, sample-id uniqueness,
  probability normalization, AA-EvidentNet evidential dimensions,
  metadata hash correctness, checkpoint-file-never-modified,
  `model.eval()` called, no optimizer ever constructed, no backward pass
  ever called, non-test/augmented manifest rejection, incompatible
  checkpoint rejection, registry update/non-update/untouched-on-failure,
  non-colliding output directories), and CLI parsing/dispatch. All tests
  use tiny synthetic fixtures under `tmp_path` — **none ever load
  `data/manifests/test_original.csv` or `data/raw/`.**
- **No real inference was performed on `test_original.csv`** as part of
  implementing or testing Task 8 — confirmed by `test_original.csv`'s
  unchanged mtime and `data/raw/`'s unchanged file count (5,335)
  throughout this work.

## Robustness evaluation reproducibility

**Naming note**: same caveat as the "Final test evaluation reproducibility
(Task 8)" section above — this feature was requested and implemented as
its own, separately-scoped unit of work (not tied to a single task number
in either numbering scheme). It is a later, additional test-time analysis
on top of Task 8's already-frozen checkpoints; it does not replace, modify,
or rerun anything Task 8 produced.

- `src/evaluation/robustness.py: run_robustness_evaluation` (CLI: `python
  run_pipeline.py robustness --model <name> --checkpoint <path>`)
  evaluates a single already-frozen checkpoint (any of the four registered
  models) against fixed, predefined image degradations. Reuses,
  unmodified: `src/evaluation/final_test.py: ALL_MODEL_NAMES` /
  `_effective_model_config`, `src/evaluation/metrics.py:
  compute_overall_metrics`, `src/models/factory.py: create_model`,
  `src/training/checkpointing.py: load_checkpoint` /
  `assert_checkpoint_compatible` / `restore_training_state`,
  `src/data/dataset.py: RetinalDataset.from_manifest(...,
  expected_split="test", require_all_original=True)`, and
  `src/data/dataloaders.py: build_eval_dataloader`. No optimizer, no
  scheduler, and no backward pass anywhere in this module; inference runs
  under `torch.inference_mode()` in plain fp32 (no AMP), and `model.eval()`
  is called before any forward pass — identical guarantees to `final_test`.
- **Test-manifest handling**: `robustness` is the one other place in the
  codebase (besides `final_test` and its own tests) that reads
  `data/manifests/test_original.csv`, using the identical safeguard
  (`expected_split="test"`, `require_all_original=True`). It is never used
  for training, model selection, hyperparameter tuning, calibration
  fitting, or threshold selection. `data/raw/` and `test_original.csv` are
  opened strictly read-only and are never modified — verified in
  `tests/test_robustness.py` by hashing every raw image file and the
  manifest before and after a full evaluation run.
- **Degradations applied in memory only**: each degradation is applied to
  an already resized/center-cropped/`ToTensor`'d `[0,1]` image tensor
  (`src/data/transforms.py: build_pre_normalize_transform`, a new,
  additive, behavior-preserving refactor of the module's existing
  `_build_transform` — `tests/test_transforms.py`'s original 10 tests
  still pass unchanged, confirming no existing transform behavior
  changed), strictly before normalization (`normalize_tensor`, applied
  explicitly afterward). Fixed, predefined severities
  (`DEFAULT_DEGRADATION_SEVERITIES`, mirrored in `configs/evaluation.yaml:
  robustness.degradations`):
  - `brightness`: 0.70, 0.85, 1.15, 1.30 (`torchvision.transforms.functional.adjust_brightness`)
  - `contrast`: 0.70, 0.85, 1.15, 1.30 (`adjust_contrast`)
  - `gaussian_noise`: 0.02, 0.05, 0.10 (std dev in `[0,1]` pixel space)
  - `gaussian_blur`: 0.5, 1.0, 2.0 (sigma; kernel size `2*ceil(3*sigma)+1`)
  - `reduced_resolution`: downsample to 168, 112, or 56 px, then bilinear-upsample back to the configured image size (224)
  Severities are never tuned from an observed result — changing them after
  seeing a robustness number would defeat the purpose of the check.
- **Deterministic Gaussian noise**: `torch.Generator().manual_seed(...)`
  seeded from the first 8 hex digits of `sha256(f"{eval_seed}:{sample_id}:gaussian_noise:{severity}")`
  — reruns with the same evaluation seed draw byte-identical noise for the
  same sample at the same severity, independent of batch order, device, or
  `num_workers`. Verified in `tests/test_robustness.py` (same
  seed/sample/severity → identical output; different sample, severity, or
  base seed → different output).
- **Metrics** (per `model`/`degradation`/`severity`, via the same
  `compute_overall_metrics` `final_test` uses): `accuracy`,
  `balanced_accuracy`, `macro_f1`; AA-EvidentNet additionally reports
  `mean_uncertainty` (mean of the model's own per-sample evidential
  uncertainty, `K / sum(alpha)`, read directly from `AAEvidentNetOutput` —
  never recomputed with a different formulation), left blank for the
  three baselines. An optional `clean_reference` row (undegraded
  inference, computed internally within the same run) is included by
  default for same-run comparison — it never reads from or writes to
  Task 8's own clean `final_test` outputs.
- **Outputs are entirely separate from Task 8's**: every invocation
  generates a fresh `robustness_run_id`
  (`robustness_<model>_seed<seed>_<timestamp>_<6-hex>`, the same
  `generate_run_id` mechanism training/`final_test` already use) and
  writes only to `results/robustness/<robustness_run_id>/` — never to
  `results/raw_predictions/` or any `results/tables/<eval_run_id>/`:
  `robustness_metrics.csv` (columns: `model, degradation, severity, n,
  accuracy, balanced_accuracy, macro_f1, mean_uncertainty`) and
  `metadata.json` (model name/architecture, the checkpoint's inferred
  training `run_id`, training seed and eval invocation seed, checkpoint
  path + SHA-256, checkpoint's saved epoch/monitor-metric/best-metric,
  test manifest path + SHA-256, exact class-name ordering, sample count,
  device, the full degradation/severity table actually used, dataset/
  models/evaluation config paths + a combined config hash, git commit,
  timestamp, and every output path). Two invocations never collide.
- **Registry is never touched**: unlike `final_test`, `robustness` does
  not call `register_run()` or `update_run()` at all — `experiments/
  registry.csv` is neither read nor written by this module.
- **45 new tests** (`tests/test_robustness.py`), plus 3 dedicated CLI
  parsing/dispatch tests in `tests/test_cli.py`, covering: shape/range
  preservation for every one of the 17 required degradation/severity
  combinations, rejection of unknown degradations and of severities
  outside the fixed table, rejection of a malformed (non-`[C,H,W]`)
  image tensor, deterministic Gaussian noise (same inputs -> identical
  output; different sample/severity/seed -> different output), raw image
  files and the test manifest verified byte-unchanged after a full
  evaluation, test-manifest safeguard enforcement (non-test-split and
  augmented-sample rejection, missing-manifest error), unknown-model and
  incompatible-checkpoint rejection, `model.eval()` called, no optimizer
  ever constructed, no backward pass ever called, outputs verified under
  `results/robustness/` and never under `results/raw_predictions/`,
  metrics-schema and row-count checks, AA-EvidentNet `mean_uncertainty`
  populated vs. baseline `mean_uncertainty` blank, `clean_reference`
  row present/absent per `include_clean_reference`, non-colliding output
  directories, and CLI dispatch. All tests use tiny synthetic fixtures
  under `tmp_path` (most with a reduced 1-2-condition degradation table
  purely to keep runtime fast) — **none run inference against the real
  438-image `data/manifests/test_original.csv`.** A dedicated test
  (`test_real_config_and_defaults_use_exact_required_severities`)
  separately confirms the real `configs/evaluation.yaml` and
  `DEFAULT_DEGRADATION_SEVERITIES` both carry the exact, unmodified
  required severities.
- **No real robustness inference was performed on `test_original.csv`**
  as part of implementing or testing this feature — confirmed by
  `test_original.csv`'s unchanged mtime and `data/raw/`'s unchanged file
  count (5,335) throughout this work.

## Feature-distance OOD detector + EDL uncertainty reproducibility

**Motivation**: the robustness evaluation above surfaced a failure mode -
AA-EvidentNet's own EDL uncertainty *decreases* under severe Gaussian
noise (std=0.10) even as accuracy collapses to ~13%. This module
(`src/evaluation/ood_uncertainty.py`, `python run_pipeline.py
ood_uncertainty --model aa_evidentnet --checkpoint <path>`) is a new
evaluation component, AA-EvidentNet only, that never retrains, fine-tunes,
or modifies the checkpoint's weights, and never overwrites anything
`final_test.py` or `robustness.py` already produced. The current frozen
final-test result (87.90% accuracy) is unaffected.

- **Reuses, unmodified**: `AAEvidentNetOutput.embedding` (the same fused
  global+local representation the classifier and EDL head already share -
  no new forward path added to `src/models/aa_evidentnet.py`),
  `src/evaluation/final_test.py: _effective_model_config`,
  `src/evaluation/robustness.py: DEFAULT_DEGRADATION_SEVERITIES` /
  `apply_degradation` / `CLEAN_REFERENCE_LABEL` / `_iter_conditions` (the
  same 17 fixed degradation/severity combinations, not redefined here),
  `src/training/checkpointing.py: load_checkpoint` /
  `assert_checkpoint_compatible` / `restore_training_state`,
  `src/data/dataset.py: RetinalDataset.from_manifest`. No optimizer, no
  scheduler, no backward pass anywhere in this module; inference runs
  under `torch.inference_mode()` in plain fp32, `model.eval()` called
  before any forward pass - identical guarantees to `final_test`/
  `robustness`.
- **Calibration uses train_original.csv and val_original.csv ONLY - never
  test_original.csv** for any decision:
  1. `compute_class_prototypes`: one forward pass over `train_original.csv`
     (the same manifest training already used), accumulating the mean
     fused embedding per class. Raises `OODUncertaintyError` if any class
     has zero training samples (a prototype is then genuinely undefined -
     never fabricated as a zero vector).
  2. One forward pass over `val_original.csv` computes, per sample: EDL
     uncertainty, **cosine distance** (`nearest_prototype_cosine_distance`
     - `1 - cosine_similarity`, L2-normalizing both the embedding and every
     prototype internally, so the metric is scale-invariant) to the
     nearest prototype, and whether the checkpoint's own prediction was
     correct.
  3. `_fit_minmax`/`_apply_minmax`: min-max normalization fit on
     val_original's distribution of each raw signal (not train - train
     samples define the prototypes, so their own distances are biased
     low). `clip((x - min) / (max - min), 0, None)` - floored at 0, but
     deliberately NOT capped above 1, so a severely out-of-distribution
     test-time sample can push a normalized score arbitrarily high. A
     degenerate (zero-span) fitted range normalizes every value to 0.0
     rather than dividing by zero.
  4. `select_combine_weight`: a fixed grid search
     (`DEFAULT_WEIGHT_GRID = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]`,
     mirrored in `configs/evaluation.yaml: ood_uncertainty.weight_grid`)
     over `combined = normalized_edl + weight * normalized_ood`, choosing
     whichever weight maximizes error-detection AUROC (is the checkpoint's
     own prediction wrong?) computed entirely on val_original's own
     predictions. Ties (including "AUROC undefined for every candidate")
     are broken toward the smallest weight, since the grid is iterated in
     ascending order and only a strict improvement replaces the current
     best - deterministic and reproducible. The full per-candidate search
     is always recorded in `metadata.json`, not just the winner.
  Calibration is recomputed fresh on every invocation (cheap, deterministic,
  no randomness) rather than cached to a separate artifact file, so every
  run's `metadata.json` fully documents exactly how its numbers were
  derived from that run's checkpoint + manifests.
- **Frozen-method evaluation** (read-only, one-shot): the calibrated
  method (prototypes + normalization + weight) is applied to clean
  `test_original.csv` (the same `expected_split="test",
  require_all_original=True` safeguard as `final_test`/`robustness`) and
  to every robustness condition, computing per condition: accuracy; mean
  EDL/OOD/combined score (both normalized and raw); error-detection
  AUROC/AUPRC for all three scores (`None` when undefined, e.g. zero
  errors in that condition - same convention as `metrics.py`); and
  (`_compute_severity_correlations`) a Spearman rank correlation (computed
  via pandas' tie-aware `.rank()` + `np.corrcoef`, not scipy, since pandas
  is already a declared project dependency) between a monotonic
  "corruption strength" convention and each score's mean, per degradation
  family - `_corruption_strength`: `|factor - 1.0|` for brightness/contrast
  (severities are symmetric around "no change"), the severity itself for
  gaussian_noise/gaussian_blur (already monotonic), and `image_size -
  target_size` for reduced_resolution (smaller target = more corruption).
  This convention is used ONLY for the correlation analysis - it plays no
  role in the actual degradation, normalization, or combination logic.
  Selective risk/coverage (`_compute_risk_coverage`, at
  `configs/evaluation.yaml: selective_prediction.coverage_levels`) is
  computed on the clean condition only, for all three scores.
- **Outputs are entirely separate from `final_test`'s and `robustness`'s**:
  every invocation generates a fresh `run_id`
  (`ood_uncertainty_aa_evidentnet_seed<seed>_<timestamp>_<6-hex>`, the same
  `generate_run_id` mechanism every other run uses) and writes only to
  `results/ood_uncertainty/<run_id>/` - never to `results/raw_predictions/`,
  any `results/tables/<eval_run_id>/`, or `results/robustness/<robustness_run_id>/`:
  `metrics.csv` (one row per condition), `selective_risk_coverage.csv`,
  `severity_vs_score.png` (one panel per degradation family, EDL vs. OOD
  vs. combined vs. severity, matplotlib `Agg` backend - headless-safe,
  never opens a display), and `metadata.json` (prototype class counts,
  both normalization ranges + which manifest each was fit on, the full
  weight-grid search + chosen weight, severity correlations, checkpoint/
  train/val/test manifest hashes, config paths + combined config hash, git
  commit, timestamp, every output path). Two invocations never collide.
  `experiments/registry.csv` is neither read nor written by this module.
- **Rejects any model other than `aa_evidentnet`** with a clear
  `OODUncertaintyError` (baselines have neither a fused embedding nor an
  evidential head - there is nothing for this module to combine).
- **matplotlib==3.10.9 is now an actual, installed, pinned dependency**
  (`requirements.txt`/`environment.yml`) - it was previously listed
  unpinned as "anticipated for later tasks" but never installed in the
  reference environment; this is the first feature that actually needs it
  (the severity-vs-score figure).
- **45 new tests** (`tests/test_ood_uncertainty.py`), plus 3 dedicated CLI
  tests in `tests/test_cli.py` (parser requires `--model`/`--checkpoint`,
  `--config` defaults to `configs/models.yaml`, a non-`aa_evidentnet`
  model is rejected with a nonzero exit code, and a full end-to-end run),
  covering: cosine-distance correctness (identical/orthogonal/opposite/
  scale-invariance/nearest-of-several), normalization fit/apply including
  the degenerate zero-span case, error-detection AUROC/AUPRC undefined
  cases, weight-grid selection (best-AUROC, tie-break-to-smallest,
  fallback-to-smallest-when-all-undefined), the fixed default weight grid,
  the corruption-strength convention per degradation family, Spearman
  correlation edge cases, `compute_class_prototypes` correctness (checked
  against an independently hand-computed per-class mean) and its
  missing-class error, calibration proven to succeed with NO
  `test_original.csv` present at all (direct proof of zero test
  dependency), calibration's missing-train/missing-val errors, full
  end-to-end schema/output-location checks, non-`aa_evidentnet` rejection,
  missing/malformed test-manifest rejection (non-test split, augmented
  sample), incompatible-checkpoint rejection, `model.eval()` called, no
  optimizer ever constructed, no backward pass ever called, checkpoint/raw
  image files/all three manifests verified byte-unchanged after a full
  run, and non-colliding output directories. All tests use tiny synthetic
  fixtures under `tmp_path` - **none run inference against the real
  train/val/test manifests or `data/raw/`.**
- **No real OOD/EDL inference was performed on the real train/val/test
  manifests** as part of implementing or testing this feature - confirmed
  by `test_original.csv`'s unchanged mtime and `data/raw/`'s unchanged
  file count (5,335) throughout this work. The real, fully-trained
  AA-EvidentNet checkpoint behind the 87.90% clean-test / 12.79%
  noisy-robustness numbers lives on Colab, not in this local repo -
  running this feature against it (to get real calibration numbers and a
  real severity-vs-score figure) is a separate, later step for whoever
  holds that checkpoint.

## Learned class-level ambiguity reproducibility (`feature/learned-ambiguity`, Phase 1)

**A new experimental research direction**, developed on its own branch (`feature/learned-ambiguity`), never merged into or affecting `master`. Not a novelty claim — only a statement that this mechanism did not previously exist in this repository under any name (a check was made: the only overlapping prior logic was `ood_uncertainty.py`'s prototype computation, relocated and shared rather than duplicated - see below).

**Four deliberately separate concepts, never combined into one score**: class-level ambiguity (a static K×K matrix - "which classes generally overlap"), sample-level ambiguity (a per-image scalar - "which class does *this* image resemble", analysis-only in this phase), EDL uncertainty (`src/losses/evidential.py`, unmodified), and OOD (`src/evaluation/ood_uncertainty.py`, unmodified). The class matrix is deliberately built from embedding geometry alone, **never from a confusion matrix or model predictions** - a confusion-matrix-based construction would conflate "these classes look alike" with "the model gets these wrong," collapsing the ambiguity/uncertainty distinction this design exists to preserve.

### Shared prototype utilities (behavior-preserving relocation)

`src/models/prototypes.py` is a new, dependency-free module holding `compute_class_prototypes` and `nearest_prototype_cosine_distance` - relocated out of `src/evaluation/ood_uncertainty.py`, which previously defined them inline. `ood_uncertainty.py` now contains thin wrapper functions of the same names that delegate to the shared module and re-raise its `PrototypeComputationError` as `OODUncertaintyError` (identical message), so its public API and error contract are byte-for-byte unchanged. **Verified**: all 45 pre-existing `tests/test_ood_uncertainty.py` tests pass unmodified after the refactor - no OOD calculation, output schema, or error message changed. `tests/test_prototypes.py` (10 tests) covers the relocated module directly, including a determinism check (two calls with identical inputs produce identical prototypes) not previously tested in isolation.

### Class-level ambiguity: exact construction and reference representation

```
P_k    = mean fused embedding of class k, over train_original.csv ONLY
S(a,b) = cosine_similarity(P_a, P_b)
A[a,b] = max(0, S(a,b))   for a != b   (rectified, [0,1], symmetric)
A[a,a] = 0                              (diagonal, explicit, never queried)
```

`src/losses/ambiguity.py: compute_class_ambiguity_matrix` implements exactly this - pure math, no I/O, matching `cs_supcon.py`/`evidential.py`'s existing purity convention. The matrix is converted to a non-trainable `torch.Tensor` (`class_ambiguity_matrix_to_buffer`, `requires_grad=False`) and installed into `CSSupConLoss` via `register_buffer` (`set_learned_ambiguity_matrix`) - confirmed by a direct gradient check (`tests/test_cs_supcon.py`-style: backward through a loss using the matrix leaves `matrix.grad is None`).

**Reference representation**: `src/training/ambiguity_setup.py: build_learned_class_ambiguity` loads the **existing, already-fully-trained AA-EvidentNet checkpoint** (Phase 1's chosen frozen development representation - see the design discussion in this branch's conversation history for why this was preferred over training a dedicated CS-SupCon-disabled checkpoint: it requires no new GPU compute, and its weights were already fit without any test-set access, so reusing them introduces no test leakage) into its **own, throwaway `create_model(...)` instance** - never the model the caller is about to train. Every parameter's `requires_grad` is explicitly forced to `False`, the model is placed in `eval()`, and the only forward passes performed are under `torch.inference_mode()` over `train_original.csv` (never `train_balanced.csv`, `val_original.csv`, or `test_original.csv`). Verified via `tests/test_ambiguity_setup.py`: `model.eval()` is called (spy), no optimizer is ever constructed (spy on `AdamW.__init__`), no backward pass ever occurs (spy on `Tensor.backward`), every reference-model parameter has `requires_grad=False` after construction (direct check), and the reference checkpoint's file hash is unchanged before/after (it is never written to).

**Methodological caveat (stated in the module docstring, `configs/losses.yaml`, and here - not hidden)**: this reference checkpoint's embedding space was itself shaped by the *existing* fixed-hard-pair CS-SupCon objective. The learned matrix therefore reflects class geometry *after* that correction has already partially acted on it - it is **not** claimed to be a from-scratch, assumption-free measurement of natural class confusability, and is **not** claimed to be independent of the previous ambiguity mechanism.

### Sample-level ambiguity: exact equation and behavior (analysis-only)

```
sim_i,k      = cosine_similarity(z_i, P_k)
raw_margin_i = sim_i,(1) - sim_i,(2)                (top-2 raw cosine-similarity gap, [0,2])
ambiguity_i  = 1 - clip((raw_margin_i - margin_min)/(margin_max - margin_min), 0, 1)
```

`margin_min`/`margin_max` (`fit_margin_normalization`) are fit once from `train_original.csv`'s own raw-margin distribution - the same forward pass used for the prototypes, never validation or test. `competing_class_i = argmax_{k != top1} sim_i,k`, computed from raw (pre-softmax) similarities, so its identity is provably temperature-independent (verified: `tests/test_ambiguity.py::test_competing_class_identity_is_temperature_independent` computes it at two wildly different `entropy_temperature` values and confirms an identical result).

**A numerical check was done before implementing this formula** (per explicit review): does a pure top-2 softmax margin genuinely approach 0 for an easy sample despite K=10 classes? Verified yes - a dominant top-1 class yields a near-1 margin regardless of K, since only the top-2 values matter. The formula was still revised to drop the softmax/temperature dependency entirely for the *primary* scalar (using the raw cosine margin, min-max normalized from train-original statistics instead - removing an uncalibrated hyperparameter), and a secondary diagnostic was added specifically because the margin *does* have a real, verified blind spot: it cannot distinguish a sharp two-way tie from a diffuse four-way confusion (both can produce a similarly small margin). `tests/test_ambiguity.py::test_entropy_distinguishes_two_way_tie_from_diffuse_confusion` encodes this numerically: a two-way near-tie among 10 classes produces a lower normalized entropy than a four-way near-tie, even though both produce a comparably small raw margin.

`ambiguity_i`/`competing_class_i` depend only on each sample's own embedding - never on a true or predicted label - so the same computation applies at development-time analysis and at any future inference use without a label-dependent branch.

### CS-SupCon: `ambiguity_source` modes

`configs/losses.yaml: cs_supcon.ambiguity_source` is restricted to exactly `{"fixed_pairs", "learned_class"}` in this phase - `"learned_class_sample"` is recognized (not treated as a config typo) but explicitly raises a clear "not implemented in this phase" error, per the explicit instruction that sample-level ambiguity must be validated as meaningful before it is allowed to influence optimization.

- **`fixed_pairs`** (default): unchanged. **Backward compatibility was verified two ways**: (1) all 46 pre-existing `tests/test_cs_supcon.py` tests and all 27 pre-existing `tests/test_combined_loss.py` tests pass unmodified; (2) a direct numerical check confirmed `ambiguity_source="learned_class"` with `ambiguity_scale * A[a,b] = ambiguity_weight - 1` for a given pair produces a bit-identical loss value to the equivalent `fixed_pairs` configuration on the same synthetic batch - i.e. `learned_class` is a strict generalization of `fixed_pairs`, not an independently-reimplemented formula.
- **`learned_class`**: `w(i,a) = 1 + ambiguity_scale * A[y_i, y_a]` for negatives (`y_i != y_a`); positives and self are handled identically to the `fixed_pairs` branch (unchanged code, not re-derived). Calling `forward()` in this mode before `set_learned_ambiguity_matrix(...)` raises `CSSupConConfigError` rather than silently defaulting to uniform weighting.

### Training integration (`src/training/run_aa_evidentnet.py`) - no `Trainer` changes

The entire mechanism is orchestrated in `run_aa_evidentnet_training`, entirely **before** `optimizer`/`scheduler`/`Trainer` are constructed - `src/training/trainer.py` was not modified at all (no new hook, no epoch-dependent recomputation, no periodic update). `ambiguity_source="fixed_pairs"` performs zero setup and requires no reference checkpoint (verified: `tests/test_run_aa_evidentnet_ambiguity.py::test_fixed_pairs_requires_no_reference_checkpoint_and_performs_no_setup` monkeypatches `build_learned_class_ambiguity` and confirms it is never called). `ambiguity_source="learned_class"` builds the matrix once, installs it into `criterion.cs_supcon_loss`, and writes a `results/logs/<run_id>/ambiguity_metadata.json` reproducibility artifact (reference checkpoint path + SHA-256, train manifest path + SHA-256, per-class sample counts, margin normalization, the full matrix, and the methodological caveat above) before `trainer.fit()` is ever called - verified via a spy on `Trainer.fit()` recording that the matrix had already been built by the time training started. `smoke_test=True` combined with `ambiguity_source="learned_class"` raises a clear `RunAAEvidentNetError` (smoke-test's synthetic data provides no real `train_original.csv` or reference checkpoint) rather than silently falling back to `fixed_pairs` or fabricating synthetic prototypes.

### Validation-only development protocol (train_original.csv + val_original.csv ONLY)

`src/evaluation/ambiguity_validation.py: run_ambiguity_validation` answers "is sample ambiguity meaningful?" using the SAME frozen reference checkpoint and the artifact already built by `ambiguity_setup`, forward-passing `val_original.csv` exactly once (read-only, `eval()`, `torch.inference_mode()`, no optimizer, no backward pass - verified by the same spy-based tests used elsewhere in this project). Seven analyses, all validation-only: error-detection AUROC/AUPRC of ambiguity vs. misclassification; competing-class hit rate among errors; ambiguity mean/median for correct vs. incorrect predictions; ambiguity mean/median for the existing fixed hard-pair classes vs. others; Spearman correlation between ambiguity and EDL uncertainty (reusing `ood_uncertainty.py`'s own `_spearman` via same-package cross-import, the established convention); a 2×2 ambiguity/EDL-uncertainty quadrant error-rate breakdown; and a ranking comparison (never a construction input) between the learned matrix and an ordinary validation confusion matrix. Outputs: `results/ambiguity/<run_id>/class_ambiguity_matrix.csv`, `validation_metrics.json`, `metadata.json`.

**The function signature has no test-manifest parameter at all** (`tests/test_ambiguity_validation.py::test_run_ambiguity_validation_has_no_test_manifest_parameter` checks this structurally, not by a fragile text search over the module source - the module's own docstring correctly *mentions* `test_original.csv` in prose to explain that it is never used, which a naive substring check would have flagged as a false positive). A separate test confirms the full analysis succeeds with no `test_original.csv` present in the fixture at all.

### Tests added

`tests/test_prototypes.py` (10), `tests/test_ambiguity.py` (33), `tests/test_ambiguity_setup.py` (14), `tests/test_ambiguity_validation.py` (18), `tests/test_run_aa_evidentnet_ambiguity.py` (6) - 81 new tests, all using tiny synthetic fixtures under `tmp_path`. **No test in this entire feature ever reads, requires, or references `data/manifests/test_original.csv`.**

### Confirmations

- `data/manifests/test_original.csv` was never opened during this work - confirmed by its unchanged mtime and unchanged file count (5,335) in `data/raw/`.
- No `final_test`, `robustness`, or `ood_uncertainty` evaluation was run as part of this work.
- No training run (real or smoke-test-scale, beyond what the tests themselves execute against synthetic data) was performed against the real dataset.
- No performance or novelty claims are made anywhere in this feature's code, configuration, or documentation.

## Phase 2: neighborhood-based learned ambiguity reproducibility (research only, `feature/learned-ambiguity`)

**Motivation, from Phase 1's own validation run**: AUROC=0.5703292638, AUPRC=0.2040891098, matrix-vs-confusion Spearman=0.3068058483, competing-class hit rate=0.0117647059, and — the specific weakness this phase investigates — the single largest observed validation confusion (Healthy ↔ Glaucoma, 35 confusions) was not strongly identified by class-prototype cosine similarity. This phase does not assume neighborhood-based ambiguity is better; it exists to measure whether it is, using the numbers actually produced by running it (see "Not yet run against real data" below).

**Phase 1 was not modified in any way** to build this phase — `src/losses/ambiguity.py`, `src/training/ambiguity_setup.py`, `src/evaluation/ambiguity_validation.py` are untouched, and all of Phase 1's own tests (`tests/test_ambiguity.py` 33, `tests/test_ambiguity_setup.py` 14, `tests/test_ambiguity_validation.py` 18, `tests/test_prototypes.py`'s original 10) were re-run unmodified and still pass, confirming zero regression.

### Shared extraction utility (additive only)

`src/models/prototypes.py` gained one new function, `extract_embeddings` (returning per-sample embeddings, labels, argmax predictions, and EDL uncertainty from one forward pass) - purely additive; `compute_class_prototypes` and `nearest_prototype_cosine_distance` (Phase 1's own functions) were not changed at all. This is the single shared extraction loop Phase 2 uses instead of re-inlining a near-duplicate copy of Phase 1's existing forward-pass loops (which themselves remain as they were, untouched, in `ambiguity_setup.py`/`ambiguity_validation.py`).

### Class-level neighborhood matrix: exact construction and complexity

`src/losses/neighborhood_ambiguity.py` (pure math, no I/O, matching every other loss module's purity convention):

- `find_cross_class_neighbors(embeddings, labels, k)`: for every sample, the k highest-cosine-similarity samples with a *different* class label (self and same-class samples excluded by construction, not by post-hoc filtering). Computed via one `[N, D] @ [D, N]` matrix multiplication (`N=3075` train samples in the real dataset → an `[N, N]` similarity matrix of a few tens of MB) - never an `[N, N, D]` tensor. Raises `NeighborhoodAmbiguityError` if fewer than k cross-class candidates exist for any sample, rather than silently returning a truncated/inconsistent neighbor list.
- `compute_neighborhood_class_matrix`: `score(a->b) = mean over class-a samples of (sum over their top-k cross-class neighbors in class b of max(0,cosine)) / k`, symmetrized, zero diagonal, then min-max rescaled using **only the matrix's own off-diagonal entries** (verified in tests to use the full [0,1] range, and to return an all-zero matrix rather than dividing by zero in the degenerate case where every pair is equally (dis)similar).
- `compute_sample_neighborhood_ambiguity`: per train sample, nearest competing class (rank-1 neighbor's class), mean top-k similarity, per-class neighbor fraction, and strongest competing class (mode of the k neighbors' classes, which can differ from the single nearest neighbor's class - both are reported, verified independently in tests).

### Validation sample ambiguity: exact equation and why no additional normalization is fit

`compute_validation_neighborhood_ambiguity`: for each validation embedding, the k overall nearest **train** embeddings (same-class neighbors allowed, unlike the class-matrix construction above - a validation sample deep in its own class's neighborhood must be able to read as low-ambiguity, which requires letting same-class train neighbors count towards `p_c`). `margin = p_(1)-p_(2)`; `ambiguity_margin = 1-margin` (PRIMARY, for direct consistency with Phase 1's own top-1-vs-top-2 margin choice); `ambiguity_entropy = H(p)/log(K)` (secondary). Both are proportions/probabilities, hence already bounded in [0,1] by construction - unlike Phase 1's raw cosine margin (which genuinely needed empirical min-max fitting since cosine-similarity gaps have no natural [0,1] bound), no additional train-fitted normalization constant is needed or computed here, and this is stated explicitly rather than silently omitted. `competing_class` is rank-2 of `p_c` (ties broken toward the lower class index via a stable sort) - computed purely from embedding geometry, never from a true or predicted label, exactly like Phase 1's own competing-class rule.

### Orchestration and leakage safeguards (identical guarantees to Phase 1)

`src/training/neighborhood_ambiguity_setup.py: build_neighborhood_class_ambiguity` loads the reference checkpoint read-only into its own throwaway model instance (`eval()`, every parameter's `requires_grad` forced `False`, no optimizer ever constructed, no backward pass ever occurs - all verified via the same spy-based tests used throughout this project), reads `train_original.csv` only, and returns an artifact carrying the full train embedding/label arrays forward (not just a per-class summary, unlike Phase 1's prototypes - the neighborhood method for validation genuinely needs the individual training embeddings to search against). `src/evaluation/neighborhood_ambiguity_validation.py: run_neighborhood_ambiguity_validation` reads `val_original.csv` only and has no test-manifest parameter at all (checked structurally in `tests/test_neighborhood_ambiguity_validation.py`, not by a fragile text search). Neither module imports anything capable of loading `test_original.csv`.

### Comparison artifact

`build_prototype_vs_neighborhood_comparison` reads Phase 1's **own already-saved** `validation_metrics.json`/`class_ambiguity_matrix.csv` (from a prior, separate `run_ambiguity_validation` call - never recomputed) and Phase 2's in-memory summary/matrix, and reports both methods' key metrics plus each of the three existing fixed hard pairs' rank and value in both matrices side by side. It makes no claim about which method is better - `tests/test_neighborhood_ambiguity_validation.py` verifies the output explicitly carries a disclaiming `"note"` field to this effect, and handles a pair whose class names don't exist in a given fixture gracefully (`None`/`None` rather than a crash).

### Tests added

`tests/test_neighborhood_ambiguity.py` (32 - neighbor exclusion of self/same-class, correctness against hand-computed toy embeddings, determinism, k-too-large rejection, class-matrix symmetry/zero-diagonal/bounds/degenerate-case handling, validation margin/entropy bounds, competing-class ranking, proof that validation samples never influence each other), `tests/test_neighborhood_ambiguity_setup.py` (14 - end-to-end correctness against independently-recomputed values, frozen-reference-model guarantees, train-only manifest dependency, checkpoint-never-modified, k-too-large-for-dataset), `tests/test_neighborhood_ambiguity_validation.py` (18 - end-to-end schema, bounded metrics, zero test-manifest dependency, comparison-artifact correctness) - 64 new tests, all synthetic-fixture-only. Phase 1's existing 81 ambiguity tests were re-run unmodified as the regression check for "Phase 1 behavior unchanged."

### Not yet run against real data

This entire phase was implemented and tested against tiny synthetic fixtures only, exactly like Phase 1 before it. The real reference checkpoint (`checkpoints/20260831_064112_aa_evidentnet_seed42_48c214/best.pt`, SHA-256 `800d6184416d77a2a2d1447680ca7869bb973d62fc361f15a6ddfa0e95be2c2a`) is a Colab-side artifact and is not present in this local repository - running `build_neighborhood_class_ambiguity` / `run_neighborhood_ambiguity_validation` against the real `train_original.csv` (3075 samples) / `val_original.csv` (880 samples) to get real numbers, and comparing them against Phase 1's real validation results, is a separate, later step for whoever holds that checkpoint (see README.md/this feature's PR description for the exact function calls).

### Confirmations

- `data/manifests/test_original.csv` was never opened during this work.
- No training, fine-tuning, `final_test`, `robustness`, or `ood_uncertainty` evaluation was run.
- Phase 1's files and test results are unchanged.
- No performance or novelty claim is made about the neighborhood method - it has not yet been run against real data.

### Real validation results (run separately, on Colab, against the real checkpoint/manifests)

Reference checkpoint: `checkpoints/20260831_064112_aa_evidentnet_seed42_48c214/best.pt` (SHA-256 `800d6184416d77a2a2d1447680ca7869bb973d62fc361f15a6ddfa0e95be2c2a`). Frozen splits: `train_original.csv` (3075 samples), `val_original.csv` (880 samples) - `test_original.csv` was not used.

|  | Phase 1 (prototype) | Phase 2 (neighborhood, k=10) |
|---|---|---|
| Error-detection AUROC | 0.5703292638 | 0.5199408065 |
| Error-detection AUPRC | 0.2040891098 | 0.1056149733 |
| Class-matrix vs. confusion Spearman | 0.3068058483 | 0.4061831051 |
| Competing-class hit rate among errors | 0.0117647059 | 0.0 |

Phase 2 improved the class-level structure correlation (0.3068 -> 0.4062) but its per-sample ambiguity became worse (lower AUROC/AUPRC) and extremely sparse (median 0.0 for both correct and incorrect predictions). Neither approach strongly identified the largest real validation confusion (Healthy <-> Glaucoma, 35 errors) - this is the specific, disclosed weakness Phase 3 (below) investigates. These are real, already-observed results, not projections - reported here exactly as measured, with no claim about which method is preferable.

## Phase 3: continuous class-affinity ambiguity reproducibility (research only, `feature/learned-ambiguity`)

**Exploratory, uses the same frozen previously-trained checkpoint as Phase 1/2. No new model training occurs.** Neither Phase 1 nor Phase 2 was modified - all of their existing tests were re-run unmodified and still pass.

### Method: exact equations

`src/losses/class_affinity_ambiguity.py` (pure math, no I/O):

- `compute_class_affinities(query, reference, reference_labels, num_classes, m=5, exclude_self=False)`: `a_i,c = mean(top-min(m, available) cosine similarities to reference embeddings of class c)`. `exclude_self=True` (query IS reference, row-aligned) removes each sample's own diagonal entry before ranking - implemented via `np.fill_diagonal(similarity, -inf)` plus a per-row effective-count adjustment so the excluded position can never be accidentally included when a class has few samples (verified by `test_affinity_exclude_self_removes_own_entry` and the degenerate-count path).
- Primary score: `margin_i = a1 - a2` (top1 minus top2 class affinity); `ambiguity_i = 1 - clip(margin_i / margin_scale, 0, 1)`. `margin_scale` = 95th percentile of TRAIN samples' own self-excluded top1-top2 margins (`fit_margin_scale`) - raises `ClassAffinityAmbiguityError` if numerically <= 0.
- Secondary diagnostic: normalized entropy of `softmax(affinities / temperature)`, `temperature=0.1` fixed.
- Label-aware boundary score (ANALYSIS ONLY - documented in the module docstring, the metadata, and here as never an inference-time candidate): `boundary_gap_i = a_i,y - max_{c!=y} a_i,c`; `label_aware_ambiguity_i = 1 - clip(boundary_gap_i / boundary_gap_scale, 0, 1)` - deliberately the identical transform shape as the primary score (no new trainable function), `boundary_gap_scale` fit the same way (95th percentile of TRAIN samples' own boundary gaps).
- Class matrix (`compute_class_affinity_matrix`): `directed(a->b) = mean over train samples in class a of their self-excluded affinity to class b`; symmetrized, zero diagonal, min-max rescaled using only its own off-diagonal entries - the same "directed score -> symmetrize -> rescale" pattern as Phase 1/Phase 2's own matrices, applied to a genuinely different underlying quantity (continuous class affinity, not a single centroid or a discrete neighbor vote).

### A real numerical fragility discovered during testing (not a code defect)

Early test runs of `tests/test_class_affinity_ambiguity_setup.py`/`tests/test_class_affinity_ambiguity_validation.py` intermittently failed with `boundary_gap_scale is numerically zero or negative`. Investigation showed this was NOT a bug in the fail-loud check (which is working exactly as specified) but a genuine property of the test fixtures: unlike `margin_scale` (always >= 0 by construction, since top1 >= top2), `boundary_gap` has no such sign guarantee, and an **untrained** reference model's embedding space (used only in tests - the real reference checkpoint is fully trained) has very little real class structure to rely on. Combined with `tests/conftest.py: make_image()`'s default per-FILE (not per-class) hash-derived color, the resulting synthetic images gave an untrained model too weak and inconsistent a signal, so the 95th percentile of boundary gaps occasionally landed just below zero purely from incidental per-test image/seed variation - not from anything the code under test did wrong. Fixed two ways, both applied in `tests/test_class_affinity_ambiguity_setup.py` and `tests/test_class_affinity_ambiguity_validation.py`: (1) a fixed `torch.manual_seed(42)` before constructing the untrained reference model, and (2) assigning each class a fixed, distinct base color (`_CLASS_COLOR_PALETTE`, with small per-sample jitter) instead of relying on incidental per-path hash noise. Neither fix touches the module under test - both are test-fixture-only changes, and the underlying fail-loud check in `src/losses/class_affinity_ambiguity.py` is unchanged.

### Orchestration and leakage safeguards (identical guarantees to Phase 1/Phase 2)

`src/training/class_affinity_ambiguity_setup.py: build_class_affinity_ambiguity` loads the reference checkpoint read-only into its own throwaway model instance (`eval()`, every parameter's `requires_grad` forced `False`, no optimizer, no backward pass - all spy-verified), reads `train_original.csv` only, and keeps the full train embedding/label arrays (validation-time affinity computation needs to search the individual training embeddings, not a summary). `src/evaluation/class_affinity_ambiguity_validation.py: run_class_affinity_ambiguity_validation` reads `val_original.csv` only and has no test-manifest parameter at all (checked structurally). Neither module's public function signature contains anything named `test*`.

### Three-way comparison

`build_three_phase_comparison` reads Phase 1's and Phase 2's own already-saved `*_metrics.json`/`*_matrix.csv` artifacts (never recomputed) plus Phase 3's own in-memory summary, and reports all three methods' key metrics and each of the three existing fixed hard pairs' rank/value in all three matrices side by side. The code contains no comparison operator or conditional that would let it assert one phase is "better" - it only assembles and writes the numbers, verified by a test that checks the output's disclaiming `"note"` field is present.

### Tests added

`tests/test_class_affinity_ambiguity.py` (32 - affinity correctness including self-exclusion, top-affinity ranking with tie-breaking, margin-scale/boundary-gap-scale fitting and their zero/negative rejection, entropy bounds, class-matrix symmetry/zero-diagonal/bounds/degenerate handling), `tests/test_class_affinity_ambiguity_setup.py` (14 - end-to-end correctness against independently-recomputed scales, frozen-reference-model guarantees, train-only manifest dependency, checkpoint-never-modified, own-sample exclusion verified by comparing against a with-exclusion-disabled recomputation), `tests/test_class_affinity_ambiguity_validation.py` (15 - end-to-end schema, bounded metrics, hard-pair rank reporting for all three named pairs, zero test-manifest dependency, three-way comparison artifact correctness) - 61 new tests, all synthetic-fixture-only. Phase 1's and Phase 2's existing tests were re-run unmodified as the regression check for "Phase 1/Phase 2 behavior unchanged."

### Not yet run against real data (superseded for Phase 3 - see below)

Exactly like Phase 1 and Phase 2 before it, this phase was implemented and tested against tiny synthetic fixtures only. Running `build_class_affinity_ambiguity` / `run_class_affinity_ambiguity_validation` / `build_three_phase_comparison` against the real checkpoint and manifests to get real numbers - and to see whether Healthy <-> Glaucoma is captured better than in Phase 1/Phase 2 - is a separate, later step (see README.md for the exact function calls).

### Real Phase 3 validation run (superseding the above)

Phase 3 was subsequently run for real, against the frozen checkpoint and `val_original.csv` (880 samples, 85 errors; `m=5`, `temperature=0.1`, `margin_scale=1.009321797913053`; run id `20260902_115241_class_affinity_ambiguity_validation_seed0_59314c`). Real numbers: error-detection AUROC 0.709478357380688, AUPRC 0.2359527095006137; mean ambiguity correct/incorrect 0.15379148446637772 / 0.23670059569672192; Spearman(ambiguity, EDL uncertainty) 0.753761860508129; matrix-vs-confusion Spearman 0.3515634766234436; Healthy<->Glaucoma rank 6 (value 0.8644270977122319, the largest real validation confusion). Full numbers, including the Phase 1/Phase 2 comparison, are in README.md's "Real Phase 1/2/3 validation numbers" table. This run's numbers are unmodified by, and predate, the Phase 3b work below - Phase 3b reuses this same run's per-sample values rather than recomputing them.

## Phase 3b: ambiguity/EDL-uncertainty complementarity reproducibility (research only, `feature/learned-ambiguity`)

**Analysis-only. No training, no new model inference, no changes to any Phase 1/2/3 equation or score.** Motivated directly by the real Phase 3 numbers above: a Spearman correlation of 0.75 between ambiguity and EDL uncertainty, and a "high ambiguity" quadrant that overlaps with "high EDL uncertainty" for 412 of 880 samples, both suggest the two signals may be substantially redundant - Phase 3b checks this directly rather than assuming it either way.

### Additive change 1: `src/models/prototypes.py: extract_embeddings`

Added `include_identifiers: bool = False` (default off) and two new `Optional[np.ndarray]` fields on `ExtractedEmbeddings` (`sample_ids`, `image_paths`), populated from `batch["original_id"]`/`batch["image_path"]` (both already present on every `RetinalDataset` sample) only when `include_identifiers=True`. Every existing caller (Phase 1's/Phase 2's setup and validation modules, Phase 3's own setup module) omits the new argument, so their behavior - verified by re-running their existing test suites unchanged - is byte-for-byte identical to before this change.

### Additive change 2: `src/evaluation/class_affinity_ambiguity_validation.py: run_class_affinity_ambiguity_validation`

Added `save_per_sample_csv: bool = False` (default off) and an `Optional[str]` field `per_sample_path` on `ClassAffinityAmbiguityValidationSummary`. When `True`, writes `class_affinity_per_sample.csv` (`sample_id, image_path, true_class_name, predicted_class_name, correct, ambiguity, edl_uncertainty`) into the run's own output directory, using values already computed earlier in the same function - no second forward pass, no recomputation. Verified not to change any existing metric: `test_per_sample_csv_does_not_change_existing_metrics` runs the function twice (with and without the flag) against the identical inputs and asserts every existing summary field is unchanged; all 14 pre-existing Phase 3 validation tests were re-run unmodified and still pass.

### `src/evaluation/ambiguity_complementarity.py`: the complementarity analysis itself

Reads ONLY the per-sample CSV above - no model, checkpoint, or manifest of any kind, and no `import torch`. `AMBIGUITY_WEIGHT = 0.5` and `EDL_WEIGHT = 0.5` are fixed module-level constants (not searched, not fit): `combined_score = 0.5 * ambiguity + 0.5 * edl_uncertainty`. Neither input is renormalized before combining - `ambiguity` is already in `[0,1]` via Phase 3's own train-derived `margin_scale`, and EDL `uncertainty = K / sum(dirichlet_alpha)` is already bounded in `(0,1]` by construction (`src/losses/evidential.py`), so there is nothing left to fit "using train data only": both quantities were already produced by a train-data-only process before this module ever sees them.

Computes, reusing `_error_detection_auroc`/`_error_detection_auprc`/`_spearman` from `src/evaluation/ood_uncertainty.py` (same cross-module private-helper-reuse convention as every other evaluation module in this project) rather than reimplementing them:

- EDL-alone / ambiguity-alone / combined error-detection AUROC and AUPRC, and the four exact pairwise differences.
- A median-split (computed on this same validation set, used only for these discrete breakdowns, never to fit anything) error-overlap table among the actual errors: both / ambiguity-only / EDL-only / neither.
- The identical breakdown among correct predictions (false alarms).
- The high-ambiguity/low-EDL and low-ambiguity/high-EDL discordant groups - count, error count, error rate - **recomputed from the loaded per-sample array on every call**; `test_discordant_groups_recomputed_not_hardcoded` constructs a 4-row fixture with a hand-verifiable answer and checks the module reproduces it, and no test or code path anywhere hardcodes the real run's 28/28 figures.
- Spearman(ambiguity, EDL uncertainty) via the same reused helper Phase 3 itself uses.
- An optional paired bootstrap 95% CI (`_bootstrap_auroc_diff_ci`): the SAME resampled indices are applied to all three scores within each iteration (so the distribution is of the matched difference, not of two independently resampled AUROCs); a resample is skipped and separately counted whenever `_error_detection_auroc` returns `None` for that resample (single-class-only, e.g. an all-correct or all-error draw); fixed `np.random.default_rng(seed)`, default 10,000 resamples, default seed 0, both overridable.

The output JSON explicitly states, as literal fields (not just something a reader has to infer): the fixed combination equation, that the weights were predetermined and never searched, that no additional normalization was fitted, that no training/checkpoint/optimizer was touched, that `test_original.csv` was not used, and that all three scores were computed on the identical sample/error counts. A dedicated `NOTE` field states the interpretation policy: the module never asserts ambiguity is or isn't complementary - `test_result_never_claims_ambiguity_is_complementary` asserts no such claim string appears anywhere in the serialized output.

### Leakage-safeguard tests (`tests/test_ambiguity_complementarity.py`, 23 tests)

No test-manifest parameter (checked structurally via `inspect.signature`); no `import torch`, no `load_checkpoint`/`restore_training_state` call, no `torch.optim`, no `.backward(` anywhere in the module's source; no literal `"test_original.csv"` path construction anywhere in the module's source; combination weights fixed at `0.5`/`0.5` (asserted both as constants and via the computed formula); no weight grid or `argmax`-style search construct anywhere in the module; identical sample/error counts confirmed feeding all three scores; deterministic output (including the bootstrap CI) for a fixed input CSV and fixed seed, verified by running the full analysis twice and comparing every field.

### Tests added

`tests/test_ambiguity_complementarity.py` (23, new module, all synthetic per-sample CSVs), plus additive coverage in `tests/test_prototypes.py` (3 new tests for `include_identifiers`) and `tests/test_class_affinity_ambiguity_validation.py` (3 new tests for `save_per_sample_csv`) - all existing tests in both files continue to pass unmodified.

### Not yet run against real data

`src/evaluation/ambiguity_complementarity.py` has not yet been run against the real `class_affinity_per_sample.csv` (that file does not exist yet - Phase 3's real run predates this additive CSV-export feature). The exact Colab steps to produce it and run the real Phase 3b analysis are in README.md / the assistant's final report for this change.

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

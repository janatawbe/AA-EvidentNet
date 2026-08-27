# AA-EvidentNet

Research project studying **uncertainty-aware retinal disease classification**
on the DS2 fundus image dataset — comparing a conventional baseline (MaxViT)
against a proposed evidential-learning model (AA-EvidentNet) on accuracy,
calibration, selective prediction, robustness, and interpretability.

> **Status: foundation only.** No dataset pipeline, model, or training code
> has been implemented yet, and **no experimental results exist**. Any
> numbers, tables, or figures that are not produced by actually running this
> codebase must never be fabricated or assumed. This README will be updated
> as each stage lands.

## Research questions

1. Does an evidential-uncertainty head (AA-EvidentNet) improve **calibration**
   and **selective prediction** over a standard softmax baseline (MaxViT),
   without sacrificing raw accuracy, on a small, class-imbalanced,
   multi-disease fundus dataset?
2. Which disease pairs are most frequently confused ("hard pairs"), and does
   the proposed model's uncertainty estimate track those confusions?
3. How robust are both models to common input perturbations (noise,
   brightness/contrast shift, blur, compression)?
4. Do model decisions correspond to clinically plausible regions of the
   fundus image (via Grad-CAM), and does this differ between models?
5. Are any observed differences between models statistically reliable across
   multiple seeds, or within the noise floor of a small dataset?

These are the questions the pipeline is being built to answer — not claims
about what the answers are.

## Planned architecture

- **Baselines (implemented; training engine verified end to end, but no
  full training run completed)**: ResNet50, EfficientNetB0, and MaxViT
  (`maxvit_tiny_tf_224`), each with a standard softmax / cross-entropy
  head. See "Baseline models" and "Training engine" below.
- **Proposed (AA-EvidentNet, not yet implemented)**: same or comparable
  backbone, with an evidential (Dirichlet-based) uncertainty head trained
  with an evidential loss, evaluated with Monte Carlo sampling for
  uncertainty estimates.

Exact architecture details, hyperparameters, and loss formulations for the
proposed model are **provisional** (see `configs/`) and will be finalized
based on baseline experiments, not fixed in advance. **No model has
completed training and no performance/accuracy numbers exist anywhere in
this repository** — the CPU-only development environment made a full
baseline training run impractical (see "The real baseline run" below);
only smoke tests and a capped real-data sanity check have been run.

## Dataset assumptions

The pipeline assumes the **DS2** retinal disease dataset is present, unpacked,
under `data/raw/`, as one subfolder per class:

```text
data/raw/<Class Name>/<image>.jpg
```

Verified in this environment (re-checked 2026-08-26, read-only): 10 classes,
5,335 JPG images total, ~1.7 GB, with substantial class imbalance (from 17
images in the smallest class to 1,509 in the largest).

The on-disk directory names under `data/raw/` are **not** modified to match
the project's canonical class names (some carry extra annotations, e.g.
`[Color Fundus]`). `configs/dataset.yaml: class_directory_mapping` is the
single source of truth mapping each canonical class name to its exact,
unmodified `data/raw/` subdirectory name — see that file rather than
duplicating the mapping here. Canonical class names and raw image counts:

| Canonical class name              | Raw directory name (data/raw/)                    | Images |
|------------------------------------|-----------------------------------------------------|-------:|
| Central Serous Chorioretinopathy   | `Central Serous Chorioretinopathy [Color Fundus]`   |    101 |
| Diabetic Retinopathy                | `Diabetic Retinopathy`                              |  1,509 |
| Disc Edema                          | `Disc Edema`                                        |    127 |
| Glaucoma                             | `Glaucoma`                                          |  1,349 |
| Healthy                              | `Healthy`                                           |  1,024 |
| Macular Scar                         | `Macular Scar`                                      |    444 |
| Myopia                               | `Myopia`                                            |    500 |
| Pterygium                            | `Pterygium`                                         |     17 |
| Retinal Detachment                   | `Retinal Detachment`                                |    125 |
| Retinitis Pigmentosa                 | `Retinitis Pigmentosa`                              |    139 |

The full dataset-preparation pipeline is implemented end to end:
`python run_pipeline.py prepare_dataset` runs the raw-data audit, the
deterministic original-image 70/20/10 split, and the balanced (2,000/class)
training-set generation, in that order. Model training is not yet
implemented.

### Known data-quality issue: cross-class exact duplicates (requires human review)

Running the audit against the real dataset found **464 exact-duplicate
image groups spanning contradictory canonical classes** (942 of 5,335
files, ~17.7% of the dataset) — i.e., the identical photograph (byte-for-byte,
same SHA-256) appears filed under two or more different disease labels.
This was independently verified with a plain `hashlib.sha256` check outside
the audit code, so it is a real property of the dataset, not an audit bug.
The most affected label pairs are Glaucoma/Healthy (163 groups) and
Glaucoma/Myopia (143 groups). A further 6 exact-duplicate groups exist
within the same class (12 files). Zero corrupted images and zero
suspicious augmentation-family filenames were found; class directory
structure and counts exactly match `configs/dataset.yaml`.

**This is a serious label-reliability problem, not something to
auto-resolve — and automated relabeling is not scientifically justified.**
No pipeline code may infer the correct label from class frequency,
directory name, filename, majority vote, a model's prediction, or any
other heuristic. Only a human reviewing the actual images can decide.

**The dataset must NOT be described as "clean" until every one of these
464 groups has been resolved by a human.** `python run_pipeline.py audit`
fails (exit code 1) by default for exactly this reason (see
`configs/dataset.yaml: audit.policies.cross_class_duplicate`), and the
failure message explicitly says human review is required.

#### Human-review workflow

- `data/audit/cross_class_duplicate_review.csv` (regenerated by the audit,
  not committed to git) — one row per conflicting group, with a
  `resolution` column a human edits directly. Allowed values, enforced by
  `src/data/duplicate_review.py`:
  - `UNRESOLVED` (the default for every group until reviewed)
  - `KEEP_CLASS` (requires `resolved_class` to be set to exactly one of the
    10 canonical classes)
  - `EXCLUDE_GROUP` (requires `resolved_class` to be empty; excludes the
    group from training entirely rather than trusting any label)
- Re-running the audit **preserves already-entered resolutions** (matched
  by SHA-256) — it never resets human review work back to `UNRESOLVED`,
  and it never overwrites an existing review file that fails validation
  (protecting completed review work from being silently destroyed by a
  typo).
- `data/audit/same_class_duplicate_report.csv` lists the 6 same-class
  duplicate groups separately — these have no label conflict and require
  no resolution, but are not auto-deleted either.
- `data/audit/cross_class_duplicate_summary.csv` gives aggregate counts
  (groups by class pair, groups unresolved/excluded/resolved-per-class).
- To inspect a group visually before deciding: `python -m
  src.data.review_duplicates --group-id DUPGROUP_0001` (or `--all
  --unresolved-only`) renders a contact sheet of every member image, its
  label, filename, and path under `data/audit/duplicate_review/` — read-only
  against `data/raw/`, never written there.

**Exact-duplicate leakage must be prevented once splitting exists:** an
image involved in an exact-duplicate group (same-class or cross-class)
must never be split across train/val/test. `src/data/eligibility.py`
exposes `assert_split_is_valid()` for that future stage to call; it never
assigns a resolution or a split itself — it only refuses invalid input.

#### Pipeline stages: raw dataset -> audit -> eligibility review -> eligible dataset -> original split -> balanced training set

```text
raw dataset (data/raw/, immutable — never renamed, moved, or modified)
    |
    v
audit (src/data/audit_dataset.py: inventory, integrity, exact-duplicate detection)
    |
    v
eligibility review (src/data/duplicate_review.py: a human resolves cross-class
                     label conflicts in cross_class_duplicate_review.csv)
    |
    v
ELIGIBLE DATASET (src/data/eligibility.py: data/audit/dataset_eligibility.csv)
    |
    v
ORIGINAL 70/20/10 SPLIT (src/data/build_split.py: data/manifests/{train,val,test}_original.csv)
    |
    v
BALANCED TRAINING SET, train split only (src/data/generate_balanced_dataset.py:
    data/manifests/train_balanced.csv + data/processed/train/) -- val/test untouched
    |
    v
future model training (NOT YET IMPLEMENTED)
```

`data/raw/` is always the permanent source archive — nothing is ever
deleted, renamed, or relabeled there, regardless of eligibility or split
decisions. The **eligible dataset** (`data/audit/dataset_eligibility.csv`)
is the population the split is drawn from; excluding a file from it is a
**data-quality decision** (an unadjudicated or rejected label conflict),
never a class-balancing strategy — classes are not up- or down-weighted
here, only genuinely conflicting/duplicate content is removed from
consideration. The split itself (below) is likewise not a balancing step:
it is stratified proportionally by class, so a rare class stays rare in
train/val/test — balancing to 2,000 samples/class happens only later, via
training-only augmentation, not here.

#### Eligibility layer

Every raw image gets exactly one row in `data/audit/dataset_eligibility.csv`
(regenerated by the audit; not committed to git), derived entirely from the
duplicate groups and the *current* `cross_class_duplicate_review.csv`
resolution state — nothing is hard-coded:

- not part of any exact-duplicate group -> eligible, no duplicate metadata.
- same-class duplicate -> eligible by default (`duplicate_policy.same_class_exact_duplicate: keep`
  in `configs/dataset.yaml`), since there is no label conflict.
- cross-class duplicate, `resolution=UNRESOLVED` -> excluded,
  `exclusion_reason=unresolved_cross_class_exact_duplicate`.
- cross-class duplicate, `resolution=EXCLUDE_GROUP` -> excluded,
  `exclusion_reason=human_excluded_cross_class_duplicate` (a human decision,
  reported with a distinct reason from the unresolved case).
- cross-class duplicate, `resolution=KEEP_CLASS` -> eligible. **The row's
  `canonical_class` is always the file's original raw-directory label and
  is never overwritten** — the human-adjudicated label lives only in
  `cross_class_duplicate_review.csv: resolved_class`, joinable by
  `duplicate_group_id`.

`data/audit/eligible_class_distribution.csv` and
`data/audit/eligibility_summary.csv` report raw/excluded/eligible counts
per class and in aggregate, always computed from the actual files —
verified on the real dataset (see below) to total exactly 5,335 raw /
942 excluded / 4,393 eligible, with **zero automatic label decisions**.

**Per-class impact is very uneven** (all 464 groups currently UNRESOLVED,
so this reflects the "exclude everything unresolved" default, not a final
outcome):

| Canonical class | Raw | Excluded | Eligible | Eligible % |
|---|---:|---:|---:|---:|
| Central Serous Chorioretinopathy | 101 | 21 | 80 | 79.21% |
| Diabetic Retinopathy | 1,509 | 59 | 1,450 | 96.09% |
| Disc Edema | 127 | 17 | 110 | 86.61% |
| Glaucoma | 1,349 | 358 | 991 | 73.46% |
| Healthy | 1,024 | 201 | 823 | 80.37% |
| Macular Scar | 444 | 92 | 352 | 79.28% |
| Myopia | 500 | 179 | 321 | 64.20% |
| Pterygium | 17 | 0 | 17 | 100.00% |
| Retinal Detachment | 125 | 4 | 121 | 96.80% |
| Retinitis Pigmentosa | 139 | 11 | 128 | 92.09% |

Myopia (64.2% eligible) and Glaucoma (73.5% eligible) lose the largest
share of their images to unresolved cross-class conflicts — human review
of the Glaucoma/Healthy and Glaucoma/Myopia groups in particular (163 and
143 groups respectively) would recover the most data.

#### Original 70/20/10 split

`python run_pipeline.py prepare_dataset` (`src/data/build_split.py`) builds
a deterministic, stratified 70/20/10 train/val/test split of the ELIGIBLE
population only — never the raw directories directly. It:

- selects only `eligible=true` rows from `dataset_eligibility.csv`, and
  re-verifies every one still exists under `data/raw/` and still hashes to
  its audited SHA-256 (catching any drift since the audit ran);
- treats every exact-duplicate group as one atomic unit that always lands
  in a single split — a same-class group's members are never divided, and
  a cross-class group can only be part of the eligible population at all
  once a human sets `KEEP_CLASS` (in which case the whole group's manifest
  `class` becomes the adjudicated `resolved_class`, while
  `dataset_eligibility.csv`'s `canonical_class` for each file is still
  never touched);
- allocates units to splits with a deterministic largest-remainder +
  greedy-deficit algorithm seeded via `src/utils/seeding.set_seed()` (same
  seed -> byte-identical manifests; different seed -> a different, still
  valid, allocation) — never uncontrolled randomness;
- runs 12 mandatory integrity checks before trusting any manifest
  (`src/data/build_split.py: validate_split_manifests`) — pairwise
  train/val/test overlap, cross-split SHA-256 overlap, duplicate-group
  atomicity, excluded-file leakage, unresolved-cross-class leakage,
  augmentation leakage, class-label consistency, intra-manifest
  duplicate rows, raw-file existence/hash re-verification, and structural
  completeness. Any failure raises `SplitValidationError` before a bad
  manifest can be trusted.

**Verified on the real dataset** (seed 42, git commit visible in
`data/audit/split_metadata.json`): 4,393 eligible images split into
**3,075 train (70.00%) / 880 val (20.03%) / 438 test (9.97%)** — all 12
integrity checks PASS, all 6 same-class duplicate groups stay intact in a
single split each, and (since all 464 cross-class groups remain
UNRESOLVED) zero cross-class-duplicate images appear anywhere in the
split. Per-class counts (train / val / test):

| Canonical class | Train | Val | Test | Total |
|---|---:|---:|---:|---:|
| Central Serous Chorioretinopathy | 56 | 16 | 8 | 80 |
| Diabetic Retinopathy | 1,015 | 290 | 145 | 1,450 |
| Disc Edema | 77 | 23 | 10 | 110 |
| Glaucoma | 694 | 198 | 99 | 991 |
| Healthy | 576 | 165 | 82 | 823 |
| Macular Scar | 246 | 71 | 35 | 352 |
| Myopia | 225 | 64 | 32 | 321 |
| Pterygium | 12 | 3 | 2 | 17 |
| Retinal Detachment | 85 | 24 | 12 | 121 |
| Retinitis Pigmentosa | 89 | 26 | 13 | 128 |

Because duplicate groups must stay atomic and counts are integers, exact
70.00/20.00/10.00 per class is not always achievable (e.g. Disc Edema
lands at 70.00/20.91/9.09) — see `data/audit/split_distribution.csv` for
the exact per-class deviation. This split is **provisional**: it will
change (more images will become eligible) as cross-class duplicate groups
get human-reviewed, so it should be regenerated after any batch of review
decisions, not treated as final while 464 groups remain UNRESOLVED.

Manifest schema (`data/manifests/{train,val,test}_original.csv`): `path`,
`class`, `split`, `original_id`, `parent_original_id`, `is_original`,
`augmentation_type`. Every row here has `is_original=true`,
`parent_original_id=original_id`, `augmentation_type=original` — no
augmented record exists yet. `original_id` is
`sha256(canonical_class|relative_path|sha256)`: deterministic, stable
across machines/runs, and independent of any absolute path (see
`compute_original_id` in `src/data/build_split.py`).

#### Balanced training set (2,000 samples/class)

`python run_pipeline.py prepare_dataset` (`src/data/generate_balanced_dataset.py`)
expands `train_original.csv` up to exactly 2,000 samples per class via
augmentation, writing `data/manifests/train_balanced.csv`. **Validation and
test are never touched by this stage** — they remain original images only,
forever. Key rules, all enforced by explicit checks, not just convention:

- Only rows from `train_original.csv` may ever be augmentation parents; an
  explicit check (`assert_no_val_test_contamination`) fails loudly if a
  validation/test path or ID is ever detected in that role.
- A parent must itself be an original image — an already-generated sample
  can never become a parent (no recursive augmentation).
- Generated files are written only under `data/processed/train/<class>/`,
  never under `data/raw/`, `data/processed/val/`, or `data/processed/test/`
  (which are never created).
- Every one of the class's original training images is kept — the balanced
  set only ever *adds* generated rows, never drops or replaces originals.
- In `train_balanced.csv`, a row's `path` is relative to `raw_dir` when
  `is_original=true` (exactly as in `train_original.csv`) and relative to
  `data/processed/train/` when `is_original=false` — the `is_original`
  column itself tells a reader which base directory to prepend.
- `augmentation_type` records the actual transform applied — one of
  `horizontal_flip`, `rotation`, `brightness_contrast`, `affine`,
  `color_jitter`, or `combined` (never a vague label like "augmented") —
  assigned round-robin across a class's active recipes together with
  round-robin parent selection, so generation is spread evenly rather than
  exhausting one parent or one recipe first.
- Generated IDs are deterministic:
  `sha256(parent_original_id|canonical_class|augmentation_index|seed|augmentation_config_hash)`;
  each generated sample also gets its own seeded RNG from that same tuple,
  so the entire generation is independent of processing order and
  byte-identical given the same original manifest, seed, and config.
- 14 mandatory checks run before any manifest is trusted
  (`validate_balanced_manifest`): exact row/per-class counts, no
  validation/test images, every generated row has a valid original-training
  parent (never a recursive one), generated class matches its parent,
  split is always `train`, every original is retained, no duplicate IDs,
  `is_original` is consistent with `augmentation_type`, and every generated
  file exists, is readable, and matches its expected dimensions.

**Verified on the real dataset** (seed 42): all 10 classes hit exactly
2,000/2,000, for a total of exactly **20,000** balanced training samples —
3,075 original + 16,925 generated. All 14 integrity checks PASS.
Independently re-verified outside the module's own checks: zero validation/test
paths or IDs were ever used as a parent or appear anywhere in
`train_balanced.csv`; zero generated rows have a generated (rather than
original) parent; zero generated files exist under `data/raw/`; every
generated file exists exactly where expected under `data/processed/train/`.

| Canonical class | Original | Generated | Total | Expansion |
|---|---:|---:|---:|---:|
| Central Serous Chorioretinopathy | 56 | 1,944 | 2,000 | 35.7x |
| Diabetic Retinopathy | 1,015 | 985 | 2,000 | 2.0x |
| Disc Edema | 77 | 1,923 | 2,000 | 26.0x |
| Glaucoma | 694 | 1,306 | 2,000 | 2.9x |
| Healthy | 576 | 1,424 | 2,000 | 3.5x |
| Macular Scar | 246 | 1,754 | 2,000 | 8.1x |
| Myopia | 225 | 1,775 | 2,000 | 8.9x |
| Pterygium | 12 | 1,988 | 2,000 | 166.7x |
| Retinal Detachment | 85 | 1,915 | 2,000 | 23.5x |
| Retinitis Pigmentosa | 89 | 1,911 | 2,000 | 22.5x |

Full per-class/per-augmentation-type breakdown (parent reuse counts, mean
and max samples generated per parent) is in
`data/audit/augmentation_statistics.csv`; provenance (seed, config hash,
source manifest hash, git commit) is in
`data/audit/balanced_dataset_metadata.json`.

**This split is provisional in the same sense as the original split above**:
it is built from `train_original.csv`, which currently excludes the 942
images tied up in unresolved cross-class duplicate conflicts — regenerate
it after any batch of human review decisions.

## Model & dataloader infrastructure

Implemented, not yet used for any actual training run.

#### Dataset / transforms / dataloaders (`src/data/dataset.py`, `transforms.py`, `dataloaders.py`)

- `RetinalDataset` (a `torch.utils.data.Dataset`) reads one manifest at a
  time. Manifest choice is fixed by convention, never accidental:
  - **training** may use either `train_original.csv` or
    `train_balanced.csv` — **the default model-training pipeline uses
    `train_balanced.csv`**.
  - **validation always uses `val_original.csv`**, **test always uses
    `test_original.csv`** — both loaded with `require_all_original=True`,
    which raises `DatasetManifestError` if a single augmented row is ever
    present, so a val/test manifest can never silently be swapped for a
    training one.
  - `validate_dataset_manifests()` runs the full one-call consistency
    check across all three (per-manifest value validity, val/test
    original-only enforcement, no cross-manifest path/ID overlap) and is
    reusable by later tasks.
- Every sample is a dict: `image` (tensor after `transform`), `label`
  (int 0-9), `class_name`, `image_path` (the exact path opened —
  resolved against `raw_dir` when `is_original=true`, against
  `data/processed/train/` when `is_original=false`), `original_id`,
  `parent_original_id`, `is_original` (bool). This is deliberately rich
  enough that later evaluation code can write `results/raw_predictions/`
  straight from a dataloader batch without touching this class again.
- **Class index mapping is centralized** in `build_class_to_idx()` —
  canonical class names sorted alphabetically, indices 0-9 assigned in
  that order. This is the *only* place a class name becomes an integer;
  every dataset, split, model, and future experiment must go through it
  (or construct a `RetinalDataset`, which calls it internally) rather than
  ever deriving an ordering from filesystem iteration order.
- **Offline augmentation (Task 4) vs. runtime preprocessing (Task 5) are
  distinct stages, never confused:** Task 4 ran once, ahead of time,
  against `train_original.csv` only, physically baking flips/rotations/
  brightness/contrast/affine/color-jitter into the files referenced by
  `train_balanced.csv`. `src/data/transforms.py`'s `build_train_transform`
  / `build_eval_transform` run every time any sample is loaded — resize
  (shorter side to `image.resize_size`, default 256) → center-crop to
  `image.size` (default 224) → tensor → ImageNet normalization — and are
  **deterministic with no additional randomness**, for training and
  eval/test alike. Validation/test never receive random augmentation of
  any kind, at either stage.
- `build_dataloader` / `build_train_dataloader` (shuffled, `drop_last=True`,
  optional seeded `torch.Generator` for reproducible shuffle order) /
  `build_eval_dataloader` (never shuffled, never drops a sample). Defaults
  (`batch_size=16`, `num_workers=4`) match `configs/dataset.yaml:
  dataloader` for the target RTX 3050 6GB; `pin_memory` auto-disables on a
  CPU-only machine to avoid the usual warning/overhead.

#### Baseline models (`src/models/base.py`, `factory.py`)

All three required baselines — **ResNet50**, **EfficientNetB0**, and
**MaxViT** (`maxvit_tiny_tf_224`) — are implemented as the *same* wrapper
class, `TimmBackboneModel`, around `timm.create_model`, so their very
different internal architectures never leak into training/evaluation code:

```python
logits = model(images)                          # [B, 10]
output = model(images, return_features=True)    # ModelOutput(logits, features)
```

`features` is the pre-classifier pooled embedding (`[B, feature_dim]`),
needed by later tasks (supervised contrastive learning, AA-EvidentNet
fusion, Grad-CAM, uncertainty experiments) without any per-architecture
special-casing. `feature_dim` is read from the instantiated model
(`backbone.num_features`), never hard-coded:

| Model | timm architecture | Parameters (total) | Feature dim |
|---|---|---:|---:|
| ResNet50 | `resnet50` | 23,528,522 | 2,048 |
| EfficientNetB0 | `efficientnet_b0` | 4,020,358 | 1,280 |
| MaxViT | `maxvit_tiny_tf_224` | 30,408,658 | 512 |

(Parameter/feature counts above are generated, not hand-typed — see
`results/tables/model_parameters.csv`, produced by `model_check` below.)
`maxvit_tiny_tf_224` was chosen (and verified against the installed
`timm==1.0.28`, not assumed) as the smallest MaxViT-TF variant with native
224×224 ImageNet-1k pretrained weights, to fit the target RTX 3050 6GB —
a hardware-driven default, not a scientifically tuned one.

`create_model("resnet50" | "efficientnetb0" | "maxvit", config)`
(`src/models/factory.py`) is the only way training/evaluation code should
instantiate a baseline; an unknown name raises `ModelConfigError` with the
list of registered names. All architecture parameters (backbone name,
`pretrained`, `num_classes`, `dropout`) live in `configs/models.yaml:
baselines.*`, never hard-coded in source.

#### CPU smoke testing (`python run_pipeline.py model_check`)

Instantiates each baseline, runs one synthetic forward pass, checks the
output shape is `[B, 10]`, the feature shape is non-empty, and there are
no NaN/Inf values — **never trains**. Runs in a few seconds on CPU with
**`pretrained=False` by default** (fully offline; this is what the pytest
suite also uses, so unit tests never need internet access or a weight
download). Pass `--pretrained` to additionally verify real
ImageNet-pretrained weights can be downloaded and instantiated for all
three models — this **requires internet access** and is a manual/CI-optional
check, never run by default `pytest`. Writes
`results/tables/model_parameters.csv` from the actually-instantiated
models (never hand-typed).

## Training engine (`src/training/`)

Implemented and used for a real (if brief) run; **no model has completed
a full training budget yet** (see below). One reusable `Trainer`
(`src/training/trainer.py`) is shared by every model — baselines now,
AA-EvidentNet later — so training logic is never duplicated per
architecture.

- **Data usage is fixed, not configurable per run**: training always
  reads `data/manifests/train_balanced.csv`; validation always reads
  `data/manifests/val_original.csv`. **The training engine never loads
  `test_original.csv` anywhere** — model selection (checkpointing, LR
  scheduling, early stopping) is driven exclusively by validation
  metrics. This is enforced by omission (the code path simply contains no
  reference to the test manifest), not by a runtime check.
- **Mixed precision** (`torch.amp`) is only ever actually enabled when
  running on CUDA; requesting it on CPU is silently (but loggably, via
  `Trainer.amp_enabled` and the run log) downgraded to disabled — it is
  never falsely reported as active. Device selection: `auto` picks CUDA
  if available else CPU with no error either way; an explicit `--device
  cuda` fails clearly if CUDA isn't actually available.
- **Gradient accumulation** (`gradient_accumulation_steps`) and
  **gradient clipping** (`gradient_clip_norm`, `clip_grad_norm_`) are both
  configurable in `configs/training.yaml`; accumulation correctly steps
  the optimizer on the final (possibly partial) batch of an epoch even if
  it doesn't complete a full accumulation window.
- **Checkpointing**: `best.pt` is saved whenever `monitor_metric`
  (default `val_macro_f1`) improves; `latest.pt` is saved every
  `checkpoint_frequency` epochs. Every checkpoint bundles model/optimizer/
  scheduler state together with metadata (model name, architecture,
  num_classes, seed, epoch, best metric, the full training config,
  the training dataset manifest's hash, and the git commit) — enough to
  know exactly what produced it without consulting anything else.
  Checkpoints for different runs are never overwritten (each run gets its
  own `results/checkpoints/<run_id>/`).
- **Resume** (`--resume <checkpoint.pt>`): restores model/optimizer/
  scheduler/epoch/best-metric and continues training from the next epoch.
  `assert_checkpoint_compatible()` rejects (raises
  `CheckpointIncompatibleError`) a checkpoint whose `model_name` or
  `num_classes` doesn't match the current run — never silently coerced.
- **Early stopping** is validation-only (default: stop after 10 epochs of
  no `val_macro_f1` improvement); the LR scheduler
  (`ReduceLROnPlateau`, default factor 0.5 / patience 5) is likewise
  validation-only. Both, plus "best checkpoint" selection, all key off the
  *same* configured `monitor_metric`/`mode`.
- **Per-run logging**: every run gets its own
  `results/logs/<run_id>/` (never overwritten — `RunLogger` refuses to
  reuse an existing directory) containing `run.log` (human-readable),
  `metrics.jsonl` (one JSON record per epoch), `config.yaml` (the full
  effective config), `environment.txt`, `dataset_hash.txt`, and
  `git_commit.txt`. `run_id` format:
  `YYYYMMDD_HHMMSS_<model>_seed<seed>[_smoke]_<6-hex>` — the trailing
  random suffix guards against collisions from same-second runs, since
  the timestamp alone is not guaranteed unique.
- **Experiment registry** (`experiments/registry.csv`, committed to git —
  unlike the heavy `results/logs`/`results/checkpoints`, this is a
  lightweight ledger worth keeping): one row per run
  (`experiment_id, model, seed, config, checkpoint, test_result, status,
  notes`), registered automatically as `running` at start and updated to
  `completed` or `failed` at the end. **`test_result` is always empty** —
  it is only ever populated by a future, separate test-evaluation task.
- **CPU-compatible smoke test**
  (`python run_pipeline.py baseline --model maxvit --smoke-test`): builds
  a tiny (8 train / 4 val image) fully synthetic dataset on the fly
  (never touches `data/raw/` or the real manifests), forces
  `pretrained=False` (offline, like `model_check`) and a small batch size
  (4, so the loader is guaranteed at least one full batch) and 2 epochs
  regardless of `configs/training.yaml`, then runs the complete real
  pipeline — forward, loss, backward, optimizer step, validation,
  checkpoint save, metrics/log write, registry registration — end to
  end. Every smoke-test run_id/registry note is tagged `smoke` /
  `smoke_test` so it can never be mistaken for a real result.

### The real baseline run: what was and wasn't done

This environment is **CPU-only** (`torch.cuda.is_available()` is
`False`); no GPU was available to attempt full training on. Per this
task's explicit instructions, a full 50-epoch MaxViT training run was
**not** attempted on CPU — a timed check of 3 real batches (48 images)
through MaxViT at 224×224 took ~75 seconds, meaning a full epoch over
the real 20,000-sample `train_balanced.csv` (1,250 batches at
`batch_size=16`) would take on the order of **8+ hours per epoch**,
against a 50-epoch configured budget. Instead:

1. The full smoke test above was run for all three baselines (see
   `experiments/registry.csv`).
2. A real-data loading/forward sanity check was performed directly
   against `train_balanced.csv` (20,000 rows) and `val_original.csv`
   (880 rows) — manifests load, class mapping is correct
   (`{'Central Serous Chorioretinopathy': 0, ..., 'Retinitis
   Pigmentosa': 9}`), a real batch is `[8, 3, 224, 224]` with valid
   labels, MaxViT's forward pass produces `[8, 10]` logits with no
   NaN/Inf, and the optimizer/scheduler both initialize correctly.
3. One very short **real-data sanity-check run** (seed 42, MaxViT,
   real `pretrained=True` ImageNet weights, real `train_balanced.csv`/
   `val_original.csv`, capped at 3 batches for training and 3 for
   validation, 1 epoch) was run end to end through the full orchestration
   — registered in `experiments/registry.csv` with notes explicitly
   reading `REAL_DATA_SANITY_CHECK_ONLY_not_a_completed_baseline...` —
   to prove the entire real pipeline (data → model → optimizer →
   checkpoint → logging → registry) works correctly on real data, without
   claiming this constitutes a trained baseline.

**A controlled, full-budget MaxViT baseline training run has NOT been
performed and no baseline performance numbers exist anywhere in this
repository.** That requires the target CUDA (RTX 3050 6GB) environment
and is left for whenever that hardware is available — this task
deliberately stops at "the engine works, end to end, on real data."

### Original images vs. augmented training samples vs. clinical observations

**The balanced 20,000-sample training set does not represent 20,000
independent clinical observations.** These three counts are distinct and
must never be conflated in any report, figure, or discussion of results:

1. **Original training images** — the real, unique photographs in a class's
   training split (a subset of the raw counts in the table above, after the
   70/20/10 split is applied). For Pterygium this is only 12 images.
2. **Augmented training samples** — the count actually seen per epoch: each
   class is expanded to exactly 2,000 training samples
   (`target_train_samples_per_class`, a fixed methodology choice, not
   provisional) by augmenting its original training images. Augmentation is
   applied to the training split only, never to validation or test data,
   which contain original images only and are never balanced or augmented.
   Every generated sample retains `parent_original_id`, so the full
   augmentation lineage back to a specific original training image is
   always auditable from `train_balanced.csv` alone.
3. **Independent clinical observations/patients** — the number of distinct
   underlying patients/eyes, which is bounded above by (1) and is
   **unaffected by augmentation**.

Reaching 2,000 augmented training samples for a minority class does **not**
mean 2,000 independent clinical observations exist for that class — for
Pterygium, 2,000 training samples come from only 12 original photographs
(a 166.7x expansion). Augmented samples must never be presented as
additional patients. Any statistical claim (confidence intervals,
generalization discussion) must be made with respect to (3), not (2).

## Repository structure

```text
configs/            YAML configuration (dataset, models, losses, training, evaluation, experiments)
data/
  raw/               Original DS2 images, one subfolder per class (present, read-only)
  processed/         train/<class>/ - generated (augmented) training images only; no val/test copies exist
  manifests/         File-level manifests + hashes describing exact dataset versions used
  audit/             Outputs of the dataset audit stage
src/
  data/              Dataset loading, manifest building, preprocessing
  models/            Model architectures (baseline + AA-EvidentNet)
  losses/            Loss functions (cross-entropy, evidential)
  training/          Training loops, optimization
  evaluation/         Metrics, calibration, selective prediction, robustness
  visualization/     Figures, Grad-CAM overlays
  statistics/        Multi-seed aggregation, significance testing
  utils/             Reproducibility utilities (seeding, config hashing, git/env info)
experiments/         One directory per experiment stage (00_data_audit ... 08_robustness)
results/             logs/, checkpoints/, raw_predictions/, tables/, figures/ (all currently empty)
paper/               figures/, tables/, supplementary/, references/ for the eventual write-up
tests/               Unit tests for the utilities and CLI built so far
run_pipeline.py      Single CLI entry point for all pipeline stages
```

## Installation

Verified working environment for this repository:

- Python 3.10.11
- torch 2.10.0 (CPU-only build; no CUDA GPU was detected in this
  environment — see `REPRODUCIBILITY.md` for how to switch to a CUDA build)
- torchvision 0.25.0 and timm 1.0.28 (pinned to match torch 2.10.0 —
  installing torchvision unpinned will pull a newer torch; see
  `REPRODUCIBILITY.md`)
- numpy, pandas, pillow, PyYAML, scikit-learn (see `requirements.txt` for
  exact pinned versions)

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

or with conda:

```bash
conda env create -f environment.yml
conda activate aa-evidentnet
```

## CLI overview

All pipeline stages run through a single entry point:

```bash
python run_pipeline.py <command> [options]
```

Commands (recognized now; `audit`, `prepare_dataset`, `model_check`, and
`baseline` are implemented, the rest fail with a clear message rather
than doing nothing):

| Command            | Purpose                                             |
|---------------------|------------------------------------------------------|
| `audit`             | Dataset audit: counts, duplicates, corruption, leakage |
| `prepare_dataset`   | Audit + eligibility + 70/20/10 original split + balanced (2,000/class) training set |
| `model_check [--pretrained]` | Instantiate ResNet50/EfficientNetB0/MaxViT, run a synthetic forward pass, report shapes/params (never trains; offline unless `--pretrained`) |
| `baseline --model {resnet50,efficientnetb0,maxvit} [--smoke-test] [--resume <ckpt>]` | Train a baseline via the reusable training engine (`src/training/`); trains on `train_balanced.csv`, validates on `val_original.csv`, never touches the test set |
| `train --model aa_evidentnet` | Train the proposed model — **Task 7, not yet implemented**; fails clearly |
| `ablation`          | Ablation studies over proposed-model components     |
| `hard_pairs`        | Confusable-class-pair analysis                      |
| `calibration`       | Calibration + uncertainty quantification evaluation |
| `selective`         | Selective prediction / risk-coverage evaluation     |
| `gradcam`           | Grad-CAM interpretability visualizations            |
| `robustness`        | Robustness to input perturbations                   |
| `multi_seed`        | Aggregate results across seeds                      |
| `publication`       | Assemble publication tables/figures                 |
| `final_test`        | Final held-out test evaluation                      |

Common options: `--config`, `--seed`, `--device`, `--batch-size`, `--epochs`,
`--smoke-test`, `--num-workers`. Run `python run_pipeline.py --help` or
`python run_pipeline.py <command> --help` for details.

## Reproducibility principles

See `REPRODUCIBILITY.md` for the full policy. In short:

- Default seed is **42**; official multi-seed runs use **42, 123, 456, 789,
  2026**.
- Every config is content-hashed (`src/utils/config.hash_config`); every
  dataset manifest is content-hashed (`src/utils/hashing.hash_manifest`).
- Every run should record its git commit and environment info
  (`src/utils/git_info`, `src/utils/env_info`).
- No experimental result is reported unless it was produced by actually
  running the pipeline against the real dataset — this codebase does not
  fabricate placeholder results.

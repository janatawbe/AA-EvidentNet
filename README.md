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

## Planned architecture (not yet implemented)

- **Baseline**: MaxViT-based image classifier with a standard softmax /
  cross-entropy head.
- **Proposed (AA-EvidentNet)**: same or comparable backbone, with an
  evidential (Dirichlet-based) uncertainty head trained with an evidential
  loss, evaluated with Monte Carlo sampling for uncertainty estimates.

Exact architecture details, hyperparameters, and loss formulations are
**provisional** (see `configs/`) and will be finalized based on the data
audit and baseline experiments, not fixed in advance.

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

The raw-data audit (`python run_pipeline.py audit`, implemented in
`src/data/audit_dataset.py` and friends) is complete; dataset splitting,
balancing, and augmentation (`prepare_dataset`) are not yet implemented.

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
must never be split across train/val/test — this is a prerequisite check
for Task 3, not yet implemented. `src/data/duplicate_review.py` exposes
`assert_ready_for_split()` for that future stage to call; it raises
`HumanReviewRequiredError` while any cross-class group remains
`UNRESOLVED` and never assigns a resolution itself.

### Original images vs. augmented training samples vs. clinical observations

These three counts are distinct and must never be conflated in any report,
figure, or discussion of results:

1. **Original training images** — the real, unique photographs in a class's
   training split (a subset of the raw counts in the table above, after the
   70/20/10 split is applied). For Pterygium this is at most ~12 images.
2. **Augmented training samples** — the count actually seen per epoch after
   the training-only augmentation pipeline resamples/augments each class up
   to `target_train_samples_per_class: 2000` (a fixed methodology choice,
   not provisional — see `configs/dataset.yaml`). Augmentation is applied to
   the training split only, never to validation or test data.
3. **Independent clinical observations/patients** — the number of distinct
   underlying patients/eyes, which is bounded above by (1) and is
   **unaffected by augmentation**.

Reaching 2,000 augmented training samples for a minority class does **not**
mean 2,000 independent clinical observations exist for that class. Any
statistical claim (confidence intervals, generalization discussion) must be
made with respect to (3), not (2).

## Repository structure

```text
configs/            YAML configuration (dataset, models, losses, training, evaluation, experiments)
data/
  raw/               Original DS2 images, one subfolder per class (present, read-only)
  processed/         Resized/cleaned images and split data (not yet generated)
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

Commands (recognized now; most are **not yet implemented** and will fail
with a clear message rather than doing nothing):

| Command            | Purpose                                             |
|---------------------|------------------------------------------------------|
| `audit`             | Dataset audit: counts, duplicates, corruption, leakage |
| `prepare_dataset`   | Build processed splits + manifests from `data/raw`  |
| `baseline --model maxvit` | Train/evaluate the baseline model             |
| `train --model aa_evidentnet` | Train the proposed model                 |
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

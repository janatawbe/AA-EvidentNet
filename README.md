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

No deduplication, quality filtering, or leakage checks have been performed
yet — that is the job of the (not yet implemented) `audit` and
`prepare_dataset` pipeline stages.

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

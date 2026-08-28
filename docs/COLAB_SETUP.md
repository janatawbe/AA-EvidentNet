# Running AA-EvidentNet training on a free Google Colab GPU

This guide covers **preparation and a smoke test only** — it does not
start a real training run. The workflow it sets up is:

```text
Laptop repository (this repo, git) --> Google Colab free GPU
    --> train --> checkpoints/logs written to Google Drive (persistent)
    --> pull results back into the laptop repository
```

Nothing about the dataset, split, augmentation methodology, model
architectures, or loss formulations changes for Colab. The only things
that differ from the CPU-only laptop workflow are: (1) the compute device
(`--device` already auto-detects CUDA — see "GPU compatibility" below),
and (2) where the raw dataset and experiment outputs physically live,
since a Colab runtime's local disk is **not persistent**.

## Table of contents

1. [Open Colab and enable a GPU runtime](#1-open-colab-and-enable-a-gpu-runtime)
2. [Clone the repository](#2-clone-the-repository)
3. [Install dependencies](#3-install-dependencies)
4. [Connect the dataset](#4-connect-the-dataset)
5. [Verify the dataset](#5-verify-the-dataset)
6. [Verify CUDA](#6-verify-cuda)
7. [Run a tiny GPU smoke test](#7-run-a-tiny-gpu-smoke-test)
8. [Start a real experiment](#8-start-a-real-experiment-not-part-of-this-preparation-task)
9. [Persistence: saving checkpoints/results](#9-persistence-saving-checkpointsresults)
10. [Resuming an interrupted run](#10-resuming-an-interrupted-run)
11. [Bringing results back to the laptop repository](#11-bringing-results-back-to-the-laptop-repository)

---

## 1. Open Colab and enable a GPU runtime

1. Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.
2. `Runtime -> Change runtime type -> Hardware accelerator -> GPU` (the free tier gives you a T4 or similar; which exact model you get varies and is intentionally never hardcoded anywhere in this project's code — see "GPU compatibility" below).
3. `Runtime -> Connect`.

## 2. Clone the repository

```bash
!git clone <your-repo-url> retinal-diseases
%cd retinal-diseases
!git log --oneline -3   # confirm you're on the commit you expect
```

If you don't want to push this repo anywhere public/private yet, `git bundle`/`scp`/Drive-upload the repo instead — the point is just "get the exact same commit onto the Colab VM," by whatever transport is convenient.

## 3. Install dependencies

Colab's base image already ships a recent, CUDA-enabled PyTorch build — in most cases it is **faster and safer to keep it** than to force-install this repo's CPU-pinned `torch==2.10.0` (that pin, see `requirements.txt` and `REPRODUCIBILITY.md`, records what was verified on the CPU-only development machine, not a requirement that Colab must match exactly).

```bash
!python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

**Recommended (keep Colab's preinstalled torch/torchvision, add the rest):**

```bash
!pip install timm==1.0.28 pandas==2.3.3 pillow==12.1.1 PyYAML==6.0.3 scikit-learn==1.7.2 pytest==9.1.1 matplotlib tqdm
```

**Only if you specifically need `torch==2.10.0` with CUDA** (e.g. to match the pinned version exactly for a reproducibility comparison), install it from the CUDA wheel index instead of plain `pip install torch==2.10.0` (which would resolve the CPU-only wheel used on the laptop):

```bash
!pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu121
!pip install timm==1.0.28 pandas==2.3.3 pillow==12.1.1 PyYAML==6.0.3 scikit-learn==1.7.2 pytest==9.1.1 matplotlib tqdm
```

(`cu121` is an example CUDA tag — check [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) for the tag matching whatever CUDA version Colab's driver reports.)

Either way, run the [GPU verification command](#6-verify-cuda) below afterward to confirm what you actually have installed.

## 4. Connect the dataset

**Do not copy the 5,335-image raw dataset into the git repository, and do not re-download/regenerate it.** The dataset access paths are entirely config-driven (`configs/dataset.yaml: paths.raw_dir / paths.processed_dir / paths.manifests_dir / paths.audit_dir`) and are read by `RetinalDataset`/`RetinalDataset.from_manifest` as plain filesystem paths — nothing in the code assumes a specific location, and manifest CSVs (`data/manifests/*.csv`) store **relative** image paths (`"<Class Name>/<file>.jpg"`), so they remain valid no matter where `raw_dir`/`processed_dir` actually point.

The recommended way to get the dataset onto Colab, in order of preference:

1. **Google Drive (recommended for repeated sessions)**: upload the dataset once to a Drive folder, then in every session:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
   Your data might then live at e.g. `/content/drive/MyDrive/aa-evidentnet-data/raw`.

2. **A cloud bucket / direct download** you control, fetched once per session into local Colab disk (`/content/data/raw`) if you want faster I/O than a Drive mount and don't mind re-fetching every session.

Then create a **Colab-specific dataset config** that changes only the `paths:` section — copy, don't hand-edit, `configs/dataset.yaml`:

```bash
!python -c "
import yaml
with open('configs/dataset.yaml') as f:
    cfg = yaml.safe_load(f)
cfg['paths']['raw_dir'] = '/content/drive/MyDrive/aa-evidentnet-data/raw'
cfg['paths']['processed_dir'] = '/content/drive/MyDrive/aa-evidentnet-data/processed'
cfg['paths']['manifests_dir'] = 'data/manifests'   # tiny CSVs - fine to keep in the cloned repo
cfg['paths']['audit_dir'] = 'data/audit'           # tiny CSVs/JSON - fine to keep in the cloned repo
with open('configs/dataset.colab.yaml', 'w') as f:
    yaml.safe_dump(cfg, f, sort_keys=False)
"
```

**Do not change anything else in this file** — `class_names`, `class_directory_mapping`, `split:` (70/20/10), `target_train_samples_per_class` (2000), and the `augmentation:` block must stay byte-identical to the original, since these are the project's fixed research methodology (see `README.md`/`configs/dataset.yaml`'s own comments). Only `paths:` should ever differ between `configs/dataset.yaml` and `configs/dataset.colab.yaml`.

`data/manifests/train_balanced.csv`, `val_original.csv`, and `test_original.csv` themselves should travel with the git repo (they're small CSVs, already committed) — **do not regenerate or alter them**. The dataset's *images* are what needs external mounting; the manifests that reference them do not change.

## 5. Verify the dataset

Before touching any model, confirm the mounted dataset actually matches what the manifests expect — this does **not** modify anything:

```bash
!python -c "
from src.utils.config import load_config
from src.data.dataset import RetinalDataset
from src.data.transforms import build_transforms_from_config

cfg = load_config('configs/dataset.colab.yaml')
classes = sorted(cfg['class_directory_mapping'].keys())
_, eval_tf = build_transforms_from_config(cfg)

for name, split in [('train_balanced', 'train'), ('val_original', 'val'), ('test_original', 'test')]:
    ds = RetinalDataset.from_manifest(
        f'data/manifests/{name}.csv', classes,
        cfg['paths']['raw_dir'], cfg['paths']['processed_dir'] + '/train',
        transform=eval_tf, expected_split=split,
        require_all_original=(name != 'train_balanced'),
    )
    sample = ds[0]
    print(name, 'samples=', len(ds), 'sample image shape=', tuple(sample['image'].shape), 'label=', sample['label'])
"
```

Expected row counts (unchanged from the laptop repo — **do not regenerate these**): `train_balanced`=20,000, `val_original`=880, `test_original`=438. If `RetinalDataset.from_manifest` raises a path-not-found error, the mount path in `configs/dataset.colab.yaml` is wrong — fix the config, never the manifest.

## 6. Verify CUDA

Run this before starting anything real — it verifies exactly the fields this project's own reproducibility logging already captures automatically in every run's `results/logs/<run_id>/environment.txt` (`src/utils/env_info.collect_environment_info()`):

```bash
!python -c "
import torch
print('torch version         :', torch.__version__)
print('cuda available        :', torch.cuda.is_available())
print('cuda version (torch)  :', torch.version.cuda)
print('cudnn version         :', torch.backends.cudnn.version() if torch.cuda.is_available() else None)
print('gpu count             :', torch.cuda.device_count() if torch.cuda.is_available() else 0)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'gpu[{i}] name          :', torch.cuda.get_device_name(i))
        print(f'gpu[{i}] total memory  :', round(props.total_memory / (1024**3), 2), 'GB')
        free, total = torch.cuda.mem_get_info(i)
        print(f'gpu[{i}] free memory   :', round(free / (1024**3), 2), 'GB')
"
```

If `cuda available` is `False`, go back to step 1 — the runtime type isn't actually set to GPU (or Colab didn't grant one right now; free-tier GPU availability is not guaranteed at all times).

## 7. Run a tiny GPU smoke test

This uses the project's **existing** smoke-test path (`--smoke-test`) — synthetic 8 train / 4 val images, `pretrained=False`, 2 epochs — the same mechanism already used and tested on the CPU-only laptop, now simply run with `--device cuda` (or `--device auto`, which will pick CUDA automatically):

```bash
!python run_pipeline.py baseline --model resnet50 --smoke-test --device cuda
```

Confirm in the printed log line that `amp_enabled=True` (mixed precision only ever actually activates on CUDA — on the laptop's CPU this always correctly read `False`; on a real GPU it should now read `True` since `configs/training.yaml: mixed_precision: true` is already configured). This exercises the entire real pipeline (model, optimizer, AMP `GradScaler`, gradient clipping, checkpoint save, logging, registry) on the actual GPU, without touching real data or running a real experiment.

Also try the AA-EvidentNet path the same way:

```bash
!python run_pipeline.py train --model aa_evidentnet --smoke-test --device cuda
```

**Do not go further than this smoke test as part of this preparation task.**

## 8. Start a real experiment (not part of this preparation task)

When you are ready (a separate, later step — not this one), a real run looks like:

```bash
!python run_pipeline.py baseline --model resnet50 --seed 42 --device cuda --dataset-config configs/dataset.colab.yaml
```

- `--dataset-config` (added for this Colab workflow) points the run at your Drive-mounted `paths:` instead of the laptop's local `configs/dataset.yaml` — no source code changes needed, and no change to `class_names`/split/augmentation/2000-per-class methodology.
- `--device cuda` is explicit here for clarity; `--device auto` (the default) would select CUDA automatically on a GPU runtime and fall back to CPU otherwise — the exact same flag and defaulting behavior as on the laptop, unmodified.
- Do **not** add `--smoke-test`, do **not** pass `--epochs`/`--batch-size` overrides to "make it faster," and do **not** use `--dataset-config` to point at a subset of the data — all of that would change the experiment, not just where it runs.
- `train --model aa_evidentnet` works identically, using the combined classification + CS-SupCon + EDL objective already implemented (`src/losses/combined.py`).

## 9. Persistence: saving checkpoints/results

**Colab's local runtime disk (`/content/...`) is deleted when the runtime disconnects, restarts, or times out — nothing there survives.** The project already writes everything to `results/logs/<run_id>/`, `results/checkpoints/<run_id>/`, and appends to `experiments/registry.csv` (see `EXPERIMENT_LOGGING.md`) — none of that changes for Colab. What changes is *where those directories physically live*.

The simplest reliable approach: symlink (or directly configure) `results/` to a Drive-backed path before starting any run:

```bash
!mkdir -p /content/drive/MyDrive/aa-evidentnet-results
!rm -rf results && ln -s /content/drive/MyDrive/aa-evidentnet-results results
```

Because `src/training/logging.RunLogger`/`checkpointing.save_checkpoint` just do plain `Path(...).mkdir(parents=True)` / file writes, writing through a symlink to Drive works with no code change. Alternatively, override `configs/training.yaml: checkpointing.save_dir` / `logging.log_dir` in a Colab-specific copy of that file the same way you made `configs/dataset.colab.yaml` above — either approach is fine; **do not** change any hyperparameter in that file while you're at it.

`experiments/registry.csv` is small and git-tracked — commit and push it (or otherwise sync it back to the laptop repo) after each session so the ledger of runs stays authoritative in git, exactly like every run on the laptop already does.

`results/raw_predictions/`, `results/tables/`, `results/figures/` (used by later, not-yet-implemented tasks) follow the same rule: whatever directory `results/` resolves to (real disk or the Drive symlink) is where they'll land — nothing about their paths is hardcoded to assume a laptop filesystem.

## 10. Resuming an interrupted run

Colab disconnects, runtime restarts, and browser closures are all "the process died" from the training engine's point of view — exactly the same situation as the laptop's own force-stopped ResNet50 run. The checkpoint/resume mechanism (`src/training/checkpointing.py`, unchanged for Colab) already saves everything needed to resume:

- `model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`
- `scaler_state_dict` (the AMP `GradScaler`'s adaptive loss-scale state — only meaningful on CUDA; on CPU it's always trivial/disabled) — **this was the one real gap fixed as part of this preparation task**: previously the checkpoint schema didn't carry scaler state at all, so a CUDA+AMP resume would silently restart loss scaling from its default rather than where it left off. Old checkpoints without this key still resume correctly (`.get(...)` returns `None`, restore is skipped).
- `epoch`, `best_metric`, `monitor_metric`
- the full `training_config` actually used, `seed`, `model_name`, `architecture`, `num_classes`, `dataset_manifest_hash`, `git_commit`, `timestamp_utc`

As long as `results/checkpoints/<run_id>/` is on persistent storage (step 9), resume after any interruption with:

```bash
!python run_pipeline.py baseline --model resnet50 --seed 42 --device cuda \
    --dataset-config configs/dataset.colab.yaml \
    --resume results/checkpoints/<run_id>/best.pt
```

`assert_checkpoint_compatible()` will refuse (raise `CheckpointIncompatibleError`) if you accidentally point `--resume` at a checkpoint for a different `model_name`/`num_classes` — it never silently loads a mismatched checkpoint.

## 11. Bringing results back to the laptop repository

Once a real run (started separately, not part of this task) has produced checkpoints/logs on Drive:

1. Pull the updated `experiments/registry.csv` back into the laptop's git working tree (or commit it directly from Colab if that clone is the same remote) and commit it there — it's small and already git-tracked.
2. Copy whichever `results/checkpoints/<run_id>/`/`results/logs/<run_id>/` directories you want to keep from Drive back onto the laptop (these are gitignored, same as they already are for laptop-run experiments — large binary checkpoints are not meant to live in git).
3. Do not hand-edit any metric, hash, or timestamp when bringing results back — every provenance field (config hash, dataset/manifest hash, git commit) is written by the same code on Colab as on the laptop, so a run's authenticity can always be traced the same way regardless of where it executed.

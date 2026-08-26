"""Reusable torch DataLoader construction for RetinalDataset.

Defaults (batch_size=16, num_workers=4) match configs/dataset.yaml:
dataloader, chosen for the target RTX 3050 6GB. On a CPU-only environment
(no CUDA), pin_memory defaults to False automatically to avoid the usual
"pin_memory set but no CUDA device" warning/overhead.
"""

from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset


def build_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    shuffle: bool = False,
    drop_last: bool = False,
    pin_memory: Optional[bool] = None,
    persistent_workers: Optional[bool] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """Generic DataLoader builder. Prefer build_train_dataloader() /
    build_eval_dataloader() below for the standard shuffle/drop_last
    conventions.

    `seed`, if given, seeds a dedicated torch.Generator used only for this
    loader's shuffling, so shuffle order is reproducible independent of
    global RNG state / call order elsewhere.
    """
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    if persistent_workers is None:
        persistent_workers = num_workers > 0

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers if num_workers > 0 else False,
        drop_last=drop_last,
        generator=generator,
    )


def build_train_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
    seed: Optional[int] = None,
) -> DataLoader:
    """Training convention: shuffled, drops a final incomplete batch (so
    batch-norm-style layers never see a batch of size 1)."""
    return build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True, seed=seed)


def build_eval_dataloader(
    dataset: Dataset,
    batch_size: int = 16,
    num_workers: int = 4,
) -> DataLoader:
    """Validation/test convention: never shuffled, never drops samples —
    every evaluation sample must be seen exactly once, in a stable order."""
    return build_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False, drop_last=False)

"""Tests for src.data.dataloaders. All tests use num_workers=0 for speed
and to avoid multiprocessing overhead/issues in the test environment."""

import torch
from torch.utils.data import Dataset

from src.data.dataloaders import build_dataloader, build_eval_dataloader, build_train_dataloader


class _TinyDataset(Dataset):
    def __init__(self, n=20):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {"image": torch.full((3, 4, 4), float(idx)), "label": idx % 3}


def test_build_dataloader_respects_batch_size():
    loader = build_dataloader(_TinyDataset(20), batch_size=4, num_workers=0)
    batch = next(iter(loader))
    assert batch["image"].shape[0] == 4
    assert batch["label"].shape[0] == 4


def test_build_train_dataloader_shuffles_and_drops_last():
    dataset = _TinyDataset(10)
    loader = build_train_dataloader(dataset, batch_size=4, num_workers=0, seed=42)
    batches = list(loader)
    # drop_last=True with 10 samples / batch_size 4 -> 2 full batches, remainder dropped.
    assert len(batches) == 2
    assert all(b["image"].shape[0] == 4 for b in batches)


def test_build_eval_dataloader_no_shuffle_no_drop():
    dataset = _TinyDataset(10)
    loader = build_eval_dataloader(dataset, batch_size=4, num_workers=0)
    batches = list(loader)
    total_samples = sum(b["image"].shape[0] for b in batches)
    assert total_samples == 10  # nothing dropped

    labels_in_order = torch.cat([b["label"] for b in batches]).tolist()
    assert labels_in_order == [idx % 3 for idx in range(10)]  # stable, unshuffled order


def test_train_dataloader_deterministic_with_seed():
    dataset = _TinyDataset(20)
    loader_a = build_train_dataloader(dataset, batch_size=4, num_workers=0, seed=42)
    loader_b = build_train_dataloader(dataset, batch_size=4, num_workers=0, seed=42)

    order_a = [b["label"].tolist() for b in loader_a]
    order_b = [b["label"].tolist() for b in loader_b]
    assert order_a == order_b


def test_train_dataloader_different_seed_can_differ():
    dataset = _TinyDataset(20)
    loader_a = build_train_dataloader(dataset, batch_size=4, num_workers=0, seed=42)
    loader_b = build_train_dataloader(dataset, batch_size=4, num_workers=0, seed=123)

    order_a = [b["label"].tolist() for b in loader_a]
    order_b = [b["label"].tolist() for b in loader_b]
    assert order_a != order_b


def test_pin_memory_defaults_to_false_on_cpu_only():
    loader = build_dataloader(_TinyDataset(4), batch_size=2, num_workers=0)
    assert loader.pin_memory == torch.cuda.is_available()

"""Tests for src.utils.seeding: seed setting and reproducible randomness."""

import random

import numpy as np
import torch

from src.utils.seeding import DEFAULT_SEED, SUPPORTED_SEEDS, set_seed


def test_default_seed_is_42():
    assert DEFAULT_SEED == 42


def test_supported_seeds_contains_required_values():
    assert SUPPORTED_SEEDS == [42, 123, 456, 789, 2026]


def test_set_seed_reproducible_python_random():
    set_seed(42)
    values_a = [random.random() for _ in range(5)]
    set_seed(42)
    values_b = [random.random() for _ in range(5)]
    assert values_a == values_b


def test_set_seed_reproducible_numpy():
    set_seed(42)
    a = np.random.rand(5)
    set_seed(42)
    b = np.random.rand(5)
    assert np.array_equal(a, b)


def test_set_seed_reproducible_torch():
    set_seed(42)
    a = torch.rand(5)
    set_seed(42)
    b = torch.rand(5)
    assert torch.equal(a, b)


def test_set_seed_different_seeds_differ():
    set_seed(42)
    a = torch.rand(5)
    set_seed(123)
    b = torch.rand(5)
    assert not torch.equal(a, b)


def test_set_seed_accepts_all_supported_seeds():
    for seed in SUPPORTED_SEEDS:
        set_seed(seed)  # must not raise

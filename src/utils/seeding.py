"""Random seed control and deterministic-behavior helpers.

The project standardizes on a default seed of 42 and a fixed set of seeds
used for multi-seed robustness experiments (see configs/experiments.yaml).
"""

import os
import random

DEFAULT_SEED = 42

# Seeds used across multi-seed experiments (statistics/robustness reporting).
SUPPORTED_SEEDS = [42, 123, 456, 789, 2026]


def set_seed(seed: int = DEFAULT_SEED, deterministic: bool = True) -> None:
    """Seed all known sources of randomness and optionally force determinism.

    Args:
        seed: Seed value. Any integer is accepted; SUPPORTED_SEEDS lists the
            seeds used for the project's official multi-seed runs.
        deterministic: If True, configure PyTorch (when available) to prefer
            deterministic algorithms. This can reduce performance and, for
            some ops, is not fully supported on CPU-only builds.
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    except ImportError:
        pass

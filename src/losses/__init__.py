"""Loss functions.

CS-SupCon (Class-Similarity Supervised Contrastive Loss, Task 8) is
implemented in `cs_supcon.py`. EDL (Evidential Deep Learning, Task 9) is
implemented in `evidential.py`. Neither is yet wired into a combined
training objective or the training loop.
"""

from .cs_supcon import (
    AmbiguityPairs,
    CSSupConConfigError,
    CSSupConLoss,
    CSSupConSettings,
    cs_supcon_loss,
    load_cs_supcon_settings,
    resolve_ambiguity_pairs,
)
from .evidential import (
    EDLLoss,
    EDLSettings,
    EvidentialConfigError,
    EvidentialHead,
    EvidentialOutput,
    compute_evidential_output,
    edl_loss,
    load_edl_settings,
)

__all__ = [
    "AmbiguityPairs",
    "CSSupConConfigError",
    "CSSupConLoss",
    "CSSupConSettings",
    "cs_supcon_loss",
    "load_cs_supcon_settings",
    "resolve_ambiguity_pairs",
    "EDLLoss",
    "EDLSettings",
    "EvidentialConfigError",
    "EvidentialHead",
    "EvidentialOutput",
    "compute_evidential_output",
    "edl_loss",
    "load_edl_settings",
]

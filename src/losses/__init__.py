"""Loss functions.

CS-SupCon (Class-Similarity Supervised Contrastive Loss, Task 8) is
implemented in `cs_supcon.py`. Evidential deep learning (EDL) losses are
not yet implemented.
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

__all__ = [
    "AmbiguityPairs",
    "CSSupConConfigError",
    "CSSupConLoss",
    "CSSupConSettings",
    "cs_supcon_loss",
    "load_cs_supcon_settings",
    "resolve_ambiguity_pairs",
]

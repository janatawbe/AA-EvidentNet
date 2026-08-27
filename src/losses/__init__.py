"""Loss functions.

CS-SupCon (Class-Similarity Supervised Contrastive Loss, Task 8) is
implemented in `cs_supcon.py`. EDL (Evidential Deep Learning, Task 9) is
implemented in `evidential.py`. The combined AA-EvidentNet training
objective (classification + CS-SupCon + EDL, Task 7 completion) is
implemented in `combined.py` and used by
`src.training.run_aa_evidentnet`.
"""

from .combined import (
    CombinedAAEvidentNetLoss,
    CombinedObjectiveConfigError,
    CombinedObjectiveSettings,
    build_combined_aa_evidentnet_loss,
    load_combined_objective_settings,
)
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
    "CombinedAAEvidentNetLoss",
    "CombinedObjectiveConfigError",
    "CombinedObjectiveSettings",
    "build_combined_aa_evidentnet_loss",
    "load_combined_objective_settings",
]

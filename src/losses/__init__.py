"""Loss functions.

CS-SupCon (Class-Similarity Supervised Contrastive Loss, Task 8) is
implemented in `cs_supcon.py`. EDL (Evidential Deep Learning, Task 9) is
implemented in `evidential.py`. The combined AA-EvidentNet training
objective (classification + CS-SupCon + EDL, Task 7 completion) is
implemented in `combined.py` and used by
`src.training.run_aa_evidentnet`. `ambiguity.py` (feature/learned-
ambiguity, Phase 1) implements the learned class-level ambiguity matrix
and the (currently analysis-only) sample-level ambiguity score that
`cs_supcon.py`'s `ambiguity_source="learned_class"` mode consumes.
"""

from .ambiguity import (
    AmbiguityComputationError,
    AmbiguityConfigError,
    AmbiguitySettings,
    DEFAULT_AMBIGUITY_SCALE,
    MarginNormalization,
    SampleAmbiguityResult,
    VALID_AMBIGUITY_SOURCES,
    class_ambiguity_matrix_to_buffer,
    compute_class_ambiguity_matrix,
    compute_raw_margins,
    compute_sample_ambiguity,
    fit_margin_normalization,
    load_ambiguity_settings,
)
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
    "AmbiguityComputationError",
    "AmbiguityConfigError",
    "AmbiguitySettings",
    "DEFAULT_AMBIGUITY_SCALE",
    "MarginNormalization",
    "SampleAmbiguityResult",
    "VALID_AMBIGUITY_SOURCES",
    "class_ambiguity_matrix_to_buffer",
    "compute_class_ambiguity_matrix",
    "compute_raw_margins",
    "compute_sample_ambiguity",
    "fit_margin_normalization",
    "load_ambiguity_settings",
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

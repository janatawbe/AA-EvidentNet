"""Combined AA-EvidentNet training objective (Task 7 completion).

    L_total = L_classification + cs_supcon_weight * L_CS-SupCon + edl_weight * L_EDL

This module does not implement any of the three terms itself - it wires
together `nn.CrossEntropyLoss`, `src.losses.cs_supcon.CSSupConLoss` (Task 8),
and `src.losses.evidential.EDLLoss` (Task 9), each already implemented and
independently unit-tested. No loss math is duplicated or altered here.

`cs_supcon_weight` and `edl_weight` are read directly from the existing
`configs/losses.yaml: cs_supcon.loss_weight` / `edl.loss_weight` fields -
those fields already existed specifically "reserved for the future
combined training objective" (see their PROVISIONAL comments in
configs/losses.yaml); this module is what finally consumes them. No new
weight parameters are invented. `cs_supcon.enabled: false` / `edl.enabled:
false` drop that term entirely (weight forced to 0, module not
constructed) rather than merely zeroing its weight.

Operates on an AAEvidentNetOutput (or any object exposing `.logits`,
`.embedding`, `.dirichlet_alpha`) and integer labels - matches
`src.models.aa_evidentnet.AAEvidentNetOutput` exactly, but does not import
that module (kept architecture-agnostic, like its two constituent losses).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn

from src.losses.cs_supcon import CSSupConLoss, CSSupConSettings, load_cs_supcon_settings
from src.losses.evidential import EDLLoss, EDLSettings, load_edl_settings

VALID_CLASS_WEIGHTING = ("none",)


class CombinedObjectiveConfigError(Exception):
    """Raised for an invalid combined-objective configuration (e.g. an
    unsupported class_weighting scheme). Never silently ignored."""


@dataclass(frozen=True)
class CombinedObjectiveSettings:
    label_smoothing: float
    cs_supcon: CSSupConSettings
    edl: EDLSettings


def load_combined_objective_settings(
    losses_config: Dict[str, Any], canonical_classes: Sequence[str]
) -> CombinedObjectiveSettings:
    """Parse+validate the existing configs/losses.yaml sections
    (`baseline`, `cs_supcon`, `edl`) needed to build the combined
    objective. Reuses `load_cs_supcon_settings`/`load_edl_settings`
    unmodified - no separate/duplicate validation of those two sections."""
    baseline_cfg = losses_config.get("baseline", {}) or {}
    label_smoothing = float(baseline_cfg.get("label_smoothing", 0.0))
    class_weighting = baseline_cfg.get("class_weighting", "none") or "none"
    if class_weighting not in VALID_CLASS_WEIGHTING:
        raise CombinedObjectiveConfigError(
            f"losses.yaml: baseline.class_weighting='{class_weighting}' is not implemented yet "
            f"(only {VALID_CLASS_WEIGHTING} is supported) - do not silently ignore a configured "
            "class-weighting scheme."
        )
    if not (0.0 <= label_smoothing < 1.0):
        raise CombinedObjectiveConfigError(f"losses.yaml: baseline.label_smoothing must be in [0, 1), got {label_smoothing}")

    cs_supcon_settings = load_cs_supcon_settings(losses_config, canonical_classes)
    edl_settings = load_edl_settings(losses_config)

    return CombinedObjectiveSettings(label_smoothing=label_smoothing, cs_supcon=cs_supcon_settings, edl=edl_settings)


class CombinedAAEvidentNetLoss(nn.Module):
    """L_total = L_classification + cs_supcon_weight*L_CS-SupCon + edl_weight*L_EDL.

    `forward(output, labels)` expects `output.logits` ([B, num_classes]),
    and, whenever the corresponding term is enabled, `output.embedding`
    ([B, D], for CS-SupCon) and `output.dirichlet_alpha` ([B, num_classes],
    for EDL) - exactly the fields `AAEvidentNetOutput` (return_features=True)
    provides. Gradients flow from all three terms back through whichever
    parts of the model produced those tensors (classifier, both
    projections + the fusion gate `alpha` for the embedding, and the
    evidential head for `dirichlet_alpha`), since PyTorch sums gradients
    from every path when a scalar loss depends on multiple outputs of the
    same forward pass.

    `set_epoch(epoch)` is called by `Trainer.fit()` once per epoch (if
    present) purely so EDL's KL-annealing coefficient can advance with
    training progress; `nn.CrossEntropyLoss` and `CSSupConLoss` do not
    depend on epoch at all.
    """

    def __init__(
        self,
        cs_supcon_loss: Optional[CSSupConLoss],
        cs_supcon_weight: float,
        edl_loss_module: Optional[EDLLoss],
        edl_weight: float,
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__()
        if cs_supcon_weight < 0:
            raise CombinedObjectiveConfigError(f"cs_supcon_weight must be >= 0, got {cs_supcon_weight}")
        if edl_weight < 0:
            raise CombinedObjectiveConfigError(f"edl_weight must be >= 0, got {edl_weight}")

        self.classification_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.cs_supcon_loss = cs_supcon_loss
        self.cs_supcon_weight = cs_supcon_weight if cs_supcon_loss is not None else 0.0
        self.edl_loss_module = edl_loss_module
        self.edl_weight = edl_weight if edl_loss_module is not None else 0.0
        self._current_epoch: Optional[int] = None

        # Populated after every forward() call - the unweighted value of
        # each term plus the final combined total, for logging/tests.
        # Never used to influence training itself.
        self.last_components: Dict[str, float] = {}

    @classmethod
    def from_settings(cls, settings: CombinedObjectiveSettings) -> "CombinedAAEvidentNetLoss":
        cs_supcon_module = CSSupConLoss.from_settings(settings.cs_supcon) if settings.cs_supcon.enabled else None
        edl_module = EDLLoss.from_settings(settings.edl) if settings.edl.enabled else None
        return cls(
            cs_supcon_loss=cs_supcon_module,
            cs_supcon_weight=settings.cs_supcon.loss_weight,
            edl_loss_module=edl_module,
            edl_weight=settings.edl.loss_weight,
            label_smoothing=settings.label_smoothing,
        )

    def set_epoch(self, epoch: int) -> None:
        self._current_epoch = epoch

    def forward(self, output: Any, labels: torch.Tensor) -> torch.Tensor:
        logits = getattr(output, "logits", None)
        if logits is None:
            raise ValueError("CombinedAAEvidentNetLoss requires a model output exposing `.logits`")

        classification = self.classification_loss(logits, labels)
        total = classification
        components = {"classification": float(classification.detach())}

        if self.cs_supcon_loss is not None and self.cs_supcon_weight > 0:
            embedding = getattr(output, "embedding", None)
            if embedding is None:
                raise ValueError(
                    "CombinedAAEvidentNetLoss: CS-SupCon is enabled but the model output has no `embedding`"
                )
            cs_value = self.cs_supcon_loss(embedding, labels)
            total = total + self.cs_supcon_weight * cs_value
            components["cs_supcon"] = float(cs_value.detach())

        if self.edl_loss_module is not None and self.edl_weight > 0:
            dirichlet_alpha = getattr(output, "dirichlet_alpha", None)
            if dirichlet_alpha is None:
                raise ValueError(
                    "CombinedAAEvidentNetLoss: EDL is enabled but the model output has no `dirichlet_alpha`"
                )
            edl_value = self.edl_loss_module(dirichlet_alpha, labels, epoch=self._current_epoch)
            total = total + self.edl_weight * edl_value
            components["edl"] = float(edl_value.detach())

        components["total"] = float(total.detach())
        self.last_components = components
        return total


def build_combined_aa_evidentnet_loss(
    losses_config: Dict[str, Any], canonical_classes: Sequence[str]
) -> CombinedAAEvidentNetLoss:
    """Convenience one-step builder: load+validate settings from
    configs/losses.yaml, then construct the combined loss module."""
    settings = load_combined_objective_settings(losses_config, canonical_classes)
    return CombinedAAEvidentNetLoss.from_settings(settings)

"""Common model interface for all baseline classifiers.

Every baseline (ResNet50, EfficientNetB0, MaxViT) is wrapped by the SAME
class, TimmBackboneModel, around timm.create_model — so their very
different internal architectures never leak into training/evaluation
code. Every model exposes:

    logits = model(images)                        # [B, num_classes]
    output = model(images, return_features=True)   # ModelOutput(logits, features)

`features` is the pre-classifier pooled embedding (via timm's
forward_head(..., pre_logits=True)), shape [B, feature_dim]. feature_dim
varies per architecture (see model.feature_dim / configs/models.yaml) and
is read from the instantiated model, never hard-coded — later tasks
(supervised contrastive learning, AA-EvidentNet fusion, Grad-CAM,
uncertainty experiments) can rely on this without any per-architecture
special-casing.
"""

from dataclasses import dataclass
from typing import Any, Optional

import timm
import torch
import torch.nn as nn


class UnknownModelError(Exception):
    """Raised when timm cannot create the requested architecture (unknown
    name, or a name/pretrained-weights combination it doesn't support)."""


@dataclass
class ModelOutput:
    logits: torch.Tensor
    features: Optional[torch.Tensor] = None


class TimmBackboneModel(nn.Module):
    def __init__(
        self,
        architecture: str,
        num_classes: int = 10,
        pretrained: bool = True,
        dropout: float = 0.0,
        **timm_kwargs: Any,
    ) -> None:
        super().__init__()
        self.architecture = architecture
        self.num_classes = num_classes

        try:
            self.backbone = timm.create_model(
                architecture,
                pretrained=pretrained,
                num_classes=num_classes,
                drop_rate=dropout,
                **timm_kwargs,
            )
        except Exception as e:  # noqa: BLE001 - re-raise as a project-specific, clearer error
            raise UnknownModelError(
                f"timm could not create model '{architecture}' "
                f"(pretrained={pretrained}, num_classes={num_classes}): {e}"
            ) from e

        self.feature_dim = self.backbone.num_features

    def forward(self, images: torch.Tensor, return_features: bool = False):
        if images.dim() != 4:
            raise ValueError(
                f"{self.architecture}: expected a 4D image batch tensor [B, C, H, W], "
                f"got shape {tuple(images.shape)}"
            )

        feature_map = self.backbone.forward_features(images)
        logits = self.backbone.forward_head(feature_map)

        if not return_features:
            return logits

        features = self.backbone.forward_head(feature_map, pre_logits=True)
        return ModelOutput(logits=logits, features=features)

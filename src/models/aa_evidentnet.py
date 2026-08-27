"""AA-EvidentNet architecture (Task 7): Ambiguity-Aware Global-Local
Representation Learning with Evidential Uncertainty for Reliable
Multi-Class Ophthalmic Classification.

ARCHITECTURE ONLY. This module deliberately does NOT implement:
  - CS-SupCon (supervised contrastive) training objective
  - EDL (evidential deep learning) loss / Dirichlet uncertainty head
  - any training loop, ablation, or evaluation logic
Those are later tasks. What this module DOES provide is the full forward
architecture those later tasks will attach to: a global branch, a local
branch, learnable adaptive fusion into a shared embedding, and a
classification head - with every intermediate representation exposed for
reuse (global feature, local feature, fused embedding, logits, alpha).

    Global branch (MaxViT backbone, configurable)
        images -> timm backbone (num_classes=0, pooled features)
                -> Linear projection -> embedding_dim

    Local branch (lightweight CNN, LocalBranch below)
        images -> 4 conv/BN/ReLU blocks (stride 2 each) -> global pool
                -> Linear projection -> embedding_dim

    Adaptive fusion
        alpha = sigmoid(learnable scalar parameter)   # in (0, 1)
        fused = alpha * global_feature + (1 - alpha) * local_feature

    Classification head
        logits = Linear(embedding_dim, num_classes)(fused)

Interface mirrors src.models.base.TimmBackboneModel exactly, so
AA-EvidentNet is a drop-in replacement anywhere a baseline model is used
(Trainer, checkpointing, create_model()):

    logits = model(images)                          # [B, num_classes]
    output = model(images, return_features=True)     # AAEvidentNetOutput
"""

from dataclasses import dataclass
from typing import Optional

import timm
import torch
import torch.nn as nn

from src.models.base import ModelOutput, UnknownModelError


@dataclass
class AAEvidentNetOutput(ModelOutput):
    """Extends ModelOutput (logits, features) with the extra
    representations later tasks (CS-SupCon, EDL, Grad-CAM, uncertainty)
    will need. `features` aliases `embedding` (the fused representation)
    so any code written against the baseline ModelOutput interface still
    works unchanged."""

    embedding: Optional[torch.Tensor] = None
    global_feature: Optional[torch.Tensor] = None
    local_feature: Optional[torch.Tensor] = None
    alpha: Optional[torch.Tensor] = None


class LocalBranch(nn.Module):
    """Lightweight convolutional branch for localized disease-relevant
    features. Deliberately small and modular relative to the global
    MaxViT backbone - four stride-2 conv/BN/ReLU blocks followed by global
    average pooling, not a second full-scale backbone. Works at any input
    resolution (including the tiny synthetic sizes used in unit tests)."""

    def __init__(self, in_channels: int = 3, out_dim: int = 128, dropout: float = 0.0):
        super().__init__()
        self.out_dim = out_dim
        channels = [in_channels, 32, 64, 128, out_dim]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Conv2d(c_in, c_out, kernel_size=3, stride=2, padding=1))
            blocks.append(nn.BatchNorm2d(c_out))
            blocks.append(nn.ReLU(inplace=True))
        self.conv_blocks = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.conv_blocks(images)
        pooled = self.pool(feature_map).flatten(1)
        return self.dropout(pooled)


class AAEvidentNet(nn.Module):
    def __init__(
        self,
        global_backbone: str = "maxvit_tiny_tf_224",
        num_classes: int = 10,
        embedding_dim: int = 256,
        local_feature_dim: int = 128,
        pretrained: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.architecture = f"aa_evidentnet[{global_backbone}]"
        self.global_backbone_name = global_backbone
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim

        try:
            # num_classes=0: timm returns the pooled pre-classifier
            # feature directly from a plain forward() call - no separate
            # forward_features/forward_head dance needed since this
            # backbone's own classifier is never used (AA-EvidentNet has
            # its own, on the fused embedding).
            self.global_backbone = timm.create_model(global_backbone, pretrained=pretrained, num_classes=0)
        except Exception as e:  # noqa: BLE001 - re-raise as a project-specific, clearer error
            raise UnknownModelError(
                f"timm could not create AA-EvidentNet's global backbone '{global_backbone}' "
                f"(pretrained={pretrained}): {e}"
            ) from e

        self.global_feature_dim = self.global_backbone.num_features
        self.local_feature_dim = local_feature_dim

        self.local_branch = LocalBranch(in_channels=3, out_dim=local_feature_dim, dropout=dropout)

        self.global_projection = nn.Linear(self.global_feature_dim, embedding_dim)
        self.local_projection = nn.Linear(local_feature_dim, embedding_dim)

        # A single learnable scalar, sigmoid-constrained to (0, 1) so the
        # fusion weight always stays interpretable/stable regardless of
        # how the raw parameter drifts during training. Initialized at 0
        # -> alpha starts at exactly 0.5 (equal global/local trust).
        self._alpha_raw = nn.Parameter(torch.zeros(1))

        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(embedding_dim, num_classes)

        # Interface parity with TimmBackboneModel: `feature_dim` is what
        # generic code (checkpointing, future evaluation code) reads to
        # learn the size of "the" representation - for AA-EvidentNet
        # that's the shared fused embedding.
        self.feature_dim = embedding_dim

    @property
    def alpha(self) -> torch.Tensor:
        """The current fusion gate value, in (0, 1). A single learned
        scalar shared across every sample - not input-dependent."""
        return torch.sigmoid(self._alpha_raw)

    def forward(self, images: torch.Tensor, return_features: bool = False):
        if images.dim() != 4:
            raise ValueError(
                f"{self.architecture}: expected a 4D image batch tensor [B, C, H, W], "
                f"got shape {tuple(images.shape)}"
            )
        batch_size = images.size(0)

        raw_global = self.global_backbone(images)
        raw_local = self.local_branch(images)

        global_feature = self.global_projection(raw_global)
        local_feature = self.local_projection(raw_local)

        alpha = self.alpha
        fused = alpha * global_feature + (1.0 - alpha) * local_feature
        fused = self.dropout(fused)

        logits = self.classifier(fused)

        if not return_features:
            return logits

        return AAEvidentNetOutput(
            logits=logits,
            features=fused,
            embedding=fused,
            global_feature=global_feature,
            local_feature=local_feature,
            alpha=alpha.expand(batch_size, 1),
        )

"""Class-Similarity Supervised Contrastive Loss (CS-SupCon) — Task 8.

Extends standard supervised contrastive learning (Khosla et al., 2020,
"Supervised Contrastive Learning") with explicit class-ambiguity
awareness. Ordinary SupCon's contrastive denominator treats every
incorrect (negative) class identically; this project specifically cares
about clinically confusable ophthalmic class pairs (see
configs/losses.yaml: cs_supcon.ambiguity_pairs and README.md), so
CS-SupCon upweights exactly those configured ambiguous negative pairs in
the denominator — pushing the model to separate embeddings harder for
pairs a clinician could plausibly confuse, while leaving ordinary
(unrelated) negatives at the standard weight.

This module implements ONLY the loss. It is architecture-agnostic (any
[B, D] embedding tensor + [B] integer labels — e.g. AA-EvidentNet's fused
embedding, but nothing here imports or depends on that model), does not
implement EDL, and is not wired into the training loop — see
configs/losses.yaml and README.md for how this fits into the eventual
combined training objective (classification + CS-SupCon + EDL).

Formulation (for anchor i in a batch, with L2-normalized embeddings z):

    sim(i, a)   = z_i . z_a / temperature
    P(i)        = {p != i : label(p) == label(i)}          (positives)
    w(i, a)     = ambiguity_weight   if label(a) != label(i) and
                                        (label(i), label(a)) is a
                                        configured ambiguous pair
                = 1.0                otherwise (a != i)
                = 0.0                if a == i (self excluded)
    D_i         = sum_{a != i} w(i, a) * exp(sim(i, a))
    L_i         = -(1 / |P(i)|) * sum_{p in P(i)} log( exp(sim(i, p)) / D_i )

Total loss = mean of L_i over anchors i that have at least one positive
(anchors with no same-class counterpart in the batch contribute nothing —
see _NO_POSITIVE handling below — rather than producing NaN/Inf).

Numerically stable via the standard log-sum-exp trick (subtract each
row's max similarity before exponentiating, detached so it doesn't affect
gradients).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_TEMPERATURE = 0.1
DEFAULT_LOSS_WEIGHT = 1.0
DEFAULT_AMBIGUITY_WEIGHT = 2.0


class CSSupConConfigError(Exception):
    """Raised for an invalid CS-SupCon configuration: an unknown class
    name, a self-paired or duplicate ambiguity pair, or an out-of-range
    temperature/weight. Never silently corrected or ignored."""


@dataclass(frozen=True)
class AmbiguityPairs:
    """Resolved (canonical class INDEX) representation of the configured
    ambiguity pairs. Built once via resolve_ambiguity_pairs() against the
    project's single canonical class ordering — never a separately
    hard-coded index mapping inside this module."""

    pairs: frozenset

    def contains(self, a: int, b: int) -> bool:
        return frozenset((a, b)) in self.pairs

    def __len__(self) -> int:
        return len(self.pairs)

    @classmethod
    def empty(cls) -> "AmbiguityPairs":
        return cls(pairs=frozenset())


def resolve_ambiguity_pairs(
    ambiguity_pair_names: Sequence[Sequence[str]],
    canonical_classes: Sequence[str],
) -> AmbiguityPairs:
    """Resolve configured (class_name_a, class_name_b) pairs into
    canonical-index pairs.

    `canonical_classes` must be the project's canonical class list
    (configs/dataset.yaml: class_names); indices are assigned by sorting
    it alphabetically, exactly matching
    src.data.dataset.build_class_to_idx — this function never invents a
    second, independent class-to-index mapping.

    Raises CSSupConConfigError (listing every problem found, not just the
    first) for: a pair that isn't exactly 2 names, an unknown class name,
    a class paired with itself, or a duplicate/conflicting pair
    (order-independent — (A, B) and (B, A) are the same pair).
    """
    class_to_idx = {name: idx for idx, name in enumerate(sorted(canonical_classes))}
    resolved = set()
    errors = []

    for i, pair in enumerate(ambiguity_pair_names):
        pair = list(pair)
        if len(pair) != 2:
            errors.append(f"ambiguity_pairs[{i}]: expected exactly 2 class names, got {pair}")
            continue

        name_a, name_b = pair
        valid = True
        if name_a not in class_to_idx:
            errors.append(f"ambiguity_pairs[{i}]: unknown class name '{name_a}'")
            valid = False
        if name_b not in class_to_idx:
            errors.append(f"ambiguity_pairs[{i}]: unknown class name '{name_b}'")
            valid = False
        if valid and name_a == name_b:
            errors.append(f"ambiguity_pairs[{i}]: a class cannot be ambiguous with itself ('{name_a}')")
            valid = False

        if valid:
            idx_pair = frozenset((class_to_idx[name_a], class_to_idx[name_b]))
            if idx_pair in resolved:
                errors.append(f"ambiguity_pairs[{i}]: duplicate/conflicting pair ({name_a}, {name_b})")
            resolved.add(idx_pair)

    if errors:
        formatted = "\n".join(f"  - {e}" for e in errors)
        raise CSSupConConfigError(f"Invalid cs_supcon.ambiguity_pairs configuration:\n{formatted}")

    return AmbiguityPairs(pairs=frozenset(resolved))


@dataclass(frozen=True)
class CSSupConSettings:
    enabled: bool
    temperature: float
    loss_weight: float
    ambiguity_weight: float
    ambiguity_pairs: AmbiguityPairs


def load_cs_supcon_settings(losses_config: Dict[str, Any], canonical_classes: Sequence[str]) -> CSSupConSettings:
    """Parse+validate configs/losses.yaml: cs_supcon into a CSSupConSettings.
    Fails clearly (CSSupConConfigError) on any invalid value rather than
    silently clamping/defaulting a bad one."""
    section = losses_config.get("cs_supcon") or {}

    enabled = bool(section.get("enabled", True))
    temperature = float(section.get("temperature", DEFAULT_TEMPERATURE))
    loss_weight = float(section.get("loss_weight", DEFAULT_LOSS_WEIGHT))
    ambiguity_weight = float(section.get("ambiguity_weight", DEFAULT_AMBIGUITY_WEIGHT))
    raw_pairs = section.get("ambiguity_pairs", []) or []

    errors = []
    if temperature <= 0:
        errors.append(f"cs_supcon.temperature must be > 0, got {temperature}")
    if loss_weight < 0:
        errors.append(f"cs_supcon.loss_weight must be >= 0, got {loss_weight}")
    if ambiguity_weight <= 0:
        errors.append(f"cs_supcon.ambiguity_weight must be > 0, got {ambiguity_weight}")
    if errors:
        raise CSSupConConfigError("Invalid cs_supcon configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    ambiguity_pairs = resolve_ambiguity_pairs(raw_pairs, canonical_classes)

    return CSSupConSettings(
        enabled=enabled,
        temperature=temperature,
        loss_weight=loss_weight,
        ambiguity_weight=ambiguity_weight,
        ambiguity_pairs=ambiguity_pairs,
    )


class CSSupConLoss(nn.Module):
    """Class-Similarity Supervised Contrastive Loss. See module docstring
    for the exact formulation."""

    def __init__(
        self,
        temperature: float = DEFAULT_TEMPERATURE,
        ambiguity_weight: float = DEFAULT_AMBIGUITY_WEIGHT,
        ambiguity_pairs: Optional[AmbiguityPairs] = None,
    ) -> None:
        super().__init__()
        if temperature <= 0:
            raise CSSupConConfigError(f"temperature must be > 0, got {temperature}")
        if ambiguity_weight <= 0:
            raise CSSupConConfigError(f"ambiguity_weight must be > 0, got {ambiguity_weight}")
        self.temperature = temperature
        self.ambiguity_weight = ambiguity_weight
        self.ambiguity_pairs = ambiguity_pairs if ambiguity_pairs is not None else AmbiguityPairs.empty()

    @classmethod
    def from_settings(cls, settings: CSSupConSettings) -> "CSSupConLoss":
        return cls(
            temperature=settings.temperature,
            ambiguity_weight=settings.ambiguity_weight,
            ambiguity_pairs=settings.ambiguity_pairs,
        )

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor, num_classes: Optional[int] = None) -> torch.Tensor:
        if embeddings.dim() != 2:
            raise ValueError(f"CS-SupCon expects embeddings of shape [B, D], got {tuple(embeddings.shape)}")
        batch_size, embedding_dim = embeddings.shape
        if embedding_dim == 0:
            raise ValueError("CS-SupCon: embedding dimension must be > 0")

        labels = labels.view(-1)
        if labels.numel() != batch_size:
            raise ValueError(
                f"CS-SupCon: labels must have shape [B]={batch_size}, got {tuple(labels.shape)}"
            )
        if num_classes is not None and batch_size > 0:
            if bool((labels < 0).any()) or bool((labels >= num_classes).any()):
                raise ValueError(
                    f"CS-SupCon: labels must be in [0, {num_classes - 1}], got range "
                    f"[{int(labels.min())}, {int(labels.max())}]"
                )

        device = embeddings.device

        if batch_size == 0:
            return embeddings.sum() * 0.0

        # Always L2-normalize internally so cosine-similarity semantics
        # hold regardless of whether the caller already normalized
        # (idempotent for already-normalized input).
        normalized = F.normalize(embeddings, p=2, dim=1)
        similarity = torch.matmul(normalized, normalized.T) / self.temperature  # [B, B]

        # log-sum-exp stability: subtract each row's max (detached).
        row_max = similarity.max(dim=1, keepdim=True).values.detach()
        logits = similarity - row_max

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=device)
        same_class = labels.unsqueeze(0) == labels.unsqueeze(1)  # [B, B]
        positive_mask = same_class & ~self_mask

        weight = torch.ones((batch_size, batch_size), device=device, dtype=embeddings.dtype)
        if len(self.ambiguity_pairs) > 0:
            labels_list = labels.tolist()
            ambiguous_negative = torch.zeros((batch_size, batch_size), dtype=torch.bool, device=device)
            for i in range(batch_size):
                for j in range(batch_size):
                    if i != j and labels_list[i] != labels_list[j] and self.ambiguity_pairs.contains(labels_list[i], labels_list[j]):
                        ambiguous_negative[i, j] = True
            weight = torch.where(ambiguous_negative, torch.full_like(weight, self.ambiguity_weight), weight)
        weight = weight.masked_fill(self_mask, 0.0)  # exclude self from the denominator entirely

        exp_logits = torch.exp(logits) * weight
        denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12)
        log_prob = logits - torch.log(denom)  # [B, B]

        positive_counts = positive_mask.sum(dim=1)  # [B]
        has_positive = positive_counts > 0

        sum_log_prob_pos = (positive_mask.to(log_prob.dtype) * log_prob).sum(dim=1)
        safe_counts = positive_counts.clamp_min(1).to(log_prob.dtype)
        mean_log_prob_pos = sum_log_prob_pos / safe_counts

        loss_per_anchor = -mean_log_prob_pos

        if bool(has_positive.any()):
            loss = loss_per_anchor[has_positive].mean()
        else:
            # No anchor in this batch has a same-class positive (e.g. one
            # sample per class) - there is no contrastive signal to learn
            # from. Return a finite, gradient-connected zero rather than
            # NaN from an empty mean().
            loss = embeddings.sum() * 0.0

        return loss


def cs_supcon_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = DEFAULT_TEMPERATURE,
    ambiguity_weight: float = DEFAULT_AMBIGUITY_WEIGHT,
    ambiguity_pairs: Optional[AmbiguityPairs] = None,
    num_classes: Optional[int] = None,
) -> torch.Tensor:
    """Functional convenience wrapper — build a CSSupConLoss and call it in
    one step, for one-off/test usage without keeping a persistent module:

        loss = cs_supcon_loss(embeddings, labels, temperature=0.1, ...)
    """
    module = CSSupConLoss(temperature=temperature, ambiguity_weight=ambiguity_weight, ambiguity_pairs=ambiguity_pairs)
    return module(embeddings, labels, num_classes=num_classes)

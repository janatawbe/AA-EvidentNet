"""Evidential Deep Learning (EDL) for AA-EvidentNet -- Task 9.

Implements the Dirichlet-based evidential classification framework of
Sensoy, Kaplan, and Kandemir (2018), "Evidential Deep Learning to Quantify
Classification Uncertainty" (NeurIPS 2018). This is ONE of several
published EDL formulations (others use a different evidence function or a
different Bayes-risk loss variant) -- it is not claimed to be the only
valid or universally optimal choice for this project.

Pipeline (K classes, raw evidential-head output o in R^K):

    evidence  = softplus(o)          >= 0                        [B, K]
    alpha     = evidence + 1         >= 1  (Dirichlet parameters) [B, K]
    S         = sum_k alpha_k        (total Dirichlet "strength") [B]
    p_k       = alpha_k / S          (expected class probability
                                       under Dir(alpha))          [B, K]
    u         = K / S                (vacuity / "not enough
                                       evidence" uncertainty)     [B]

Because evidence >= 0, alpha is always >= 1, so S is always >= K. This
means u = K/S is always in (0, 1] and p always sums to exactly 1, for any
finite raw output -- no division-by-zero, log(0), or digamma-near-0 is
reachable. u -> 1 as evidence -> 0 for every class (maximum uncertainty,
no evidence at all); u -> 0 as evidence -> infinity for at least one class
(the model has accumulated a large amount of supporting evidence).

Loss (Sensoy et al. 2018, Eq. 3-5): the Bayes risk of the cross-entropy
loss under the predicted Dirichlet, i.e. the expectation of the ordinary
cross-entropy loss over class probabilities p ~ Dir(alpha), which has the
closed form (digamma = psi, the derivative of log-Gamma):

    L_i^CE = sum_k y_ik * (psi(S_i) - psi(alpha_ik))

plus an annealed KL-divergence regularizer that shrinks evidence for
INCORRECT classes toward the uniform (evidence-free) Dirichlet Dir(1,...,1)
-- it does not penalize evidence for the correct class:

    alpha_tilde_i = y_i + (1 - y_i) * alpha_i     (correct class reset to 1)
    KL_i          = KL[Dir(alpha_tilde_i) || Dir(1,...,1)]
                  = lgamma(S~_i) - lgamma(K) - sum_k lgamma(alpha_tilde_ik)
                    + sum_k (alpha_tilde_ik - 1) *
                      (psi(alpha_tilde_ik) - psi(S~_i))

    loss_i = L_i^CE + lambda_t * KL_i

lambda_t is annealed linearly from 0 up to kl_weight_max over
kl_annealing_epochs (configs/losses.yaml: edl.kl_annealing_epochs /
edl.kl_weight_max) -- the caller passes the current epoch; this module
never tracks epoch/training state itself. Every alpha_tilde_ik is always
>= 1 (it is either exactly 1, or a copy of alpha_ik which is itself >= 1),
so both lgamma and digamma are evaluated only on inputs >= 1, where they
are smooth and well-behaved -- no epsilon guarding is mathematically
required for this term either.

This module implements ONLY the evidential head + EDL loss. It is not
wired into a combined training objective with cross-entropy or CS-SupCon,
and no training has been run against it -- see README.md and
REPRODUCIBILITY.md for how it is intended to eventually combine with
classification and CS-SupCon losses.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

DEFAULT_LOSS_WEIGHT = 1.0
DEFAULT_KL_ANNEALING_EPOCHS = 10
DEFAULT_KL_WEIGHT_MAX = 1.0
DEFAULT_EPSILON = 1e-8


class EvidentialConfigError(Exception):
    """Raised for an invalid EDL configuration: a non-positive
    kl_annealing_epochs/epsilon, or a negative loss_weight/kl_weight_max.
    Never silently corrected or defaulted."""


@dataclass
class EvidentialOutput:
    """Everything downstream training/evaluation/checkpointing code might
    need from one evidential-head forward pass. `raw_output` (the
    pre-softplus evidential-head output) is kept rather than discarded, in
    case it is useful for debugging or checkpointing."""

    raw_output: torch.Tensor  # [B, K] -- pre-softplus evidential head output
    evidence: torch.Tensor  # [B, K], >= 0
    alpha: torch.Tensor  # [B, K], >= 1 (Dirichlet parameters)
    probabilities: torch.Tensor  # [B, K], sums to 1 along dim=1
    uncertainty: torch.Tensor  # [B], in (0, 1]


def compute_evidential_output(raw_output: torch.Tensor, epsilon: float = DEFAULT_EPSILON) -> EvidentialOutput:
    """Evidence -> alpha -> probability -> uncertainty, from a raw
    evidential-head output of shape [B, K].

    `epsilon` defensively floors the Dirichlet strength S before dividing
    by it. This is mathematically unreachable in ordinary use (S >= K >= 1
    always, since alpha = softplus(raw) + 1 >= 1 for every class), but is
    kept -- consistent with this project's other losses -- as an explicit,
    configurable numerical safety net rather than a bare, unguarded
    division.
    """
    if raw_output.dim() != 2:
        raise ValueError(f"Evidential head output must be [B, K], got {tuple(raw_output.shape)}")
    num_classes = raw_output.size(1)
    if num_classes == 0:
        raise ValueError("Evidential head output must have K > 0 classes")
    if epsilon <= 0:
        raise EvidentialConfigError(f"epsilon must be > 0, got {epsilon}")

    evidence = F.softplus(raw_output)
    alpha = evidence + 1.0
    strength = alpha.sum(dim=1, keepdim=True).clamp_min(epsilon)  # [B, 1]
    probabilities = alpha / strength
    uncertainty = (num_classes / strength).squeeze(1)  # [B]

    return EvidentialOutput(
        raw_output=raw_output,
        evidence=evidence,
        alpha=alpha,
        probabilities=probabilities,
        uncertainty=uncertainty,
    )


class EvidentialHead(nn.Module):
    """A Linear layer mapping a shared embedding to a raw evidential
    output, plus the evidence/alpha/probability/uncertainty pipeline
    above. Deliberately separate from any existing ordinary classification
    head -- both can be attached to the same embedding without interfering
    with each other; neither is removed nor repurposed by the other."""

    def __init__(self, embedding_dim: int, num_classes: int, epsilon: float = DEFAULT_EPSILON):
        super().__init__()
        if embedding_dim <= 0:
            raise ValueError(f"embedding_dim must be > 0, got {embedding_dim}")
        if num_classes <= 0:
            raise ValueError(f"num_classes must be > 0, got {num_classes}")
        self.linear = nn.Linear(embedding_dim, num_classes)
        self.num_classes = num_classes
        self.epsilon = epsilon

    def forward(self, embedding: torch.Tensor) -> EvidentialOutput:
        if embedding.dim() != 2:
            raise ValueError(f"EvidentialHead expects embeddings of shape [B, D], got {tuple(embedding.shape)}")
        raw_output = self.linear(embedding)
        return compute_evidential_output(raw_output, epsilon=self.epsilon)


def _kl_dirichlet_to_uniform(alpha_tilde: torch.Tensor) -> torch.Tensor:
    """KL[Dir(alpha_tilde) || Dir(1,...,1)], closed form (see module
    docstring). `alpha_tilde` must be elementwise >= 1 (guaranteed by
    construction in edl_loss: it is either exactly 1, or a copy of an
    alpha value that is itself >= 1) so lgamma/digamma are always
    evaluated on well-behaved (>= 1) inputs."""
    num_classes = alpha_tilde.size(1)
    strength_tilde = alpha_tilde.sum(dim=1)  # [B]

    log_norm = (
        torch.lgamma(strength_tilde)
        - torch.lgamma(torch.full_like(strength_tilde, float(num_classes)))
        - torch.lgamma(alpha_tilde).sum(dim=1)
    )
    digamma_term = (
        (alpha_tilde - 1.0) * (torch.digamma(alpha_tilde) - torch.digamma(strength_tilde).unsqueeze(1))
    ).sum(dim=1)

    return log_norm + digamma_term


def edl_loss(
    alpha: torch.Tensor,
    labels: torch.Tensor,
    epoch: Optional[float] = None,
    kl_annealing_epochs: int = DEFAULT_KL_ANNEALING_EPOCHS,
    kl_weight_max: float = DEFAULT_KL_WEIGHT_MAX,
) -> torch.Tensor:
    """The Sensoy et al. (2018) EDL loss: expected cross-entropy under
    Dir(alpha), plus an annealed KL regularizer that shrinks evidence for
    incorrect classes toward the uniform Dirichlet. See module docstring
    for the full closed-form derivation. Returns a single scalar (mean
    over the batch).

    `epoch`: the current training epoch, used only to compute the KL
    annealing coefficient `lambda_t = kl_weight_max * min(1, epoch /
    kl_annealing_epochs)`. If None, the regularizer is applied at its full
    weight (`lambda_t = kl_weight_max`) -- this module has no notion of
    "current epoch" of its own and never tracks training state.
    """
    if alpha.dim() != 2:
        raise ValueError(f"EDL loss expects alpha of shape [B, K], got {tuple(alpha.shape)}")
    batch_size, num_classes = alpha.shape
    if num_classes == 0:
        raise ValueError("EDL loss: alpha must have K > 0 classes")
    if bool((alpha < 1.0 - 1e-6).any()):
        raise ValueError(
            "EDL loss: alpha must be >= 1 everywhere (alpha = evidence + 1, "
            "with evidence = softplus(raw_output) >= 0) -- got a value below 1"
        )

    labels = labels.view(-1)
    if labels.numel() != batch_size:
        raise ValueError(f"EDL loss: labels must have shape [B]={batch_size}, got {tuple(labels.shape)}")
    if bool((labels < 0).any()) or bool((labels >= num_classes).any()):
        raise ValueError(
            f"EDL loss: labels must be in [0, {num_classes - 1}], got range "
            f"[{int(labels.min())}, {int(labels.max())}]"
        )
    if kl_annealing_epochs <= 0:
        raise EvidentialConfigError(f"kl_annealing_epochs must be > 0, got {kl_annealing_epochs}")
    if kl_weight_max < 0:
        raise EvidentialConfigError(f"kl_weight_max must be >= 0, got {kl_weight_max}")
    if epoch is not None and epoch < 0:
        raise ValueError(f"EDL loss: epoch must be >= 0, got {epoch}")

    labels_one_hot = F.one_hot(labels, num_classes=num_classes).to(alpha.dtype)

    strength = alpha.sum(dim=1, keepdim=True)  # [B, 1]
    ce_term = (labels_one_hot * (torch.digamma(strength) - torch.digamma(alpha))).sum(dim=1)  # [B]

    if kl_weight_max > 0:
        alpha_tilde = labels_one_hot + (1.0 - labels_one_hot) * alpha
        kl_term = _kl_dirichlet_to_uniform(alpha_tilde)  # [B]
        lambda_t = kl_weight_max if epoch is None else kl_weight_max * min(1.0, float(epoch) / float(kl_annealing_epochs))
        per_sample_loss = ce_term + lambda_t * kl_term
    else:
        per_sample_loss = ce_term

    return per_sample_loss.mean()


@dataclass(frozen=True)
class EDLSettings:
    enabled: bool
    loss_weight: float
    kl_annealing_epochs: int
    kl_weight_max: float
    epsilon: float


def load_edl_settings(losses_config: Dict[str, Any]) -> EDLSettings:
    """Parse+validate configs/losses.yaml: edl into an EDLSettings. Fails
    clearly (EvidentialConfigError) on any invalid value rather than
    silently clamping/defaulting a bad one."""
    section = losses_config.get("edl") or {}

    enabled = bool(section.get("enabled", True))
    loss_weight = float(section.get("loss_weight", DEFAULT_LOSS_WEIGHT))
    kl_annealing_epochs = int(section.get("kl_annealing_epochs", DEFAULT_KL_ANNEALING_EPOCHS))
    kl_weight_max = float(section.get("kl_weight_max", DEFAULT_KL_WEIGHT_MAX))
    epsilon = float(section.get("epsilon", DEFAULT_EPSILON))

    errors = []
    if loss_weight < 0:
        errors.append(f"edl.loss_weight must be >= 0, got {loss_weight}")
    if kl_annealing_epochs <= 0:
        errors.append(f"edl.kl_annealing_epochs must be > 0, got {kl_annealing_epochs}")
    if kl_weight_max < 0:
        errors.append(f"edl.kl_weight_max must be >= 0, got {kl_weight_max}")
    if epsilon <= 0:
        errors.append(f"edl.epsilon must be > 0, got {epsilon}")
    if errors:
        raise EvidentialConfigError("Invalid edl configuration:\n" + "\n".join(f"  - {e}" for e in errors))

    return EDLSettings(
        enabled=enabled,
        loss_weight=loss_weight,
        kl_annealing_epochs=kl_annealing_epochs,
        kl_weight_max=kl_weight_max,
        epsilon=epsilon,
    )


class EDLLoss(nn.Module):
    """nn.Module wrapper around edl_loss, for use alongside CSSupConLoss
    (src/losses/cs_supcon.py) with the same construction pattern."""

    def __init__(
        self,
        kl_annealing_epochs: int = DEFAULT_KL_ANNEALING_EPOCHS,
        kl_weight_max: float = DEFAULT_KL_WEIGHT_MAX,
    ) -> None:
        super().__init__()
        if kl_annealing_epochs <= 0:
            raise EvidentialConfigError(f"kl_annealing_epochs must be > 0, got {kl_annealing_epochs}")
        if kl_weight_max < 0:
            raise EvidentialConfigError(f"kl_weight_max must be >= 0, got {kl_weight_max}")
        self.kl_annealing_epochs = kl_annealing_epochs
        self.kl_weight_max = kl_weight_max

    @classmethod
    def from_settings(cls, settings: EDLSettings) -> "EDLLoss":
        return cls(kl_annealing_epochs=settings.kl_annealing_epochs, kl_weight_max=settings.kl_weight_max)

    def forward(self, alpha: torch.Tensor, labels: torch.Tensor, epoch: Optional[float] = None) -> torch.Tensor:
        return edl_loss(
            alpha,
            labels,
            epoch=epoch,
            kl_annealing_epochs=self.kl_annealing_epochs,
            kl_weight_max=self.kl_weight_max,
        )

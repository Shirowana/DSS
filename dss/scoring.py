from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


DSSScoreMethod = str

MEAN_ABS: Final[DSSScoreMethod] = "mean_abs"
ABS_MEAN: Final[DSSScoreMethod] = "abs_mean"
MEAN_SQUARE: Final[DSSScoreMethod] = "mean_square"
RMS_OVER_PARAM: Final[DSSScoreMethod] = "rms_over_param"
ABS_MEAN_OVER_PARAM: Final[DSSScoreMethod] = "abs_mean_over_param"
SNR: Final[DSSScoreMethod] = "snr"
NEWTON_LIKE: Final[DSSScoreMethod] = "newton_like"

VALID_SCORE_METHODS: Final[tuple[DSSScoreMethod, ...]] = (
    MEAN_ABS,
    ABS_MEAN,
    MEAN_SQUARE,
    RMS_OVER_PARAM,
    ABS_MEAN_OVER_PARAM,
    SNR,
    NEWTON_LIKE,
)


@dataclass
class ScoreStats:
    """Statistics accumulated over one DSS refresh window.

    The layer stores candidate gradients across `n` optimizer steps and exposes
    them here as sample-average ingredients:

    - `grad_sum = Σ g`
    - `abs_grad_sum = Σ |g|`
    - `grad_sq_sum = Σ g²`
    - `count = n`
    """

    grad_sum: torch.Tensor | None
    abs_grad_sum: torch.Tensor | None
    grad_sq_sum: torch.Tensor | None
    count: int


def required_stats_for(method: DSSScoreMethod) -> frozenset[str]:
    """Return which cached statistics must be maintained for a score method."""

    if method == MEAN_ABS:
        return frozenset({"abs_grad_sum"})
    if method == ABS_MEAN:
        return frozenset({"grad_sum"})
    if method == MEAN_SQUARE:
        return frozenset({"grad_sq_sum"})
    if method == RMS_OVER_PARAM:
        return frozenset({"grad_sq_sum", "theta0_abs"})
    if method == ABS_MEAN_OVER_PARAM:
        return frozenset({"grad_sum", "theta0_abs"})
    if method == SNR:
        return frozenset({"grad_sum", "grad_sq_sum"})
    if method == NEWTON_LIKE:
        return frozenset({"grad_sum", "grad_sq_sum"})
    raise ValueError(f"Unsupported DSS score method: {method!r}")


def _empty_like_reference(stats: ScoreStats, theta0_abs: torch.Tensor | None) -> torch.Tensor:
    for tensor in (stats.grad_sum, stats.abs_grad_sum, stats.grad_sq_sum, theta0_abs):
        if tensor is not None:
            return torch.empty(0, device=tensor.device, dtype=torch.float32)
    return torch.empty(0, dtype=torch.float32)


def _ensure_method(method: DSSScoreMethod) -> None:
    if method not in VALID_SCORE_METHODS:
        raise ValueError(f"Unsupported DSS score method: {method!r}")


def compute_score(
    method: DSSScoreMethod,
    stats: ScoreStats,
    theta0_abs: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute one of the seven single-position importance scores.

    Method notes:

    - `mean_abs`: S^(1) = E[|g|]
      Prefers positions with large average gradient magnitude, even if the sign
      changes across steps. This is the natural counterpoint to `abs_mean`.

    - `abs_mean`: S^(2) = |E[g]|
      Prefers positions with a stable net push in one direction. This is the
      current DSS scoring rule.

    - `mean_square`: S^(3) = E[g^2]
      Fisher / second-moment style baseline. Usually more stable because sign
      cancellation does not reduce the score.

    - `rms_over_param`: S^(4) = sqrt(E[g^2]) / (|θ^(0)| + ε)
      Rewards strong gradients on positions with small initial parameter
      magnitude. Similar in spirit to a second-moment GaLLoP variant.

    - `abs_mean_over_param`: S^(5) = |E[g]| / (|θ^(0)| + ε)
      Combines DSS-style directional stability with GaLLoP-like parameter
      normalization.

    - `snr`: S^(6) = |E[g]| / (std(g) + ε)
      Signal-to-noise ratio. Prefers positions with a large stable push and low
      temporal variance; useful when candidate sampling is noisy.

    - `newton_like`: S^(7) = |E[g]| / (E[g^2] + ε)
      A diagonal Newton / inverse-Fisher style one-dimensional approximation.
      Often more sensitive to the choice of ε and overall gradient scale.
    """

    _ensure_method(method)

    if stats.count <= 0:
        return _empty_like_reference(stats, theta0_abs)

    count = float(stats.count)

    if method == MEAN_ABS:
        if stats.abs_grad_sum is None or stats.abs_grad_sum.numel() == 0:
            return _empty_like_reference(stats, theta0_abs)
        return (stats.abs_grad_sum / count).float()

    if method == ABS_MEAN:
        if stats.grad_sum is None or stats.grad_sum.numel() == 0:
            return _empty_like_reference(stats, theta0_abs)
        return (stats.grad_sum / count).abs().float()

    if stats.grad_sq_sum is None or stats.grad_sq_sum.numel() == 0:
        return _empty_like_reference(stats, theta0_abs)

    mean_square = (stats.grad_sq_sum / count).float()

    if method == MEAN_SQUARE:
        return mean_square

    if theta0_abs is None or theta0_abs.numel() == 0:
        return _empty_like_reference(stats, theta0_abs)
    denom_param = theta0_abs.float() + eps

    if method == RMS_OVER_PARAM:
        return torch.sqrt(mean_square) / denom_param

    if stats.grad_sum is None or stats.grad_sum.numel() == 0:
        return _empty_like_reference(stats, theta0_abs)

    abs_mean = (stats.grad_sum / count).abs().float()

    if method == ABS_MEAN_OVER_PARAM:
        return abs_mean / denom_param

    if method == SNR:
        variance = torch.clamp(mean_square - abs_mean.square(), min=0.0)
        std = torch.sqrt(variance)
        return abs_mean / (std + eps)

    return abs_mean / (mean_square + eps)

"""Shared local-rank coordinate computation for VVBP models and evaluation.

All models that operate on 3×3 VVBP patches need to compute local-rank
coordinates q ∈ [0,1] for each center-pixel value relative to its 3×3
neighbourhood. This module provides the single implementation.
"""

from __future__ import annotations

import math

import torch


def compute_local_rank(values_sorted: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local-rank coordinates q and center values.

    Args:
        values_sorted: [B, J, K] — sorted VVBP values per local position.
                        J = patch_size² (9 for 3×3), K = number of views.

    Returns:
        q_local: [B, K] — local-rank coordinates in [0, 1].
        center_values: [B, K] — raw center-pixel VVBP values.
    """
    B, J, K = values_sorted.shape
    center_idx = J // 2
    L = J * K

    center_values = values_sorted[:, center_idx, :].contiguous()  # [B, K]

    merged_values = values_sorted.reshape(B, L).contiguous()
    merged_sorted = torch.sort(merged_values, dim=-1).values      # [B, L]

    left = torch.searchsorted(merged_sorted, center_values, right=False)
    right = torch.searchsorted(merged_sorted, center_values, right=True) - 1
    left = left.clamp(min=0, max=L - 1)
    right = right.clamp(min=0, max=L - 1)

    local_rank = 0.5 * (left.float() + right.float())              # [B, K]
    q_local = local_rank / float(L - 1)                             # [B, K]

    return q_local, center_values


def compute_local_rank_sorted(
    values_sorted: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute local-rank coordinates, then sort center values by q.

    Returns:
        q_sorted: [B, K] — q coordinates sorted ascending.
        center_sorted: [B, K] — center values sorted by their q.
    """
    q_local, center_values = compute_local_rank(values_sorted)
    q_sorted, order = torch.sort(q_local, dim=-1)
    center_sorted = torch.gather(center_values, dim=-1, index=order)
    return q_sorted, center_sorted


def compute_local_rank_closed_integral(values_sorted: torch.Tensor) -> torch.Tensor:
    """Non-parametric baseline: closed-interval trapezoid integral over q.

    Uses local-rank coordinates with closed endpoints (q=0, v=v_min) and
    (q=1, v=v_max), then integrates via non-uniform trapezoid rule:
        pred = π · Σ ½ Δq (vᵢ + vᵢ₋₁)

    Args:
        values_sorted: [N, J, K] — sorted VVBP values per local position.

    Returns:
        [N] prediction.
    """
    q_sorted, center_sorted = compute_local_rank_sorted(values_sorted)

    N = values_sorted.shape[0]
    q0 = torch.zeros(N, 1, device=values_sorted.device, dtype=q_sorted.dtype)
    q1 = torch.ones(N, 1, device=values_sorted.device, dtype=q_sorted.dtype)
    v0 = center_sorted[:, :1]
    v1 = center_sorted[:, -1:]

    q_ext = torch.cat([q0, q_sorted, q1], dim=-1)      # [N, K+2]
    v_ext = torch.cat([v0, center_sorted, v1], dim=-1)  # [N, K+2]

    dq = q_ext[:, 1:] - q_ext[:, :-1]                    # [N, K+1]
    v_sum = v_ext[:, :-1] + v_ext[:, 1:]                 # [N, K+1]
    pred = math.pi * (0.5 * dq * v_sum).sum(dim=-1)      # [N]

    return pred

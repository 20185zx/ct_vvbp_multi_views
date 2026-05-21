"""Feature builders for cached sorted VVBP values."""

from __future__ import annotations

import torch

from .local_vvbp import get_relative_coords


def make_model_features_from_values(
    values_sorted: torch.Tensor,
    stats: dict,
    use_coord: bool = True,
    patch_size: int = 3,
    use_tail_encoding: bool = False,
) -> torch.Tensor:
    """Build neural-network input features from cached sorted VVBP values.

    Args:
        values_sorted: Tensor with shape ``[N, J, K]``.
        stats: Dictionary containing ``v_mean`` and ``v_std``.
        use_coord: Whether to append local patch coordinates ``[du, dv]``.
        patch_size: Local patch size. For example, ``3`` means ``J=9``.
        use_tail_encoding: Whether to append ``|2r - 1|`` as a tail feature.

    Returns:
        If ``use_coord`` is False: ``[N, J, K, 2 or 3]``.
        If ``use_coord`` is True: ``[N, J, K, 4 or 5]``.
    """
    device = values_sorted.device
    n_points, n_patch, n_views = values_sorted.shape

    v_mean = stats["v_mean"].to(device)
    v_std = stats["v_std"].to(device)
    value_norm = (values_sorted - v_mean) / v_std

    rank = torch.linspace(0.0, 1.0, n_views, device=device)
    rank = rank.view(1, 1, n_views).expand(n_points, n_patch, n_views)

    rank_features = [rank.unsqueeze(-1)]
    if use_tail_encoding:
        tail = torch.abs(2.0 * rank - 1.0)
        rank_features.append(tail.unsqueeze(-1))

    value_feature = value_norm.unsqueeze(-1)

    if not use_coord:
        return torch.cat(rank_features + [value_feature], dim=-1)

    rel = get_relative_coords(patch_size=patch_size, device=device)
    rel = rel.view(1, n_patch, 1, 2).expand(n_points, n_patch, n_views, 2)
    return torch.cat([rel] + rank_features + [value_feature], dim=-1)

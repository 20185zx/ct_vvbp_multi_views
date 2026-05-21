import torch
import torch.nn as nn
import math

from src.data.local_rank import compute_local_rank_sorted


class LocalRankCenterIntegralMLPNet(nn.Module):
    """
    Local-rank center VVBP integral MLP.

    Difference from LocalRankCenterMLPNet:
        - LocalRankCenterMLPNet uses average pooling over K tokens.
        - This model uses normalized non-uniform trapezoid weights
          based on local-rank coordinates.

    Input:
        values_sorted: [B, J, K]  (J=9 for 3x3 patch, K=sparse_views)

    Token:
        e_k = [q_k^local, normalized_center_value_k]

    Aggregation:
        z = sum_k omega_k^local * phi_a(e_k)
        sum_k omega_k^local = 1
    """

    def __init__(
        self,
        point_hidden=64,
        point_dim=64,
        out_hidden=128,
        dropout=0.0,
    ):
        super().__init__()

        self.use_coord = False
        self.input_mode = "values_sorted"

        self.point_mlp = nn.Sequential(
            nn.Linear(2, point_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(point_hidden, point_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(point_hidden, point_dim),
            nn.ReLU(inplace=True),
        )

        self.out_mlp = nn.Sequential(
            nn.Linear(point_dim, out_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_hidden, out_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(out_hidden, 1),
        )

    @staticmethod
    def _nonuniform_trapezoid_weights(q_sorted):
        """
        Compute point-wise non-uniform trapezoid weights.

        q_sorted: [B, K], sorted local-rank coordinates.

        Return:
            point_w: [B, K], normalized so that sum_k point_w = 1.
        """
        B, K = q_sorted.shape

        if K < 2:
            return torch.ones_like(q_sorted)

        dq = q_sorted[:, 1:] - q_sorted[:, :-1]  # [B, K-1]
        dq = dq.clamp_min(0.0)

        point_w = torch.zeros_like(q_sorted)

        # Left endpoint: 0.5 * (q_2 - q_1)
        point_w[:, :-1] += 0.5 * dq

        # Right endpoint: 0.5 * (q_K - q_{K-1})
        point_w[:, 1:] += 0.5 * dq

        # Normalize weight sum to 1.
        # This corresponds to the version mentioned by your senior:
        # "把权重总和 / 1 做归一化".
        point_w = point_w / (point_w.sum(dim=-1, keepdim=True) + 1e-12)

        return point_w

    def forward(self, values_sorted, stats):
        """
        values_sorted: [B, J, K]  (J=9 for 3x3 patch, K=sparse_views)
        stats: dict with v_mean, v_std
        """
        _, J, _ = values_sorted.shape

        if J != 9:
            raise ValueError(f"Expected 3x3 patch, J=9, but got J={J}")

        q_sorted, center_sorted = compute_local_rank_sorted(values_sorted)

        # Normalize VVBP values.
        v_mean = stats["v_mean"].to(values_sorted.device)
        v_std = stats["v_std"].to(values_sorted.device)
        value_norm = (center_sorted - v_mean) / v_std  # [B, K]

        # Tokens: [q_local, normalized_value]
        tokens = torch.stack([q_sorted, value_norm], dim=-1)  # [B, K, 2]

        # Point-wise encoding.
        h = self.point_mlp(tokens)  # [B, K, C]

        # Normalized non-uniform trapezoid weights.
        point_w = self._nonuniform_trapezoid_weights(q_sorted)  # [B, K]
        point_w = point_w.unsqueeze(-1)                         # [B, K, 1]

        # Integral-style aggregation.
        pooled = math.pi * (h * point_w).sum(dim=1)  # [B, C]

        return self.out_mlp(pooled)
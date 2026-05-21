import torch
import torch.nn as nn

from src.data.local_rank import compute_local_rank_sorted


class LocalRankCenterClosedMLPNet(nn.Module):
    """
    Local-rank center VVBP closed-interval MLP.

    This is the learnable counterpart of:
        local-rank center integral closed

    Difference from LocalRankCenterIntegralMLPNet:
        - IntegralMLP uses only the center points and normalizes weights to sum to 1.
        - ClosedMLP adds two boundary tokens:
              (q=0, v=v_min), (q=1, v=v_max)
          and then applies non-uniform trapezoid pooling on the closed interval [0,1].

    Input:
        values_sorted: [B, J, K]  (J=9 for 3x3 patch, K=sparse_views)

    Token:
        e_i = [q_i, normalized_value_i]

    Aggregation:
        z = sum_i omega_i^closed * phi_a(e_i)
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
    def _closed_trapezoid_weights(q_ext):
        """
        Compute non-uniform trapezoid weights on a closed interval.

        q_ext: [B, K+2]
            sorted coordinates including q=0 and q=1.

        Return:
            point_w: [B, K+2]
            The weights should sum approximately to 1 because q_ext spans [0,1].
        """
        B, M = q_ext.shape

        if M < 2:
            return torch.ones_like(q_ext)

        dq = q_ext[:, 1:] - q_ext[:, :-1]  # [B, M-1]
        dq = dq.clamp_min(0.0)

        point_w = torch.zeros_like(q_ext)

        point_w[:, :-1] += 0.5 * dq
        point_w[:, 1:] += 0.5 * dq

        # In theory, since q_ext starts from 0 and ends at 1,
        # point_w.sum(dim=-1) should be 1.
        # This normalization is only for numerical safety.
        point_w = point_w / (point_w.sum(dim=-1, keepdim=True) + 1e-12)

        return point_w

    def forward(self, values_sorted, stats):
        """
        values_sorted: [B, J, K], J=9 (3x3 patch), K=sparse_views
        stats: dict with v_mean, v_std
        """
        B, J, K = values_sorted.shape

        if J != 9:
            raise ValueError(f"Expected 3x3 patch, J=9, but got J={J}")

        q_sorted, center_sorted = compute_local_rank_sorted(values_sorted)

        # Closed interval:
        # Add (q=0, v=v_min) and (q=1, v=v_max).
        q0 = torch.zeros(B, 1, device=values_sorted.device, dtype=q_sorted.dtype)
        q1 = torch.ones(B, 1, device=values_sorted.device, dtype=q_sorted.dtype)

        v0 = center_sorted[:, :1]
        v1 = center_sorted[:, -1:]

        q_ext = torch.cat([q0, q_sorted, q1], dim=-1)          # [B, K+2]
        value_ext = torch.cat([v0, center_sorted, v1], dim=-1) # [B, K+2]

        # Normalize VVBP values.
        v_mean = stats["v_mean"].to(values_sorted.device)
        v_std = stats["v_std"].to(values_sorted.device)
        value_norm = (value_ext - v_mean) / v_std             # [B, K+2]

        # Tokens: [q, normalized_value]
        tokens = torch.stack([q_ext, value_norm], dim=-1)     # [B, K+2, 2]

        # Point-wise encoding.
        h = self.point_mlp(tokens)                            # [B, K+2, C]

        # Closed interval non-uniform trapezoid weights.
        point_w = self._closed_trapezoid_weights(q_ext)        # [B, K+2]
        point_w = point_w.unsqueeze(-1)                        # [B, K+2, 1]

        # Integral-style aggregation on [0,1].
        pooled = (h * point_w).sum(dim=1)                      # [B, C]

        return self.out_mlp(pooled)
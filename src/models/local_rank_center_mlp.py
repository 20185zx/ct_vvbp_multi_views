import torch
import torch.nn as nn

from src.data.local_rank import compute_local_rank


class LocalRankCenterMLPNet(nn.Module):
    """
    Local-rank center VVBP MLP.

    Input:
        values_sorted: [B, J, K]  (J=9 for 3x3 patch, K=sparse_views)

    For each center pixel:
        1. take center K VVBP values;
        2. compute their rank coordinates in the merged 3x3×K distribution;
        3. token = [q_local, normalized_center_value];
        4. point MLP -> average pooling -> output MLP.
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

    def forward(self, values_sorted, stats):
        """
        values_sorted: [B, J, K], J=9 (3x3 patch), K=sparse_views
        stats: dict with v_mean, v_std
        """
        B, J, K = values_sorted.shape

        if J != 9:
            raise ValueError(f"Expected 3x3 patch, J=9, but got J={J}")

        q_local, center_values = compute_local_rank(values_sorted)

        v_mean = stats["v_mean"].to(values_sorted.device)
        v_std = stats["v_std"].to(values_sorted.device)
        value_norm = (center_values - v_mean) / v_std  # [B, K]

        tokens = torch.stack([q_local, value_norm], dim=-1)  # [B, K, 2]

        h = self.point_mlp(tokens)  # [B, K, C]
        pooled = h.mean(dim=1)      # [B, C]

        return self.out_mlp(pooled)
import torch
import torch.nn as nn



def trapezoid_rank_weights(length, device):
    """Return normalized trapezoid-rule weights on a rank grid."""
    if length < 2:
        return torch.ones(length, device=device)
    weights = torch.ones(length, device=device) / float(length - 1)
    weights[0] *= 0.5
    weights[-1] *= 0.5
    return weights


class MergedLocalSortedVVBPNet(nn.Module):
    """
    3x3 merged sorted VVBP integral model.

    Input features:
        features: [B, J, K, 2]
        last dim = [rank, value_norm]

    Here:
        J = 9 for 3x3 patch
        K = number of sparse views (dynamic)

    Main idea:
        1. Take all J × K normalized VVBP values.
        2. Merge them into one set of length L = J × K.
        3. Sort again.
        4. Treat it as one rank-domain curve.
        5. Apply integral-style neural aggregation.
    """

    def __init__(
        self,
        point_hidden=64,
        point_dim=64,
        out_hidden=128,
        dropout=0.0,
    ):
        super().__init__()

        # Important:
        # use_coord = False means feature_builder returns [rank, value_norm],
        # not [du, dv, rank, value_norm].
        self.use_coord = False

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

    def forward(self, features):
        """
        features: [B, J, K, 2]
            features[..., 0] = original rank within each local curve
            features[..., 1] = normalized sorted VVBP value

        Return:
            pred: [B, 1]
        """
        B, J, K, D = features.shape
        if D != 2:
            raise ValueError(f"MergedLocalSortedVVBPNet expects input dim=2, got {D}")

        # Extract normalized VVBP values.
        # [B, J, K]
        value_norm = features[..., 1]

        # Merge 3x3 local values into one long curve.
        # [B, J*K]
        merged_values = value_norm.reshape(B, J * K)

        # Re-sort all J*K values together.
        merged_sorted = torch.sort(merged_values, dim=-1).values

        L = J * K

        # New merged rank coordinate q_l in [0, 1].
        # [B, L]
        rank = torch.linspace(
            0.0,
            1.0,
            L,
            device=features.device,
            dtype=features.dtype,
        ).view(1, L).expand(B, L)

        # Token: [q_l, merged_sorted_value_l]
        # [B, L, 2]
        tokens = torch.stack([rank, merged_sorted], dim=-1)

        # Point-wise encoding.
        # [B, L, C]
        h = self.point_mlp(tokens)

        # Integral-style pooling along merged rank dimension.
        # [1, L, 1]
        w = trapezoid_rank_weights(L, device=features.device).to(features.dtype)
        w = w.view(1, L, 1)

        # [B, C]
        pooled = (h * w).sum(dim=1)

        # [B, 1]
        return self.out_mlp(pooled)
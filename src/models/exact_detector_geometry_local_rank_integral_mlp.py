"""Exact-detector geometry-aware local-rank integral MLP.

Extends the original LocalRankCenterIntegralMLPNet by adding two per-view
tokens computed from exact fan-beam detector coordinates:

    G_{p,v}   — local detector-direction slope  (VBBP gradient w.r.t. detector index)
    R2_{p,v}  — local linear-fit reliability     ∈ [0, 1]

Token per view:  [u,  Q_norm,  G_norm,  R2]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.data.local_rank import compute_local_rank


class ExactDetectorGeometryLocalRankIntegralMLPNet(nn.Module):
    """Local-rank centre integral MLP with exact-detector geometry tokens.

    Input:
        values_sorted: [B, J, K]   (J=9 for 3×3 patch, K = sparse_views)
        deltaI_patch:  [B, J, K]   detector-index offsets  (centre row is zero)

    Token per view:
        e_k = [q_k, centre_value_norm, G_norm, R2]

    Integral aggregation: non-uniform trapezoid weights over q.
    """

    def __init__(
        self,
        point_hidden: int = 64,
        point_dim: int = 64,
        out_hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_coord = False
        self.input_mode = "values_sorted"

        # 4 input features: [q, value_norm, G_norm, R2]
        self.point_mlp = nn.Sequential(
            nn.Linear(4, point_hidden),
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
    def _nonuniform_trapezoid_weights(q_sorted: torch.Tensor) -> torch.Tensor:
        B, K = q_sorted.shape
        if K < 2:
            return torch.ones_like(q_sorted)
        dq = (q_sorted[:, 1:] - q_sorted[:, :-1]).clamp_min(0.0)
        point_w = torch.zeros_like(q_sorted)
        point_w[:, :-1] += 0.5 * dq
        point_w[:, 1:] += 0.5 * dq
        point_w = point_w / (point_w.sum(dim=-1, keepdim=True) + 1e-12)
        return point_w

    # ------------------------------------------------------------------
    # G  /  R²  computation from values_sorted + deltaI
    # ------------------------------------------------------------------
    @staticmethod
    def compute_G_R2_from_raw(
        raw_patch: torch.Tensor,
        deltaI_patch: torch.Tensor,
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute G and R² from RAW (unsorted, per-view) patch VVBP.

        ``raw_patch[b, p, j, v]`` = VVBP at view *v*, patch position *j*, pixel *p*.
        This preserves view correspondence, unlike ``values_sorted`` which sorts
        each position independently.

        Through-origin OLS per view:

            G[v] = Σⱼ D[j,v]·Y[j,v] / (Σⱼ D[j,v]² + ε)
            R²[v] = 1 − SS_res / SS_tot

        where Y[j,v] = raw[j,v] − raw[centre,v] and D[j,v] = ΔI[j,v].
        SS_tot = Σⱼ Y[j,v]²  (uncentered — null model is Y=0, not Y=mean(Y)).

        Returns:
            G:  [B, K]   local slope.
            R2: [B, K]   ∈ [0,1].
        """
        B, J, K = raw_patch.shape
        centre_idx = J // 2

        Q_centre = raw_patch[:, centre_idx, :]          # [B, K]
        Y = raw_patch - Q_centre[:, None, :]             # [B, J, K]
        D = deltaI_patch                                 # [B, J, K]

        # Through-origin OLS
        D_sq = (D * D).sum(dim=1)                        # [B, K]
        DY = (D * Y).sum(dim=1)                           # [B, K]
        G = DY / (D_sq + eps)                            # [B, K]

        # R²  (uncentered SS_tot for through-origin regression)
        Y_hat = D * G[:, None, :]                        # [B, J, K]
        SS_res = ((Y - Y_hat) ** 2).sum(dim=1)          # [B, K]
        SS_tot = (Y * Y).sum(dim=1) + eps               # [B, K]  uncentered
        R2 = 1.0 - SS_res / SS_tot
        R2 = R2.clamp(0.0, 1.0)

        return G, R2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, values_sorted: torch.Tensor, stats: dict):
        """Forward pass.

        ``stats`` must contain:

            v_mean, v_std     — VVBP value normalisation
            G_mean, G_std     — G normalisation
            G                 — [B, K] pre-computed G  (from trainer / eval)
            R2                — [B, K] pre-computed R²

        G and R2 are computed EXTERNALLY from raw (unsorted) per-view VVBP
        + exact detector ΔI and passed through stats.  The model only uses them
        as tokens — it does not compute them from ``values_sorted``.

        Returns:
            [B, 1] prediction.
        """
        B, J, K = values_sorted.shape
        if J != 9:
            raise ValueError(f"Expected 3×3 patch (J=9), got J={J}")

        # ---- Geometry tokens: G, R2 (pre-computed, from stats) ----
        G = stats.get("G", None)
        R2 = stats.get("R2", None)
        if G is not None and R2 is not None:
            G = G.to(values_sorted.device)
            R2 = R2.to(values_sorted.device)
            if G.ndim != 2 or G.shape != (B, K):
                raise ValueError(
                    f"stats['G'] shape {tuple(G.shape)}, expected ({B}, {K})"
                )
            G_mean = stats["G_mean"].to(values_sorted.device)
            G_std = stats["G_std"].to(values_sorted.device)
            G_norm = (G - G_mean) / (G_std + 1e-8)
        else:
            # Fallback: zero padding
            G_norm = torch.zeros(B, K, device=values_sorted.device, dtype=values_sorted.dtype)
            R2 = torch.zeros(B, K, device=values_sorted.device, dtype=values_sorted.dtype)

        # ---- Local-rank tokens ----
        # ``values_sorted`` has each position's views sorted by VVBP value.
        # ``G`` / ``R2`` arrive pre-aligned to the centre row's ascending-value
        # order (trainer reorders via raw_centre.argsort).  ``compute_local_rank``
        # returns centre values in the same ascending-value order as
        # values_sorted[:, centre, :], so q, centre, G, R2 share one ordering.
        q_local, centre_raw = compute_local_rank(values_sorted)  # [B, K] each

        # q_local is (nearly) monotonic from sorted centre values — we still
        # sort explicitly in case of ties, then reorder all channels together.
        q_sorted, order = torch.sort(q_local, dim=-1)
        centre_sorted = torch.gather(centre_raw, dim=-1, index=order)
        G_sorted = torch.gather(G_norm, dim=-1, index=order)
        R2_sorted = torch.gather(R2, dim=-1, index=order)

        # --- Debug: print raw centre sort order & verify G/R2 alignment ---
        if not hasattr(self, "_sort_debug_done"):
            self._sort_debug_done = True
            with torch.no_grad():
                p = 0
                # centre_raw is in ascending-value order (same as values_sorted centre).
                # Show that order is identity (or nearly).
                print("[DEBUG align] pixel 0, K=%d" % K)
                print("  centre_raw[:8]:", centre_raw[p, :8].cpu().tolist())
                print("  q_sorted  [:8]:", q_sorted[p, :8].cpu().tolist())
                print("  sort_order[:8]:", order[p, :8].cpu().tolist())
                # Verify: centre_sorted[k] == centre_raw[order[k]]
                c_check = torch.gather(centre_raw[p], dim=0, index=order[p])
                cs = centre_sorted[p]
                print("  centre reorder check (max |diff|): %.2e (expect 0)"
                      % (c_check - cs).abs().max().item())
                G_check = torch.gather(G_norm[p], dim=0, index=order[p])
                print("  G reorder check  (max |diff|): %.2e (expect 0)"
                      % (G_check - G_sorted[p]).abs().max().item())
                R2_check = torch.gather(R2[p], dim=0, index=order[p])
                print("  R2 reorder check (max |diff|): %.2e (expect 0)"
                      % (R2_check - R2_sorted[p]).abs().max().item())
                print("  G_sorted[:5]:", G_sorted[p, :5].cpu().tolist())
                print("  R2_sorted[:5]:", R2_sorted[p, :5].cpu().tolist())

        v_mean = stats["v_mean"].to(values_sorted.device)
        v_std = stats["v_std"].to(values_sorted.device)
        value_norm = (centre_sorted - v_mean) / v_std

        # ---- Build per-view tokens  [B, K, 4] ----
        tokens = torch.stack([q_sorted, value_norm, G_sorted, R2_sorted], dim=-1)

        # ---- Point-wise encoding ----
        h = self.point_mlp(tokens)  # [B, K, C]

        # ---- Non-uniform trapezoid weights ----
        point_w = self._nonuniform_trapezoid_weights(q_sorted).unsqueeze(-1)

        # ---- Integral-style aggregation ----
        pooled = math.pi * (h * point_w).sum(dim=1)  # [B, C]

        return self.out_mlp(pooled)

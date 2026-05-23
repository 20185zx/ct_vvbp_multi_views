"""Exact-detector geometry GATED local-rank integral MLP.

Preserves the original [u, Q] two-channel token branch unchanged.
Adds a multiplicative per-view gate derived from abs(G) * R2:

    s_v    = |G_v| * R2_v                     (reliable HF strength)
    gate_v = sigmoid(a * s_norm + b)                              (∈ (0,1))
    mod_v  = 1 + lambda_geo * (2 * gate_v − 1)                    (∈ (1−λ, 1+λ))
    h_v    ← h_v * mod_v         (centred: >0.5 enhance, <0.5 suppress)

G and R2 come from exact fan-beam detector ΔI (through-origin OLS).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn

from src.data.local_rank import compute_local_rank


class ExactDetectorGeometryGatedLocalRankIntegralMLPNet(nn.Module):
    """Local-rank integral MLP with geometry-feature multiplicative gate.

    Token branch:  e_v = [q_local_v, Q_norm_v]  →  h_v = phi(e_v)
    Gate:          g_v = sigmoid(a * s_norm_v + b)
    Modulation:    h_v ← h_v * (1 + lambda_geo * g_v)

    ``s_v = |G_v| * R2_v`` is the reliable geometric high-frequency strength
    at view v.  The gate suppresses noisy views and amplifies informative ones.
    """

    def __init__(
        self,
        point_hidden: int = 64,
        point_dim: int = 64,
        out_hidden: int = 128,
        dropout: float = 0.0,
        lambda_geo: float = 0.05,
        gate_a_init: float = 1.0,
        gate_b_init: float = 0.0,
        learnable_gate_affine: bool = True,
    ):
        super().__init__()
        self.use_coord = False
        self.input_mode = "values_sorted"
        self.lambda_geo = float(lambda_geo)

        # Same token branch as original  [u, Q] → 2 input dims
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

        # Learnable gate affine parameters
        if learnable_gate_affine:
            self.gate_a = nn.Parameter(torch.tensor(float(gate_a_init)))
            self.gate_b = nn.Parameter(torch.tensor(float(gate_b_init)))
        else:
            self.register_buffer("gate_a", torch.tensor(float(gate_a_init)))
            self.register_buffer("gate_b", torch.tensor(float(gate_b_init)))
        self._learnable_gate = learnable_gate_affine

    @staticmethod
    def _nonuniform_trapezoid_weights(q_sorted: torch.Tensor) -> torch.Tensor:
        B, K = q_sorted.shape
        if K < 2:
            return torch.ones_like(q_sorted).unsqueeze(-1)
        dq = (q_sorted[:, 1:] - q_sorted[:, :-1]).clamp_min(0.0)
        point_w = torch.zeros_like(q_sorted)
        point_w[:, :-1] += 0.5 * dq
        point_w[:, 1:] += 0.5 * dq
        point_w = point_w / (point_w.sum(dim=-1, keepdim=True) + 1e-12)
        return point_w.unsqueeze(-1)

    def forward(self, values_sorted: torch.Tensor, stats: dict):
        B, J, K = values_sorted.shape
        if J != 9:
            raise ValueError(f"Expected 3×3 patch (J=9), got J={J}")

        # ---- Local-rank tokens (same as original) ----
        q_local, centre_raw = compute_local_rank(values_sorted)

        # Sort by q — reorder centre values + G/R2 together
        q_sorted, order = torch.sort(q_local, dim=-1)
        centre_sorted = torch.gather(centre_raw, dim=-1, index=order)

        v_mean = stats["v_mean"].to(values_sorted.device)
        v_std = stats["v_std"].to(values_sorted.device)
        value_norm = (centre_sorted - v_mean) / v_std

        tokens = torch.stack([q_sorted, value_norm], dim=-1)  # [B, K, 2]
        h = self.point_mlp(tokens)                            # [B, K, C]

        # ---- Geometry gate ----
        G = stats.get("G", None)
        R2 = stats.get("R2", None)
        if G is not None and R2 is not None and self.lambda_geo > 0:
            G = G.to(values_sorted.device)
            R2 = R2.to(values_sorted.device)
            if G.ndim != 2 or G.shape != (B, K):
                raise ValueError(
                    f"stats['G'] shape {tuple(G.shape)}, expected ({B}, {K})"
                )
            # Reorder G/R2 by same sort index
            G_sorted = torch.gather(G, dim=-1, index=order)
            R2_sorted = torch.gather(R2, dim=-1, index=order)

            # Compute gate feature  s = |G| * R2
            s = G_sorted.abs() * R2_sorted                     # [B, K]

            # Normalise
            s_mean = stats.get("s_mean", s.mean())
            s_std = stats.get("s_std", s.std().clamp_min(1e-8))
            if isinstance(s_mean, torch.Tensor):
                s_mean = s_mean.to(values_sorted.device)
            if isinstance(s_std, torch.Tensor):
                s_std = s_std.to(values_sorted.device)
            s_norm = (s - s_mean) / (s_std + 1e-8)

            # Gate  g = sigmoid(a * s_norm + b)
            gate = torch.sigmoid(self.gate_a * s_norm + self.gate_b)  # [B, K]

            # Centered multiplicative modulation:
            #   gate ∈ (0,1) → modulation = 1 + λ·(2·gate − 1) ∈ (1−λ, 1+λ)
            #   gate > 0.5 → enhance,  gate < 0.5 → suppress,  gate = 0.5 → unchanged
            modulation = 1.0 + self.lambda_geo * (2.0 * gate - 1.0)
            h = h * modulation.unsqueeze(-1)
        else:
            gate = None

        # ---- Per-batch gate stats for epoch logging ----
        if gate is not None:
            if not hasattr(self, "_gate_epoch"):
                self._gate_epoch = {"s": [], "gate": []}
            self._gate_epoch["s"].append(float(s.mean().cpu()))
            self._gate_epoch["gate"].append((float(gate.mean().cpu()),
                                              float(gate.min().cpu()),
                                              float(gate.max().cpu())))

        # ---- Trapezoid pooling ----
        point_w = self._nonuniform_trapezoid_weights(q_sorted)
        pooled = math.pi * (h * point_w).sum(dim=1)                    # [B, C]

        # ---- Log diagnostics (first forward only) ----
        if not hasattr(self, "_gate_debug_done"):
            self._gate_debug_done = True
            with torch.no_grad():
                p = 0
                print(f"[GATED] lambda_geo={self.lambda_geo:.4f}  "
                      f"gate_a={float(self.gate_a):.4f}  gate_b={float(self.gate_b):.4f}")
                print(f"[GATED] pixel {p}:")
                print(f"  centre_raw[:8]: {[float(x) for x in centre_raw[p,:8]]}")
                print(f"  q_sorted[:8]:   {[float(x) for x in q_sorted[p,:8]]}")
                print(f"  sort_order[:8]: {order[p,:8].cpu().tolist()}")
                if gate is not None:
                    G_sorted_p = G_sorted[p]
                    R2_sorted_p = R2_sorted[p]
                    s_p = s[p]
                    # Reorder checks
                    c_check = torch.gather(centre_raw[p], dim=0, index=order[p])
                    c_err = (c_check - centre_sorted[p]).abs().max().item()
                    G_check = torch.gather(G[p], dim=0, index=order[p])
                    G_err = (G_check - G_sorted_p).abs().max().item()
                    R2_check = torch.gather(R2[p], dim=0, index=order[p])
                    R2_err = (R2_check - R2_sorted_p).abs().max().item()
                    # s reorder: |G_sorted|*R2_sorted should equal gather(|G|*R2)
                    s_raw_unsorted = G[p].abs() * R2[p]
                    s_check = torch.gather(s_raw_unsorted, dim=0, index=order[p])
                    s_err = (s_check - s_p).abs().max().item()
                    print(f"  G_sorted[:8]:    {[float(x) for x in G_sorted_p[:8]]}")
                    print(f"  R2_sorted[:8]:   {[float(x) for x in R2_sorted_p[:8]]}")
                    print(f"  s_sorted[:8]:    {[float(x) for x in s_p[:8]]}")
                    print(f"  centre reorder check |err|: {c_err:.2e}")
                    print(f"  G      reorder check |err|: {G_err:.2e}")
                    print(f"  R2     reorder check |err|: {R2_err:.2e}")
                    print(f"  s      reorder check |err|: {s_err:.2e}")
                    print(f"  s mean={float(s.mean()):.6e}  std={float(s.std()):.6e}")
                    print(f"  gate mean={float(gate.mean()):.4f}  "
                          f"[{float(gate.min()):.4f}, {float(gate.max()):.4f}]")
                else:
                    print("  [gate DISABLED]")

        return self.out_mlp(pooled)

    def get_gate_epoch_stats(self) -> dict:
        """Collect and reset per-batch gate statistics for epoch logging."""
        if not hasattr(self, "_gate_epoch") or not self._gate_epoch.get("s"):
            return {}
        s_list = self._gate_epoch["s"]
        gate_list = self._gate_epoch["gate"]
        gate_means, gate_mins, gate_maxs = zip(*gate_list) if gate_list else ([], [], [])
        stats = {
            "s_batch_mean": float(np.mean(s_list)),
            "s_batch_std": float(np.std(s_list)),
            "gate_mean": float(np.mean(gate_means)),
            "gate_min": float(np.min(gate_mins)) if gate_mins else 0.0,
            "gate_max": float(np.max(gate_maxs)) if gate_maxs else 0.0,
            "gate_a": float(self.gate_a.detach().cpu()),
            "gate_b": float(self.gate_b.detach().cpu()),
            "lambda_geo": self.lambda_geo,
        }
        self._gate_epoch = {"s": [], "gate": []}
        return stats

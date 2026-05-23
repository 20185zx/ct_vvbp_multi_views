"""Exact-detector geometry GATED ADDITIVE RESIDUAL local-rank integral MLP.

Preserves the original [u, Q] token branch unchanged.  Adds a small
geometry-conditioned residual branch:

    s_v    = |G_v| * R2_v
    gate_v = sigmoid(a * s_norm_v + b)                 ∈ (0,1)
    r_v    = psi([G_v * R2_v,  |G_v| * R2_v])          [B, K, C]

    h_v    ← h_v  +  lambda_geo * gate_v * r_v

The residual MLP psi's last Linear layer is zero-initialised so the
model starts identical to the original local-rank MLP.
"""

from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.data.local_rank import compute_local_rank


class ExactDetectorGeometryGatedResidualLocalRankIntegralMLPNet(nn.Module):
    def __init__(
        self,
        point_hidden: int = 64,
        point_dim: int = 64,
        out_hidden: int = 128,
        dropout: float = 0.0,
        lambda_geo: float = 0.01,
        gate_a_init: float = 1.0,
        gate_b_init: float = 0.0,
        learnable_gate_affine: bool = True,
    ):
        super().__init__()
        self.use_coord = False
        self.input_mode = "values_sorted"
        self.lambda_geo = float(lambda_geo)

        # ---- Main branch  [u, Q] → h  (unchanged) ----
        self.point_mlp = nn.Sequential(
            nn.Linear(2, point_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(point_hidden, point_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(point_hidden, point_dim),
            nn.ReLU(inplace=True),
        )

        # ---- Geometry residual branch  [G*R2, |G|*R2] → r ----
        self.residual_mlp = nn.Sequential(
            nn.Linear(2, point_hidden),
            nn.SiLU(),
            nn.Linear(point_hidden, point_dim),
        )
        # Zero-init last layer so residual starts at zero
        nn.init.zeros_(self.residual_mlp[-1].weight)
        nn.init.zeros_(self.residual_mlp[-1].bias)

        # ---- Output head ----
        self.out_mlp = nn.Sequential(
            nn.Linear(point_dim, out_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(out_hidden, out_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(out_hidden, 1),
        )

        # ---- Learnable gate affine ----
        if learnable_gate_affine:
            self.gate_a = nn.Parameter(torch.tensor(float(gate_a_init)))
            self.gate_b = nn.Parameter(torch.tensor(float(gate_b_init)))
        else:
            self.register_buffer("gate_a", torch.tensor(float(gate_a_init)))
            self.register_buffer("gate_b", torch.tensor(float(gate_b_init)))

    @staticmethod
    def _trapezoid_weights(q_sorted):
        B, K = q_sorted.shape
        if K < 2:
            return torch.ones_like(q_sorted).unsqueeze(-1)
        dq = (q_sorted[:, 1:] - q_sorted[:, :-1]).clamp_min(0.0)
        w = torch.zeros_like(q_sorted)
        w[:, :-1] += 0.5 * dq
        w[:, 1:] += 0.5 * dq
        return (w / (w.sum(dim=-1, keepdim=True) + 1e-12)).unsqueeze(-1)

    def forward(self, values_sorted: torch.Tensor, stats: dict):
        B, J, K = values_sorted.shape
        if J != 9:
            raise ValueError(f"Expected 3x3 patch (J=9), got J={J}")
        dev = values_sorted.device

        # ---- Local-rank tokens ----
        q_local, centre_raw = compute_local_rank(values_sorted)
        q_sorted, order = torch.sort(q_local, dim=-1)
        centre_sorted = torch.gather(centre_raw, dim=-1, index=order)

        v_mean = stats["v_mean"].to(dev)
        v_std = stats["v_std"].to(dev)
        value_norm = (centre_sorted - v_mean) / v_std

        tokens = torch.stack([q_sorted, value_norm], dim=-1)   # [B, K, 2]
        h = self.point_mlp(tokens)                              # [B, K, C]

        # ---- Geometry residual ----
        G = stats.get("G", None)
        R2 = stats.get("R2", None)
        residual_norm_mean = 0.0
        residual_norm_max = 0.0

        if G is not None and R2 is not None and self.lambda_geo > 0:
            G = G.to(dev); R2 = R2.to(dev)
            G_sorted = torch.gather(G, dim=-1, index=order)
            R2_sorted = torch.gather(R2, dim=-1, index=order)

            # Gate feature  s = |G| * R2
            s = G_sorted.abs() * R2_sorted
            s_mean = stats.get("s_mean", s.mean()); s_std = stats.get("s_std", s.std().clamp_min(1e-8))
            if isinstance(s_mean, torch.Tensor): s_mean = s_mean.to(dev)
            if isinstance(s_std, torch.Tensor): s_std = s_std.to(dev)
            s_norm = (s - s_mean) / (s_std + 1e-8)
            gate = torch.sigmoid(self.gate_a * s_norm + self.gate_b)   # [B, K]

            # Residual features  gr = G*R2,  absgr = |G|*R2
            gr = G_sorted * R2_sorted
            absgr = s  # same as |G|*R2
            gr_mean = stats.get("gr_mean", gr.mean()); gr_std = stats.get("gr_std", gr.std().clamp_min(1e-8))
            if isinstance(gr_mean, torch.Tensor): gr_mean = gr_mean.to(dev)
            if isinstance(gr_std, torch.Tensor): gr_std = gr_std.to(dev)
            gr_norm = (gr - gr_mean) / (gr_std + 1e-8)
            absgr_norm = (absgr - s_mean) / (s_std + 1e-8)   # reuse s stats

            r_in = torch.stack([gr_norm, absgr_norm], dim=-1)         # [B, K, 2]
            r = self.residual_mlp(r_in)                                # [B, K, C]

            # Fusion
            h = h + self.lambda_geo * gate.unsqueeze(-1) * r

            with torch.no_grad():
                rn = r.norm(dim=-1)  # [B, K]
                residual_norm_mean = float(rn.mean().cpu())
                residual_norm_max = float(rn.max().cpu())

            # Gate epoch stats
            if not hasattr(self, "_gate_epoch"):
                self._gate_epoch = {"s": [], "gate": [], "rn": []}
            self._gate_epoch["s"].append(float(s.mean().cpu()))
            self._gate_epoch["gate"].append((float(gate.mean().cpu()),
                                              float(gate.min().cpu()),
                                              float(gate.max().cpu())))
            self._gate_epoch["rn"].append((residual_norm_mean, residual_norm_max))

        # ---- Trapezoid pooling ----
        w = self._trapezoid_weights(q_sorted)
        pooled = math.pi * (h * w).sum(dim=1)                          # [B, C]

        # ---- Debug (first forward only) ----
        if not hasattr(self, "_res_debug_done"):
            self._res_debug_done = True
            with torch.no_grad():
                print(f"[GATED-RES] lambda_geo={self.lambda_geo:.4f}  "
                      f"gate_a={float(self.gate_a):.4f}  gate_b={float(self.gate_b):.4f}")
                if gate is not None:
                    print(f"[GATED-RES] s mean={float(s.mean()):.6e}  std={float(s.std()):.6e}")
                    print(f"[GATED-RES] gate mean={float(gate.mean()):.4f}  "
                          f"[{float(gate.min()):.4f}, {float(gate.max()):.4f}]")
                    print(f"[GATED-RES] residual norm mean={residual_norm_mean:.6e}  max={residual_norm_max:.6e}")
                    print(f"[GATED-RES] h norm mean={float(h.norm(dim=-1).mean().cpu()):.6e}")

        return self.out_mlp(pooled)

    def get_gate_epoch_stats(self) -> dict:
        if not hasattr(self, "_gate_epoch") or not self._gate_epoch.get("s"):
            return {}
        s_list = self._gate_epoch["s"]
        gate_list = self._gate_epoch["gate"]
        rn_list = self._gate_epoch["rn"]
        gate_means, gate_mins, gate_maxs = zip(*gate_list) if gate_list else ([], [], [])
        rn_means, rn_maxs = zip(*rn_list) if rn_list else ([], [])
        stats = {
            "s_batch_mean": float(np.mean(s_list)),
            "s_batch_std": float(np.std(s_list)),
            "gate_mean": float(np.mean(gate_means)),
            "gate_min": float(np.min(gate_mins)) if gate_mins else 0.0,
            "gate_max": float(np.max(gate_maxs)) if gate_maxs else 0.0,
            "gate_a": float(self.gate_a.detach().cpu()),
            "gate_b": float(self.gate_b.detach().cpu()),
            "residual_norm_mean": float(np.mean(rn_means)) if rn_means else 0.0,
            "residual_norm_max": float(np.max(rn_maxs)) if rn_maxs else 0.0,
            "lambda_geo": self.lambda_geo,
        }
        self._gate_epoch = {"s": [], "gate": [], "rn": []}
        return stats

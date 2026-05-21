from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn.functional as F

from src.models.udno import UDNO
from src.models.real_norm_udno_base import RealNormUDNOBase


class RealNormUDNO(RealNormUDNOBase):
    """Real-valued normalized UDNO wrapper with fixed padded shape."""

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        radius_cutoff: float = 0.02,
        in_shape: Tuple[int, int] = (256, 256),
        kernel_shape: Tuple[int, int] = (6, 7),
        drop_prob: float = 0.0,
        use_norm: bool = True,
        unnorm_output: bool = True,
    ):
        super().__init__()
        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.use_norm = bool(use_norm)
        self.unnorm_output = bool(unnorm_output)

        h, w = int(in_shape[0]), int(in_shape[1])

        # UDNO receives padded input, so in_shape must be the padded shape.
        h_mult = ((h - 1) | 15) + 1
        w_mult = ((w - 1) | 15) + 1

        self.original_in_shape = (h, w)
        self.padded_in_shape = (h_mult, w_mult)

        self.udno = UDNO(
            in_chans=in_chans,
            out_chans=out_chans,
            radius_cutoff=radius_cutoff,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            in_shape=self.padded_in_shape,
            kernel_shape=kernel_shape,
        )

    def pad(self, x: torch.Tensor):
        _, _, h, w = x.shape

        h_mult, w_mult = self.padded_in_shape

        if h > h_mult or w > w_mult:
            raise ValueError(
                f"Input shape {(h, w)} is larger than configured padded shape "
                f"{self.padded_in_shape}."
            )

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def _scale_output(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        if self.unnorm_output:
            if self.in_chans != self.out_chans:
                raise RuntimeError(
                    "RealNormUDNO unnormalization requires in_chans == out_chans. "
                    f"Got in_chans={self.in_chans}, out_chans={self.out_chans}."
                )
            x = self.unnorm(x, mean, std)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

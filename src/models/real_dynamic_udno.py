from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from src.models.dynamic_udno import DynamicUDNO
from src.models.real_norm_udno_base import RealNormUDNOBase


class RealDynamicNormUDNO(RealNormUDNOBase):
    """Real-valued normalized DynamicUDNO wrapper.

    Pads input to a multiple of 2**num_pool_layers on the fly, so it accepts
    variable spatial shapes. DynamicUDNO re-discretizes DISCO kernels
    accordingly; learnable parameters are shared across all input shapes.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        chans: int = 32,
        num_pool_layers: int = 4,
        radius_cutoff: float = 0.02,
        kernel_shape=(6, 7),
        drop_prob: float = 0.0,
        use_norm: bool = True,
        output_scale_mode: str = "residual",
        padding_mode: str = "constant",
        zero_init_output: bool = True,
    ):
        super().__init__()

        self.in_chans = int(in_chans)
        self.out_chans = int(out_chans)
        self.use_norm = bool(use_norm)
        self.output_scale_mode = str(output_scale_mode)

        if self.output_scale_mode not in ["full", "residual", "none"]:
            raise ValueError(
                f"Unknown output_scale_mode={self.output_scale_mode}. "
                "Expected one of ['full', 'residual', 'none']."
            )

        self.num_pool_layers = int(num_pool_layers)

        self.udno = DynamicUDNO(
            in_chans=in_chans,
            out_chans=out_chans,
            radius_cutoff=radius_cutoff,
            chans=chans,
            num_pool_layers=num_pool_layers,
            drop_prob=drop_prob,
            kernel_shape=kernel_shape,
            padding_mode=padding_mode,
        )

        if zero_init_output:
            self.udno.zero_init_output()

    def pad(self, x: torch.Tensor):
        _, _, h, w = x.shape

        multiple = 2 ** self.num_pool_layers

        h_mult = math.ceil(h / multiple) * multiple
        w_mult = math.ceil(w / multiple) * multiple

        w_pad = [math.floor((w_mult - w) / 2), math.ceil((w_mult - w) / 2)]
        h_pad = [math.floor((h_mult - h) / 2), math.ceil((h_mult - h) / 2)]

        x = F.pad(x, w_pad + h_pad)
        return x, (h_pad, w_pad, h_mult, w_mult)

    def _scale_output(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        if self.output_scale_mode == "full":
            if self.in_chans != self.out_chans:
                raise RuntimeError(
                    "Full unnormalization requires in_chans == out_chans."
                )
            x = x * std + mean

        elif self.output_scale_mode == "residual":
            if self.in_chans != self.out_chans:
                raise RuntimeError(
                    "Residual scaling requires in_chans == out_chans."
                )
            x = x * std

        elif self.output_scale_mode == "none":
            pass

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x)

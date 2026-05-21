"""Shared base class for RealNormUDNO and RealDynamicNormUDNO.

Both wrappers apply instance normalization + padding around a UDNO/DynamicUDNO
core. The only differences are:
  - how padding size is determined (fixed vs dynamic);
  - how the output is scaled after unnorm.
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class RealNormUDNOBase(nn.Module):
    """Base class: norm → pad → udno → unpad → output_scale."""

    def norm(self, x: torch.Tensor):
        mean = x.mean(dim=(-2, -1), keepdim=True)
        std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        return (x - mean) / std, mean, std

    def unnorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
        return x * std + mean

    def unpad(
        self,
        x: torch.Tensor,
        h_pad: List[int],
        w_pad: List[int],
        h_mult: int,
        w_mult: int,
    ):
        return x[..., h_pad[0]: h_mult - h_pad[1], w_pad[0]: w_mult - w_pad[1]]

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            x, mean, std = self.norm(x)
        else:
            mean = std = None

        x, pad_sizes = self.pad(x)
        x = self.udno(x)
        x = self.unpad(x, *pad_sizes)

        if self.use_norm:
            x = self._scale_output(x, mean, std)

        return x

    # Subclasses must implement:
    #   pad(self, x) -> (padded, (h_pad, w_pad, h_mult, w_mult))
    #   _scale_output(self, x, mean, std) -> x

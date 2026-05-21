"""CTO-adapted multi-rate sparse-view CT baseline with strict sparse-matrix data consistency.

This is not the official CTO implementation. It keeps the important CTO ingredients:

1. sinogram-space operator NO_s with spatial and detector-frequency branches;
2. fixed fan-beam FBP initialization;
3. image-space operator NO_i cascades;
4. explicit CT data-consistency update using S^T(S x - y), where S is an
   ASTRA-generated fan-beam sparse projection matrix.

The strict projector is provided by src.geometry.astra_sparse_projector.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.geometry.fbp import LInFBPFixedLinearFBPBatch
from src.geometry.astra_sparse_projector import AstraSparseFanBeamProjector
from src.models.real_dynamic_udno import RealDynamicNormUDNO

def check_finite(name: str, x: torch.Tensor):
    if not torch.isfinite(x).all():
        finite = torch.isfinite(x)
        print(f"[NaN/Inf detected] {name}")
        print(f"  shape = {tuple(x.shape)}")
        print(f"  finite ratio = {finite.float().mean().item():.6f}")
        if finite.any():
            xf = x[finite]
            print(f"  min = {xf.min().item():.6e}")
            print(f"  max = {xf.max().item():.6e}")
            print(f"  mean = {xf.mean().item():.6e}")
            print(f"  std = {xf.std().item():.6e}")
        raise RuntimeError(f"Non-finite tensor detected: {name}")
    
class ResidualConvBlock(nn.Module):
    def __init__(self, channels: int, padding_mode: str = "replicate"):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode=padding_mode)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, padding_mode=padding_mode)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SmallOperator2D(nn.Module):
    """Light U-shaped convolutional operator block.

    This is a practical placeholder for DISCO/UDNO. It is fully convolutional,
    so it can accept variable sinogram view counts and image resolutions.
    """

    def __init__(self, in_ch=1, out_ch=1, hidden=32, blocks=3, padding_mode="replicate"):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1, padding_mode=padding_mode),
            nn.GELU(),
        )
        self.body = nn.Sequential(*[ResidualConvBlock(hidden, padding_mode=padding_mode) for _ in range(blocks)])
        self.exit = nn.Conv2d(hidden, out_ch, 3, padding=1, padding_mode=padding_mode)

    def forward(self, x):
        return self.exit(self.body(self.entry(x)))


class SinogramOperator(nn.Module):
    """Shared-parameter sinogram-space UDNO/DISCO operator.

    Route A:
    one spatial UDNO and one frequency UDNO are shared across all sparse-view
    rates. Different view counts only induce different dynamic DISCO
    discretizations inside DynamicUDNO.
    """

    def __init__(
        self,
        sparse_views,
        n_detec: int = 672,
        hidden: int = 32,
        num_pool_layers: int = 4,
        radius_cutoff: float = 0.02,
        kernel_shape=(6, 7),
        residual_scale: float = 0.01,
        drop_prob: float = 0.0,
        sino_padding_mode: str = "sino_circular",
    ):
        super().__init__()

        self.residual_scale = float(residual_scale)
        self.sparse_views = [int(v) for v in sparse_views]
        self.n_detec = int(n_detec)
        self.sino_padding_mode = sino_padding_mode

        self.spatial_op = RealDynamicNormUDNO(
            in_chans=1,
            out_chans=1,
            chans=hidden,
            num_pool_layers=num_pool_layers,
            radius_cutoff=radius_cutoff,
            kernel_shape=kernel_shape,
            drop_prob=drop_prob,
            use_norm=True,
            output_scale_mode="residual",
            padding_mode=sino_padding_mode,   # 当前用 "sino_circular"
            zero_init_output=True,
        )

        self.freq_op = RealDynamicNormUDNO(
            in_chans=2,
            out_chans=2,
            chans=hidden,
            num_pool_layers=num_pool_layers,
            radius_cutoff=radius_cutoff,
            kernel_shape=kernel_shape,
            drop_prob=drop_prob,
            use_norm=True,
            output_scale_mode="residual",
            padding_mode=sino_padding_mode,
            zero_init_output=True,
        )

    def forward(self, sino):
        # sino: [B,1,V,D]
        V = int(sino.shape[2])
        if V not in self.sparse_views:
            raise ValueError(f"Unsupported view count {V}; expected {self.sparse_views}")

        spatial_res = self.spatial_op(sino)

        z = torch.fft.rfft(sino.squeeze(1), dim=-1, norm="ortho")  # [B,V,Df]
        z_ri = torch.stack([z.real, z.imag], dim=1)                 # [B,2,V,Df]

        freq_res_ri = self.freq_op(z_ri)
        z_res = torch.complex(freq_res_ri[:, 0], freq_res_ri[:, 1])
        freq_res = torch.fft.irfft(
            z_res,
            n=sino.shape[-1],
            dim=-1,
            norm="ortho",
        ).unsqueeze(1)

        res = 0.5 * (spatial_res + freq_res)
        return sino + self.residual_scale * res

class CTOCascade(nn.Module):
    def __init__(
        self,
        image_size: int = 256,
        hidden: int = 32,
        num_pool_layers: int = 4,
        radius_cutoff: float = 0.02,
        kernel_shape=(6, 7),
        dc_step: float = 1e-7,
        reg_scale: float = 0.01,
        drop_prob: float = 0.0,
        image_padding_mode: str = "reflect",
    ):
        super().__init__()

        self.image_op = RealDynamicNormUDNO(
            in_chans=1,
            out_chans=1,
            chans=hidden,
            num_pool_layers=num_pool_layers,
            radius_cutoff=radius_cutoff,
            kernel_shape=kernel_shape,
            drop_prob=drop_prob,
            use_norm=True,
            output_scale_mode="residual",
            padding_mode=image_padding_mode,
            zero_init_output=True,
        )

        dc_init = max(float(dc_step), 1e-12)
        reg_init = max(float(reg_scale), 1e-12)

        # Fixed DC step from config (non-learnable hyperparameter)
        self.register_buffer('dc_step', torch.tensor(dc_init, dtype=torch.float32))

        # Learnable regularization scale
        self.reg_log = nn.Parameter(
            torch.tensor(math.log(math.expm1(reg_init)), dtype=torch.float32)
        )

    def reg_scale(self):
        return F.softplus(self.reg_log)

    def forward(self, x, sino_measured, views: int, projector: AstraSparseFanBeamProjector):
        reg = self.image_op(x)

        sino_pred = projector.forward(x, views)
        sino_residual = sino_pred - sino_measured.to(device=x.device, dtype=x.dtype)
        dc = projector.adjoint(sino_residual, views)

        eta = self.dc_step  # fixed hyperparameter from config
        lam = torch.clamp(self.reg_scale(), min=0.0, max=1.0)

        return x - eta * dc + lam * reg
    

class CTOAdaptedNet(nn.Module):
    """CTO-adapted network for multi-rate fan-beam sparse-view CT.

    Parameters
    ----------
    geo_dict:
        Mapping V -> LInFBP geometry used for fixed FBP initialization.
    projector:
        AstraSparseFanBeamProjector used for strict sparse-matrix data consistency S^T(Sx - y).
    sparse_views:
        Supported sparse-view counts, e.g. [9,18,36,72].
    """

    def __init__(
        self,
        geo_dict: Dict[int, Dict],
        projector: AstraSparseFanBeamProjector,
        sparse_views: Iterable[int],
        image_size: int = 256,
        n_detec: int = 672,
        sino_hidden: int = 32,
        image_hidden: int = 32,
        cascades: int = 3,
        sino_residual_scale: float = 0.05,
        dc_step: float = 1e-7,
        image_reg_scale: float = 0.01,
        udno_pools: int = 4,
        udno_radius_cutoff: float = 0.02,
        udno_kernel_shape=(6, 7),
        udno_drop_prob: float = 0.0,
        sino_padding_mode: str = "sino_circular",
        image_padding_mode: str = "reflect",
    ):
        super().__init__()
        self.projector = projector
        self.sparse_views = [int(v) for v in sparse_views]
        self.sino_op = SinogramOperator(
            sparse_views=self.sparse_views,
            n_detec=n_detec,
            hidden=sino_hidden,
            num_pool_layers=udno_pools,
            radius_cutoff=udno_radius_cutoff,
            kernel_shape=udno_kernel_shape,
            residual_scale=sino_residual_scale,
            drop_prob=udno_drop_prob,
            sino_padding_mode=sino_padding_mode,
        )
        self.fbp_layers = nn.ModuleDict({str(int(v)): LInFBPFixedLinearFBPBatch(geo_dict[int(v)]) for v in self.sparse_views})
        self.cascades = nn.ModuleList([
            CTOCascade(
                image_size=image_size,
                hidden=image_hidden,
                num_pool_layers=udno_pools,
                radius_cutoff=udno_radius_cutoff,
                kernel_shape=udno_kernel_shape,
                dc_step=dc_step,
                reg_scale=image_reg_scale,
                drop_prob=udno_drop_prob,
                image_padding_mode=image_padding_mode,
            )
            for _ in range(cascades)
        ])

    def forward(self, sino_sparse):
        views = int(sino_sparse.shape[2])
        if views not in self.sparse_views:
            raise ValueError(f"Unsupported view count {views}; expected one of {self.sparse_views}")

        check_finite(f"sino_sparse V={views}", sino_sparse)

        sino_corr = self.sino_op(sino_sparse)
        check_finite(f"sino_corr V={views}", sino_corr)

        x = self.fbp_layers[str(views)](sino_corr)
        check_finite(f"FBP init x V={views}", x)

        for i, cascade in enumerate(self.cascades):
            x = cascade(x, sino_sparse, views, self.projector)
            check_finite(f"cascade {i} output V={views}", x)

        return x

from __future__ import annotations

import math
from typing import Tuple, Union, List, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_harmonics.convolution import _precompute_convolution_tensor_2d


def _pad_detector_axis(x: torch.Tensor, left: int, right: int, mode: str = "reflect"):
    """Pad detector/frequency axis, i.e. W dimension."""
    if left == 0 and right == 0:
        return x

    if mode == "reflect":
        # reflect requires padding size < input size
        if left >= x.shape[-1] or right >= x.shape[-1]:
            return F.pad(x, (left, right, 0, 0), mode="replicate")
        return F.pad(x, (left, right, 0, 0), mode="reflect")

    if mode == "replicate":
        return F.pad(x, (left, right, 0, 0), mode="replicate")

    return F.pad(x, (left, right, 0, 0), mode="constant", value=0.0)


def _pad_angle_axis_circular(
    x: torch.Tensor,
    top: int,
    bottom: int,
    flip_detector: bool = False,
):
    """Circular pad angle axis, optionally flipping detector axis when wrapping.

    x: [B, C, V, D]
    top/bottom pad along V dimension.
    """
    if top == 0 and bottom == 0:
        return x

    V = x.shape[-2]
    parts = []

    if top > 0:
        top_idx = torch.arange(-top, 0, device=x.device) % V
        top_part = x.index_select(-2, top_idx)
        if flip_detector:
            top_part = torch.flip(top_part, dims=[-1])
        parts.append(top_part)

    parts.append(x)

    if bottom > 0:
        bottom_idx = torch.arange(0, bottom, device=x.device) % V
        bottom_part = x.index_select(-2, bottom_idx)
        if flip_detector:
            bottom_part = torch.flip(bottom_part, dims=[-1])
        parts.append(bottom_part)

    return torch.cat(parts, dim=-2)


def pad_for_disco_2d(
    x: torch.Tensor,
    left: int,
    right: int,
    top: int,
    bottom: int,
    padding_mode: str,
):
    """Padding used before dynamic DISCO convolution.

    Modes:
        - "constant": ordinary zero padding
        - "reflect": ordinary reflect padding
        - "sino_circular": circular along angle, reflect along detector
        - "sino_pi_flip": circular along angle + detector flip at boundary
    """
    if padding_mode == "constant":
        return F.pad(x, (left, right, top, bottom), mode="constant", value=0.0)

    if padding_mode == "reflect":
        return F.pad(x, (left, right, top, bottom), mode="reflect")

    if padding_mode == "replicate":
        return F.pad(x, (left, right, top, bottom), mode="replicate")

    if padding_mode == "sino_circular":
        # detector axis: normal reflect padding
        x = _pad_detector_axis(x, left, right, mode="reflect")
        # angle axis: circular, no detector flip
        x = _pad_angle_axis_circular(x, top, bottom, flip_detector=False)
        return x

    if padding_mode == "sino_pi_flip":
        # detector axis: normal reflect padding
        x = _pad_detector_axis(x, left, right, mode="reflect")
        # angle axis: circular with detector flip
        x = _pad_angle_axis_circular(x, top, bottom, flip_detector=True)
        return x

    raise ValueError(f"Unknown padding_mode: {padding_mode}")


class DynamicEquidistantDISCO2d(nn.Module):
    """Shape-dynamic equidistant DISCO convolution.

    This keeps the learnable DISCO kernel weights shared across resolutions,
    while dynamically compiling the local interpolation tensor psi_loc according
    to the current input spatial shape.

    This is the key change needed for route A:
        same parameters, different discretizations.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_shape: Union[int, List[int], Tuple[int, int]],
        groups: int = 1,
        bias: bool = True,
        radius_cutoff: Optional[float] = None,
        padding_mode: str = "constant",
        use_min_dim: bool = True,
    ):
        super().__init__()

        if isinstance(kernel_shape, int):
            self.kernel_shape = [kernel_shape]
        else:
            self.kernel_shape = list(kernel_shape)

        if len(self.kernel_shape) == 1:
            self.kernel_size = self.kernel_shape[0]
        elif len(self.kernel_shape) == 2:
            self.kernel_size = (self.kernel_shape[0] - 1) * self.kernel_shape[1] + 1
        else:
            raise ValueError("kernel_shape must be one- or two-dimensional.")

        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.groups = int(groups)
        self.padding_mode = padding_mode
        self.radius_cutoff = radius_cutoff
        self.use_min_dim = bool(use_min_dim)

        if self.in_channels % self.groups != 0:
            raise ValueError("in_channels must be divisible by groups.")
        if self.out_channels % self.groups != 0:
            raise ValueError("out_channels must be divisible by groups.")

        self.groupsize = self.in_channels // self.groups
        scale = math.sqrt(1.0 / self.groupsize)
        self.weight = nn.Parameter(
            scale * torch.randn(self.out_channels, self.groupsize, self.kernel_size)
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(self.out_channels))
        else:
            self.bias = None

        # Cache psi_loc by spatial shape and device.
        self._psi_cache: Dict[tuple, torch.Tensor] = {}

    def _compile_psi_loc(
        self,
        h: int,
        w: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        f = min if self.use_min_dim else max
        dim_ref = f(h, w)

        if self.radius_cutoff is None:
            radius_cutoff = 2.0 * self.kernel_shape[0] / float(dim_ref)
        else:
            radius_cutoff = float(self.radius_cutoff)

        if radius_cutoff <= 0:
            raise ValueError("radius_cutoff must be positive.")

        psi_local_size = math.floor(2 * radius_cutoff * dim_ref / 2) + 1
        psi_local_size = max(int(psi_local_size), 1)

        x = torch.linspace(
            -radius_cutoff,
            radius_cutoff,
            psi_local_size,
            dtype=torch.float32,
        )
        x, y = torch.meshgrid(x, x, indexing="ij")
        grid_in = torch.stack([x.reshape(-1), y.reshape(-1)])
        grid_out = torch.tensor([[0.0], [0.0]], dtype=torch.float32)

        idx, vals = _precompute_convolution_tensor_2d(
            grid_in,
            grid_out,
            self.kernel_shape,
            radius_cutoff=radius_cutoff,
            periodic=False,
        )

        psi_loc = torch.zeros(
            self.kernel_size,
            psi_local_size * psi_local_size,
            dtype=torch.float32,
        )

        for ie in range(len(vals)):
            fidx = idx[0, ie]
            j = idx[2, ie]
            v = vals[ie]
            psi_loc[fidx, j] = v

        psi_loc = psi_loc.reshape(
            self.kernel_size,
            psi_local_size,
            psi_local_size,
        )

        # Same normalization logic as original EquidistantDiscreteContinuousConv2d.
        psi_loc = 4.0 * psi_loc / float(h * w)

        return psi_loc.to(device=device, dtype=dtype)

    def get_psi_loc(self, h: int, w: int, x: torch.Tensor) -> torch.Tensor:
        key = (int(h), int(w), str(x.device), str(x.dtype))

        if key not in self._psi_cache:
            self._psi_cache[key] = self._compile_psi_loc(
                h=int(h),
                w=int(w),
                device=x.device,
                dtype=x.dtype,
            )

        return self._psi_cache[key]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape

        psi_loc = self.get_psi_loc(h, w, x)
        kernel = torch.einsum("kxy,ogk->ogxy", psi_loc, self.weight)

        psi_local_size = psi_loc.shape[-1]
        left_pad = psi_local_size // 2
        right_pad = (psi_local_size + 1) // 2 - 1

        x = pad_for_disco_2d(
            x,
            left=left_pad,
            right=right_pad,
            top=left_pad,
            bottom=right_pad,
            padding_mode=self.padding_mode,
        )

        out = F.conv2d(
            x,
            kernel,
            self.bias,
            stride=1,
            dilation=1,
            padding=0,
            groups=self.groups,
        )

        return out


class DynamicDISCOBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        drop_prob: float,
        kernel_shape: Tuple[int, int] = (6, 7),
        padding_mode: str = "constant",
    ):
        super().__init__()

        self.layers = nn.Sequential(
            DynamicEquidistantDISCO2d(
                in_chans,
                out_chans,
                kernel_shape=kernel_shape,
                bias=False,
                radius_cutoff=radius_cutoff,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
            DynamicEquidistantDISCO2d(
                out_chans,
                out_chans,
                kernel_shape=kernel_shape,
                bias=False,
                radius_cutoff=radius_cutoff,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(drop_prob),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DynamicTransposeDISCOBlock(nn.Module):
    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        kernel_shape: Tuple[int, int] = (6, 7),
        padding_mode: str = "constant",
    ):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            DynamicEquidistantDISCO2d(
                in_chans,
                out_chans,
                kernel_shape=kernel_shape,
                bias=False,
                radius_cutoff=radius_cutoff / 2.0,
                padding_mode=padding_mode,
            ),
            nn.InstanceNorm2d(out_chans),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class DynamicUDNO(nn.Module):
    """U-shaped DISCO Neural Operator with shared parameters across input shapes.

    Unlike the original UDNO, this module does not store shape-specific DISCO
    buffers at initialization. DISCO discretization is compiled dynamically in
    forward based on the current tensor shape.
    """

    def __init__(
        self,
        in_chans: int,
        out_chans: int,
        radius_cutoff: float,
        chans: int = 32,
        num_pool_layers: int = 4,
        drop_prob: float = 0.0,
        kernel_shape: Tuple[int, int] = (6, 7),
        padding_mode: str = "constant",
    ):
        super().__init__()

        self.in_chans = in_chans
        self.out_chans = out_chans
        self.chans = chans
        self.num_pool_layers = num_pool_layers
        self.drop_prob = drop_prob
        self.kernel_shape = kernel_shape

        self.down_sample_layers = nn.ModuleList(
            [
                DynamicDISCOBlock(
                    in_chans,
                    chans,
                    radius_cutoff,
                    drop_prob,
                    kernel_shape=kernel_shape,
                    padding_mode=padding_mode,
                )
            ]
        )

        ch = chans
        r = radius_cutoff * 2.0

        for _ in range(num_pool_layers - 1):
            self.down_sample_layers.append(
                DynamicDISCOBlock(
                    ch,
                    ch * 2,
                    r,
                    drop_prob,
                    kernel_shape=kernel_shape,
                    padding_mode=padding_mode,
                )
            )
            ch *= 2
            r *= 2.0

        self.bottleneck = DynamicDISCOBlock(
            ch,
            ch * 2,
            r,
            drop_prob,
            kernel_shape=kernel_shape,
            padding_mode=padding_mode,
        )

        self.up = nn.ModuleList()
        self.up_transpose = nn.ModuleList()

        for _ in range(num_pool_layers - 1):
            self.up_transpose.append(
                DynamicTransposeDISCOBlock(
                    ch * 2,
                    ch,
                    r,
                    kernel_shape=kernel_shape,
                    padding_mode=padding_mode,
                )
            )
            r /= 2.0

            self.up.append(
                DynamicDISCOBlock(
                    ch * 2,
                    ch,
                    r,
                    drop_prob,
                    kernel_shape=kernel_shape,
                    padding_mode=padding_mode,
                )
            )
            ch //= 2

        self.up_transpose.append(
            DynamicTransposeDISCOBlock(
                ch * 2,
                ch,
                r,
                kernel_shape=kernel_shape,
                padding_mode=padding_mode,
            )
        )
        r /= 2.0

        self.up.append(
            nn.Sequential(
                DynamicDISCOBlock(
                    ch * 2,
                    ch,
                    r,
                    drop_prob,
                    kernel_shape=kernel_shape,
                    padding_mode=padding_mode,
                ),
                nn.Conv2d(ch, self.out_chans, kernel_size=1, stride=1),
            )
        )

    def zero_init_output(self):
        """Zero-initialize the final 1x1 output convolution.

        This makes the neural operator initially output zero residual, so the
        whole CTO model starts close to FBP.
        """
        final_module = self.up[-1]

        if isinstance(final_module, nn.Sequential):
            last = final_module[-1]
            if isinstance(last, nn.Conv2d):
                nn.init.zeros_(last.weight)
                if last.bias is not None:
                    nn.init.zeros_(last.bias)
                return

        raise RuntimeError("Could not find final Conv2d for zero initialization.")


    def forward(self, image: torch.Tensor) -> torch.Tensor:
        stack = []
        output = image

        for layer in self.down_sample_layers:
            output = layer(output)
            stack.append(output)
            output = F.avg_pool2d(output, kernel_size=2, stride=2, padding=0)

        output = self.bottleneck(output)

        for transpose, disco in zip(self.up_transpose, self.up):
            downsample_layer = stack.pop()
            output = transpose(output)

            padding = [0, 0, 0, 0]
            if output.shape[-1] != downsample_layer.shape[-1]:
                padding[1] = downsample_layer.shape[-1] - output.shape[-1]
            if output.shape[-2] != downsample_layer.shape[-2]:
                padding[3] = downsample_layer.shape[-2] - output.shape[-2]

            if any(p != 0 for p in padding):
                output = F.pad(output, padding, "reflect")

            output = torch.cat([output, downsample_layer], dim=1)
            output = disco(output)

        return output
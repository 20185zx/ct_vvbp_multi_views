"""ASTRA-based differentiable fan-beam forward and adjoint projectors.

This module provides the CT physics operator needed by CTO-style unrolled
networks:

    A      : image -> fan-beam sinogram
    A^*    : sinogram -> unfiltered fan-beam backprojection

Important: A^* here is the mathematical adjoint/backprojection used in the
normal-equation data-consistency term A^*(A x - y). It is not FBP. FBP applies
ramp filtering and fan-beam weighting and is used separately for initialization.

The PyTorch interface is differentiable through custom autograd.Functions:
- backward of forward projection uses ASTRA backprojection
- backward of backprojection uses ASTRA forward projection

ASTRA itself performs the fixed linear operators; PyTorch receives tensors with
correct gradients for unrolled training.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Optional

import numpy as np
import torch
from torch.autograd import Function

try:
    import astra  # type: ignore
except Exception as exc:  # pragma: no cover - handled at runtime
    astra = None
    _ASTRA_IMPORT_ERROR = exc
else:
    _ASTRA_IMPORT_ERROR = None


class _AstraForwardFn(Function):
    @staticmethod
    def forward(ctx, image: torch.Tensor, operator: "AstraFanBeamProjector", views: int):
        ctx.operator = operator
        ctx.views = int(views)
        ctx.input_shape = tuple(image.shape)
        with torch.no_grad():
            sino = operator._forward_no_grad(image, int(views))
        return sino

    @staticmethod
    def backward(ctx, grad_sino: torch.Tensor):
        with torch.no_grad():
            grad_image = ctx.operator._adjoint_no_grad(grad_sino, ctx.views)
        # image, operator, views
        return grad_image, None, None


class _AstraBackprojectFn(Function):
    @staticmethod
    def forward(ctx, sino: torch.Tensor, operator: "AstraFanBeamProjector", views: int):
        ctx.operator = operator
        ctx.views = int(views)
        ctx.input_shape = tuple(sino.shape)
        with torch.no_grad():
            image = operator._adjoint_no_grad(sino, int(views))
        return image

    @staticmethod
    def backward(ctx, grad_image: torch.Tensor):
        with torch.no_grad():
            scale = float(ctx.operator.get_adjoint_scale(ctx.views))
            grad_sino = ctx.operator._forward_no_grad(grad_image, ctx.views) * scale
        # sino, operator, views
        return grad_sino, None, None


class AstraFanBeamProjector:
    """ASTRA fan-beam A/A* wrapper with optional PyTorch autograd.

    Parameters follow the AAPM-style fan-beam setting used in the project:
    image_size=256, detector_elements=672, DSO=595, DOD=480, angle range [0,2π).

    Input/output tensor conventions:
        forward(image, V): image [B,1,H,W] -> sino [B,1,V,D]
        adjoint(sino, V): sino [B,1,V,D] -> image [B,1,H,W]

    Notes
    -----
    - This class caches one ASTRA projector per view count V.
    - `adjoint` is unfiltered backprojection, not FBP.
    - ASTRA uses NumPy arrays internally; this wrapper copies tensors CPU<->GPU.
      It is correct and useful for a faithful CTO baseline, though not the most
      optimized implementation.
    """

    def __init__(
        self,
        image_size: int = 256,
        n_detec: int = 672,
        d_detec: float = 1.0,
        d_voxel: float = 1.0,
        DSO: float = 595.0,
        DOD: float = 480.0,
        start_angle: float = 0.0,
        end_angle: float = 2 * math.pi,
        projector_type: Optional[str] = None,
        use_cuda: Optional[bool] = None,
        sparse_views: Optional[Iterable[int]] = None,
    ):
        if astra is None:
            raise ImportError(
                "ASTRA toolbox is required for AstraFanBeamProjector. "
                f"Original import error: {_ASTRA_IMPORT_ERROR}"
            )

        self.image_size = int(image_size)
        self.n_detec = int(n_detec)
        self.d_detec = float(d_detec)
        self.d_voxel = float(d_voxel)
        self.DSO = float(DSO)
        self.DOD = float(DOD)
        self.DSD = self.DSO + self.DOD
        self.start_angle = float(start_angle)
        self.end_angle = float(end_angle)

        if use_cuda is None:
            use_cuda = bool(astra.use_cuda())
        self.use_cuda = bool(use_cuda)
        self.projector_type = projector_type or ("cuda" if self.use_cuda else "line_fanflat")

        s_voxel = self.image_size * self.d_voxel
        self.vol_geom = astra.create_vol_geom(
            self.image_size,
            self.image_size,
            -s_voxel / 2,
            s_voxel / 2,
            -s_voxel / 2,
            s_voxel / 2,
        )

        self._proj_cache: Dict[int, Dict] = {}
        self.adjoint_scales: Dict[int, float] = {}
        if sparse_views is not None:
            for v in sparse_views:
                self.get_geometry(int(v))

    @classmethod
    def from_experiment_config(cls, cfg, sparse_views=None) -> "AstraFanBeamProjector":
        return cls(
            image_size=cfg.image_size,
            n_detec=cfg.n_detec,
            d_detec=cfg.d_detec,
            d_voxel=cfg.d_voxel,
            DSO=cfg.DSO,
            DOD=cfg.DOD,
            sparse_views=sparse_views if sparse_views is not None else cfg.sparse_views,
        )

    def _angles(self, views: int) -> np.ndarray:
        return np.linspace(self.start_angle, self.end_angle, int(views), endpoint=False).astype(np.float32)

    def get_geometry(self, views: int) -> Dict:
        views = int(views)
        if views not in self._proj_cache:
            proj_geom = astra.create_proj_geom(
                "fanflat",
                self.d_detec,
                self.n_detec,
                self._angles(views),
                self.DSO,
                self.DOD,
            )
            projector_id = astra.create_projector(self.projector_type, proj_geom, self.vol_geom)
            self._proj_cache[views] = {
                "proj_geom": proj_geom,
                "projector_id": projector_id,
            }
        return self._proj_cache[views]

    def clear(self) -> None:
        if astra is None:
            return
        for item in self._proj_cache.values():
            try:
                astra.projector.delete(item["projector_id"])
            except Exception:
                pass
        self._proj_cache.clear()

    def __del__(self):  # pragma: no cover - cleanup best effort
        try:
            self.clear()
        except Exception:
            pass

    @staticmethod
    def _as_image_bchw(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 2:
            image = image[None, None]
        elif image.ndim == 3:
            image = image[:, None]
        if image.ndim != 4 or image.shape[1] != 1:
            raise ValueError(f"Expected image [B,1,H,W], [B,H,W], or [H,W], got {tuple(image.shape)}")
        return image

    @staticmethod
    def _as_sino_bcvd(sino: torch.Tensor) -> torch.Tensor:
        if sino.ndim == 2:
            sino = sino[None, None]
        elif sino.ndim == 3:
            sino = sino[:, None]
        if sino.ndim != 4 or sino.shape[1] != 1:
            raise ValueError(f"Expected sino [B,1,V,D], [B,V,D], or [V,D], got {tuple(sino.shape)}")
        return sino

    def _forward_numpy_single(self, image_hw: np.ndarray, views: int) -> np.ndarray:
        projector_id = self.get_geometry(views)["projector_id"]
        sino_id, sino = astra.create_sino(np.ascontiguousarray(image_hw, dtype=np.float32), projector_id)
        astra.data2d.delete(sino_id)
        return sino.astype(np.float32, copy=False)

    def _adjoint_numpy_single(self, sino_vd: np.ndarray, views: int) -> np.ndarray:
        projector_id = self.get_geometry(views)["projector_id"]
        bp_id, bp = astra.create_backprojection(np.ascontiguousarray(sino_vd, dtype=np.float32), projector_id)
        astra.data2d.delete(bp_id)
        return bp.astype(np.float32, copy=False)

    def _forward_no_grad(self, image: torch.Tensor, views: int) -> torch.Tensor:
        image = self._as_image_bchw(image)
        device = image.device
        dtype = image.dtype
        arr = image.detach().float().cpu().numpy()
        out = [self._forward_numpy_single(arr[b, 0], int(views)) for b in range(arr.shape[0])]
        sino = np.stack(out, axis=0)[:, None, :, :]  # [B,1,V,D]
        return torch.from_numpy(sino).to(device=device, dtype=dtype)

    def _adjoint_raw_no_grad(self, sino: torch.Tensor, views: int) -> torch.Tensor:
        """Raw ASTRA unfiltered backprojection without adjoint scale."""
        sino = self._as_sino_bcvd(sino)
        device = sino.device
        dtype = sino.dtype
        arr = sino.detach().float().cpu().numpy()
        out = [self._adjoint_numpy_single(arr[b, 0], int(views)) for b in range(arr.shape[0])]
        image = np.stack(out, axis=0)[:, None, :, :]  # [B,1,H,W]
        return torch.from_numpy(image).to(device=device, dtype=dtype)

    def _adjoint_no_grad(self, sino: torch.Tensor, views: int) -> torch.Tensor:
        """Scaled unfiltered fan-beam backprojection used as A*."""
        views = int(views)
        image = self._adjoint_raw_no_grad(sino, views)
        scale = float(self.adjoint_scales.get(views, 1.0))
        return image * scale

    def forward(self, image: torch.Tensor, views: int) -> torch.Tensor:
        """Differentiable fan-beam forward projection A(x)."""
        return _AstraForwardFn.apply(image, self, int(views))

    def adjoint(self, sino: torch.Tensor, views: int) -> torch.Tensor:
        """Differentiable unfiltered fan-beam backprojection A*(sino)."""
        return _AstraBackprojectFn.apply(sino, self, int(views))

    def forward_no_grad(self, image: torch.Tensor, views: int) -> torch.Tensor:
        """Non-autograd forward projection, useful for tests/evaluation."""
        with torch.no_grad():
            return self._forward_no_grad(image, int(views))

    def adjoint_no_grad(self, sino: torch.Tensor, views: int) -> torch.Tensor:
        """Non-autograd unfiltered backprojection, useful for tests/evaluation."""
        with torch.no_grad():
            return self._adjoint_no_grad(sino, int(views))
        
    def set_adjoint_scale(self, views: int, scale: float) -> None:
        """Set multiplicative scale for A*(.) at a given view count."""
        self.adjoint_scales[int(views)] = float(scale)

    def get_adjoint_scale(self, views: int) -> float:
        """Get multiplicative scale for A*(.) at a given view count."""
        return float(self.adjoint_scales.get(int(views), 1.0))
    

    @torch.no_grad()
    def calibrate_adjoint_scale(
        self,
        views: int,
        num_trials: int = 10,
        device: str | torch.device = "cuda",
        seed: int = 0,
        image_scale: float = 0.01,
        sino_scale: float = 0.01,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """Calibrate a scalar so that <A x, y> ≈ <x, scale * BP y>.

        ASTRA FP/BP can differ by a constant discretization scale.
        This function estimates that scale using random x and y.

        Returns a dict with scale_mean, scale_std, and adjoint errors before/after scaling.
        """
        views = int(views)
        device = torch.device(device if torch.cuda.is_available() else "cpu")

        old_scale = self.get_adjoint_scale(views)
        self.set_adjoint_scale(views, 1.0)

        torch.manual_seed(seed)

        scales = []
        errs_before = []
        errs_after = []

        for i in range(num_trials):
            x = torch.randn(
                1, 1, self.image_size, self.image_size,
                device=device, dtype=torch.float32
            ) * image_scale

            y = torch.randn(
                1, 1, views, self.n_detec,
                device=device, dtype=torch.float32
            ) * sino_scale

            Ax = self._forward_no_grad(x, views)
            BPy_raw = self._adjoint_raw_no_grad(y, views)

            lhs = (Ax * y).sum()
            rhs_raw = (x * BPy_raw).sum()

            scale = lhs / (rhs_raw + 1e-12)
            scales.append(float(scale.detach().cpu()))

            err_before = torch.abs(lhs - rhs_raw) / (
                torch.abs(lhs) + torch.abs(rhs_raw) + 1e-12
            )
            errs_before.append(float(err_before.detach().cpu()))

        scale_mean = float(np.mean(scales))
        scale_std = float(np.std(scales))

        # Check after applying mean scale
        self.set_adjoint_scale(views, scale_mean)

        torch.manual_seed(seed)

        for i in range(num_trials):
            x = torch.randn(
                1, 1, self.image_size, self.image_size,
                device=device, dtype=torch.float32
            ) * image_scale

            y = torch.randn(
                1, 1, views, self.n_detec,
                device=device, dtype=torch.float32
            ) * sino_scale

            Ax = self._forward_no_grad(x, views)
            BPy_scaled = self._adjoint_no_grad(y, views)

            lhs = (Ax * y).sum()
            rhs_scaled = (x * BPy_scaled).sum()

            err_after = torch.abs(lhs - rhs_scaled) / (
                torch.abs(lhs) + torch.abs(rhs_scaled) + 1e-12
            )
            errs_after.append(float(err_after.detach().cpu()))

        result = {
            "views": float(views),
            "scale_mean": scale_mean,
            "scale_std": scale_std,
            "scale_cv": float(scale_std / (abs(scale_mean) + 1e-12)),
            "err_before_mean": float(np.mean(errs_before)),
            "err_before_std": float(np.std(errs_before)),
            "err_after_mean": float(np.mean(errs_after)),
            "err_after_std": float(np.std(errs_after)),
        }

        if verbose:
            print(f"[Adjoint scale calibration] V={views}")
            print(f"  scale mean = {result['scale_mean']:.8e}")
            print(f"  scale std  = {result['scale_std']:.8e}")
            print(f"  scale cv   = {result['scale_cv']:.8e}")
            print(f"  err before = {result['err_before_mean']:.8e} ± {result['err_before_std']:.8e}")
            print(f"  err after  = {result['err_after_mean']:.8e} ± {result['err_after_std']:.8e}")

        return result
"""Loss functions for VVBP model training.

Provides pixel-wise MSE and high-frequency spatial losses (Sobel / Laplacian)
that require 2D spatial layout.

All 2D losses expect input shape [B, 1, H, W] or [H, W] (B,H,W >= 2).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional


# ---------------------------------------------------------------------------
# Kernel builders
# ---------------------------------------------------------------------------
def _sobel_kernels(device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:
    """Return (Kx, Ky) Sobel 3×3 kernels as [1,1,3,3] tensors."""
    kx = torch.tensor(
        [[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)
    ky = torch.tensor(
        [[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)
    return kx, ky


def _laplacian_kernel(device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return Laplacian 3×3 kernel as [1,1,3,3]."""
    return torch.tensor(
        [[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]],
        device=device, dtype=dtype,
    ).view(1, 1, 3, 3)


# ---------------------------------------------------------------------------
# Shape helpers
# ---------------------------------------------------------------------------
def _to_bchw(x: Tensor) -> Tensor:
    """Normalise to [B, 1, H, W]."""
    if x.ndim == 2:
        return x[None, None, :, :]
    if x.ndim == 3:
        return x[:, None, :, :]
    if x.ndim == 4:
        return x
    raise ValueError(f"Expected 2D/3D/4D tensor, got {x.ndim}D: {tuple(x.shape)}")


# ---------------------------------------------------------------------------
# Pixel-wise losses
# ---------------------------------------------------------------------------
def mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """Mean squared error, works on [N,1], [N], or [B,1,H,W]."""
    return F.mse_loss(pred, target)


# ---------------------------------------------------------------------------
# High-frequency spatial losses  (require [B,1,H,W])
# ---------------------------------------------------------------------------
def gradient_loss_2d(
    pred_2d: Tensor,
    target_2d: Tensor,
    operator: str = "sobel",
) -> Tensor:
    """L1 loss between gradient magnitudes of pred and target.

    Args:
        pred_2d, target_2d: [B, 1, H, W] or [H, W].
        operator: "sobel" (default).

    Returns:
        scalar loss.
    """
    pred_4d = _to_bchw(pred_2d)
    target_4d = _to_bchw(target_2d)
    B, C, H, W = pred_4d.shape

    if H < 3 or W < 3:
        # Patch too small for Sobel — fall back to identity
        return F.l1_loss(pred_4d, target_4d)

    device = pred_4d.device
    dtype = pred_4d.dtype
    kx, ky = _sobel_kernels(device, dtype)

    grad_x_pred = F.conv2d(pred_4d, kx, padding=1)
    grad_y_pred = F.conv2d(pred_4d, ky, padding=1)
    grad_x_target = F.conv2d(target_4d, kx, padding=1)
    grad_y_target = F.conv2d(target_4d, ky, padding=1)

    loss_x = F.l1_loss(grad_x_pred, grad_x_target)
    loss_y = F.l1_loss(grad_y_pred, grad_y_target)
    return loss_x + loss_y


def laplacian_loss_2d(
    pred_2d: Tensor,
    target_2d: Tensor,
    operator: str = "laplacian_3x3",
) -> Tensor:
    """L1 loss between Laplacian responses of pred and target.

    Args:
        pred_2d, target_2d: [B, 1, H, W] or [H, W].
        operator: "laplacian_3x3" (default).

    Returns:
        scalar loss.
    """
    pred_4d = _to_bchw(pred_2d)
    target_4d = _to_bchw(target_2d)
    B, C, H, W = pred_4d.shape

    if H < 3 or W < 3:
        return F.l1_loss(pred_4d, target_4d)

    device = pred_4d.device
    dtype = pred_4d.dtype
    klap = _laplacian_kernel(device, dtype)

    lap_pred = F.conv2d(pred_4d, klap, padding=1)
    lap_target = F.conv2d(target_4d, klap, padding=1)

    return F.l1_loss(lap_pred, lap_target)


# ---------------------------------------------------------------------------
# Combined loss builder
# ---------------------------------------------------------------------------
def compute_total_loss(
    pred_2d: Tensor,
    target_2d: Tensor,
    lambda_grad: float = 0.0,
    lambda_lap: float = 0.0,
) -> dict:
    """Compute MSE + optional gradient / laplacian losses on 2D grids.

    Args:
        pred_2d, target_2d: [B, 1, H, W] or [H, W] tensors.
        lambda_grad: weight for gradient loss  (0 = disabled).
        lambda_lap:  weight for laplacian loss  (0 = disabled).

    Returns:
        dict with keys: loss_total, loss_img, loss_grad, loss_lap.
    """
    pred_4d = _to_bchw(pred_2d)
    target_4d = _to_bchw(target_2d)

    loss_img = F.mse_loss(pred_4d, target_4d)
    loss_total = loss_img

    loss_grad = torch.tensor(0.0, device=pred_4d.device)
    loss_lap = torch.tensor(0.0, device=pred_4d.device)

    if lambda_grad > 0:
        loss_grad = gradient_loss_2d(pred_4d, target_4d)
        loss_total = loss_total + lambda_grad * loss_grad

    if lambda_lap > 0:
        loss_lap = laplacian_loss_2d(pred_4d, target_4d)
        loss_total = loss_total + lambda_lap * loss_lap

    return {
        "loss_total": loss_total,
        "loss_img": loss_img,
        "loss_grad": loss_grad,
        "loss_lap": loss_lap,
    }


def build_loss_fn(loss_cfg: Optional[dict] = None):
    """Build a loss function from a config dict.

    Args:
        loss_cfg: dict with keys:
            type (str): "mse_hf" or "mse"
            lambda_grad (float, default 0)
            lambda_lap  (float, default 0)
            gradient_operator (str, default "sobel")
            laplacian_operator (str, default "laplacian_3x3")

    Returns:
        Callable (pred_2d, target_2d) -> dict.
    """
    if loss_cfg is None:
        loss_cfg = {}

    loss_type = loss_cfg.get("type", "mse")
    lambda_grad = float(loss_cfg.get("lambda_grad", 0.0))
    lambda_lap = float(loss_cfg.get("lambda_lap", 0.0))
    grad_op = loss_cfg.get("gradient_operator", "sobel")
    lap_op = loss_cfg.get("laplacian_operator", "laplacian_3x3")

    if loss_type == "mse" or (lambda_grad == 0 and lambda_lap == 0):
        def _loss_mse(pred_2d: Tensor, target_2d: Tensor) -> dict:
            p = _to_bchw(pred_2d)
            t = _to_bchw(target_2d)
            img = F.mse_loss(p, t)
            zero = torch.tensor(0.0, device=p.device)
            return {"loss_total": img, "loss_img": img,
                    "loss_grad": zero, "loss_lap": zero}
        return _loss_mse

    def _loss_hf(pred_2d: Tensor, target_2d: Tensor) -> dict:
        return compute_total_loss(pred_2d, target_2d,
                                  lambda_grad=lambda_grad,
                                  lambda_lap=lambda_lap)
    return _loss_hf

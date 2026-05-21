"""Uniform angular subsampling for sinograms."""

from __future__ import annotations

import numpy as np
import torch


def uniform_subsample_views(sino_full: torch.Tensor, V: int) -> torch.Tensor:
    """Uniformly subsample V views from a full-view sinogram.

    Args:
        sino_full: [..., N_full, D] — sinogram with N_full equally-spaced views over [0, 2π).
        V: number of sparse views to keep.

    Returns:
        sino_sparse: [..., V, D] with views uniformly spaced over [0, 2π).

    The stride is N_full // V.  When V does not divide N_full, the spacing
    is as close to uniform as integer arithmetic permits.
    """
    N_full = sino_full.shape[-2]
    if V > N_full:
        raise ValueError(f"V={V} > N_full={N_full}")
    if N_full % V == 0:
        step = N_full // V
        idx = torch.arange(0, N_full, step, device=sino_full.device)
    else:
        idx = torch.linspace(0, N_full - 1, V, device=sino_full.device).round().long()
    return sino_full[..., idx, :]


def uniform_subsample_views_np(sino_full: np.ndarray, V: int) -> np.ndarray:
    """NumPy equivalent of uniform_subsample_views."""
    N_full = sino_full.shape[-2]
    if V > N_full:
        raise ValueError(f"V={V} > N_full={N_full}")
    if N_full % V == 0:
        step = N_full // V
        idx = np.arange(0, N_full, step)
    else:
        idx = np.linspace(0, N_full - 1, V).round().astype(int)
    return sino_full[..., idx, :]

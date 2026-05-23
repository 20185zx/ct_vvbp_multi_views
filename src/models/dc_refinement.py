"""Data-consistency refinement wrapper using sparse-matrix projector.

Provides a drop-in post-processing refinement for any reconstruction method:

    x_{k+1} = x_k - eta * A^T (A x_k - y)

where A is the ``AstraSparseFanBeamProjector`` sparse projection matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.geometry.astra_sparse_projector import AstraSparseFanBeamProjector


class DCRefinement(nn.Module):
    """Data-consistency refinement using strict sparse-matrix adjoint.

    Applies ``n_steps`` gradient-descent updates on the data-consistency term:

        x <- x - eta * A^T (A x - y_measured)

    Args:
        projector: ``AstraSparseFanBeamProjector`` with matching geometry.
        n_steps: Number of DC updates.
        step_size: Gradient-descent step size for DC update.
    """

    def __init__(
        self,
        projector: AstraSparseFanBeamProjector,
        n_steps: int = 3,
        step_size: float = 1e-7,
    ):
        super().__init__()
        self.projector = projector
        self.n_steps = int(n_steps)
        self.step_size = float(step_size)

    def forward(
        self,
        image: torch.Tensor,
        sino_measured: torch.Tensor,
        views: int,
    ) -> torch.Tensor:
        """Refine ``image`` by ``n_steps`` DC updates.

        Args:
            image: [B, 1, H, W]  initial reconstruction.
            sino_measured: [B, 1, V, D]  measured (sparse-view) sinogram.
            views: number of projection views.

        Returns:
            refined: [B, 1, H, W] after DC steps.
        """
        x = image
        for _ in range(self.n_steps):
            grad = self.projector.data_consistency_gradient(x, sino_measured, views)
            x = x - self.step_size * grad
        return x

    def extra_repr(self) -> str:
        return f"n_steps={self.n_steps}, step_size={self.step_size:.2e}"

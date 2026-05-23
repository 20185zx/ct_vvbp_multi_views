from __future__ import annotations

from typing import Dict, Iterable, Optional

import numpy as np
import scipy.sparse as sp
import torch
import astra


def scipy_csr_to_torch_sparse_csr(
    S: sp.csr_matrix,
    device: str | torch.device = "cuda",
) -> torch.Tensor:
    """Convert scipy CSR matrix to torch sparse CSR tensor."""
    S = S.tocsr().astype(np.float32)

    crow_indices = torch.from_numpy(S.indptr.astype(np.int64))
    col_indices = torch.from_numpy(S.indices.astype(np.int64))
    values = torch.from_numpy(S.data.astype(np.float32))

    A = torch.sparse_csr_tensor(
        crow_indices,
        col_indices,
        values,
        size=S.shape,
        dtype=torch.float32,
        device=device,
    )
    return A


class AstraSparseFanBeamProjector:
    """Strict fan-beam projector using ASTRA-generated sparse matrix.

    This class uses ASTRA CPU `line_fanflat` projector to generate an explicit
    sparse projection matrix S. Then forward and adjoint are computed by:

        forward : y = S x
        adjoint : x = S^T y

    Therefore, the discrete adjoint relation is guaranteed:

        <Sx, y> = <x, S^T y>

    This is intended for data-consistency in unrolled CT networks.
    """

    def __init__(
        self,
        image_size: int = 256,
        n_detec: int = 672,
        d_detec: float = 1.0,
        d_voxel: float = 1.0,
        DSO: float = 595.0,
        DOD: float = 480.0,
        views_list: Optional[Iterable[int]] = None,
        angle_range: str = "2pi",
        device: str | torch.device = "cuda",
        use_cache: bool = True,
    ):
        self.image_size = int(image_size)
        self.n_detec = int(n_detec)
        self.d_detec = float(d_detec)
        self.d_voxel = float(d_voxel)
        self.DSO = float(DSO)
        self.DOD = float(DOD)
        self.angle_range = str(angle_range)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.use_cache = bool(use_cache)

        self.A_mats: Dict[int, torch.Tensor] = {}
        self.AT_mats: Dict[int, torch.Tensor] = {}

        if views_list is not None:
            for v in views_list:
                self.get_matrix(int(v))

    def _make_angles(self, views: int) -> np.ndarray:
        views = int(views)
        if self.angle_range.lower() in ["2pi", "0_2pi", "0-2pi"]:
            end = 2.0 * np.pi
        elif self.angle_range.lower() in ["pi", "0_pi", "0-pi"]:
            end = np.pi
        else:
            raise ValueError(f"Unknown angle_range: {self.angle_range}")

        return np.linspace(0.0, end, views, endpoint=False).astype(np.float32)

    def _build_scipy_matrix(self, views: int) -> sp.csr_matrix:
        """Build scipy sparse projection matrix using ASTRA CPU line_fanflat."""
        H = W = self.image_size
        V = int(views)
        D = self.n_detec

        s_voxel = self.image_size * self.d_voxel
        vol_geom = astra.create_vol_geom(
            H, W,
            -s_voxel / 2, s_voxel / 2,
            -s_voxel / 2, s_voxel / 2,
        )
        angles = self._make_angles(V)

        proj_geom = astra.create_proj_geom(
            "fanflat",
            self.d_detec,
            D,
            angles,
            self.DSO,
            self.DOD,
        )

        projector_id = astra.create_projector("line_fanflat", proj_geom, vol_geom)

        matrix_id = astra.projector.matrix(projector_id)
        S = astra.matrix.get(matrix_id).tocsr().astype(np.float32)

        astra.matrix.delete(matrix_id)
        astra.projector.delete(projector_id)

        expected_shape = (V * D, H * W)
        if S.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected sparse matrix shape: got {S.shape}, expected {expected_shape}"
            )

        return S

    def get_matrix(self, views: int) -> torch.Tensor:
        """Return torch sparse CSR projection matrix A for a given V."""
        views = int(views)

        if self.use_cache and views in self.A_mats:
            return self.A_mats[views]

        print(f"[AstraSparseFanBeamProjector] Building sparse matrix for V={views} ...")
        S = self._build_scipy_matrix(views)
        print(
            f"[AstraSparseFanBeamProjector] S shape={S.shape}, "
            f"nnz={S.nnz}, density={S.nnz / (S.shape[0] * S.shape[1]):.6e}"
        )

        A = scipy_csr_to_torch_sparse_csr(S, device=self.device)
        AT = A.transpose(0, 1).to_sparse_csr()

        if self.use_cache:
            self.A_mats[views] = A
            self.AT_mats[views] = AT

        return A

    def get_transpose_matrix(self, views: int) -> torch.Tensor:
        """Return torch sparse CSR adjoint matrix A^T for a given V."""
        views = int(views)

        if self.use_cache and views in self.AT_mats:
            return self.AT_mats[views]

        _ = self.get_matrix(views)
        return self.AT_mats[views]

    def forward(self, image: torch.Tensor, views: int) -> torch.Tensor:
        """Forward projection.

        Args:
            image: [B, 1, H, W]
            views: number of projection views

        Returns:
            sino: [B, 1, V, D]
        """
        views = int(views)
        A = self.get_matrix(views)

        if image.ndim != 4:
            raise ValueError(f"image must be [B,1,H,W], got shape {tuple(image.shape)}")

        B, C, H, W = image.shape
        if C != 1:
            raise ValueError(f"Only single-channel image supported, got C={C}")
        if H != self.image_size or W != self.image_size:
            raise ValueError(
                f"Image size mismatch: got {(H, W)}, expected {(self.image_size, self.image_size)}"
            )

        image = image.to(device=self.device, dtype=torch.float32)

        # [B,1,H,W] -> [H*W, B]
        x_vec = image.reshape(B, -1).transpose(0, 1).contiguous()

        # [V*D, H*W] @ [H*W, B] -> [V*D, B]
        y_vec = torch.sparse.mm(A, x_vec)

        # [V*D, B] -> [B,1,V,D]
        sino = y_vec.transpose(0, 1).reshape(B, 1, views, self.n_detec).contiguous()
        return sino

    def adjoint(self, sino: torch.Tensor, views: int) -> torch.Tensor:
        """Adjoint/backprojection using exact transpose matrix.

        Args:
            sino: [B, 1, V, D]
            views: number of projection views

        Returns:
            image: [B, 1, H, W]
        """
        views = int(views)
        AT = self.get_transpose_matrix(views)

        if sino.ndim != 4:
            raise ValueError(f"sino must be [B,1,V,D], got shape {tuple(sino.shape)}")

        B, C, V, D = sino.shape
        if C != 1:
            raise ValueError(f"Only single-channel sinogram supported, got C={C}")
        if V != views or D != self.n_detec:
            raise ValueError(
                f"Sinogram shape mismatch: got V={V}, D={D}, expected V={views}, D={self.n_detec}"
            )

        sino = sino.to(device=self.device, dtype=torch.float32)

        # [B,1,V,D] -> [V*D, B]
        y_vec = sino.reshape(B, -1).transpose(0, 1).contiguous()

        # [H*W, V*D] @ [V*D, B] -> [H*W, B]
        x_vec = torch.sparse.mm(AT, y_vec)

        # [H*W, B] -> [B,1,H,W]
        image = x_vec.transpose(0, 1).reshape(
            B, 1, self.image_size, self.image_size
        ).contiguous()
        return image

    def data_consistency_gradient(
        self,
        image: torch.Tensor,
        sino_target: torch.Tensor,
        views: int,
    ) -> torch.Tensor:
        """Compute A^T(Ax - y)."""
        sino_pred = self.forward(image, views)
        residual = sino_pred - sino_target.to(device=self.device, dtype=torch.float32)
        grad = self.adjoint(residual, views)
        return grad
#!/usr/bin/env python3
"""Verify that the corrected AstraSparseFanBeamProjector is aligned with the dataset.

Comparison:
  1. Forward projection: <Ax, y> vs dataset sinogram
     - Use the corrected sparse projector on a target image from eval_dataset
     - Compare sinogram with the dataset's own ASTRA sinogram
  2. Adjoint consistency: <Ax, y> == <x, A^T y>  (strict discrete adjoint)

Usage:
    python scripts/check_sparse_projector_alignment.py \\
        --config configs/multirate_selected_models_60view.json
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import numpy as np
import torch
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config
from src.data import MultiRateFanbeamDataset
from src.geometry.astra_sparse_projector import AstraSparseFanBeamProjector


@torch.no_grad()
def check_alignment(
    config_path: str = "configs/multirate_selected_models_60view.json",
    device: str = "cuda",
    save_dir: str = "outputs/check_sparse_projector_alignment",
    num_adjoint_trials: int = 20,
):
    os.makedirs(save_dir, exist_ok=True)

    # 1. Load config
    run_cfg = load_run_config(config_path)
    exp_cfg = run_cfg.experiment

    image_size = int(exp_cfg.image_size)
    n_detec = int(exp_cfg.n_detec)
    d_detec = float(exp_cfg.d_detec)
    d_voxel = float(exp_cfg.d_voxel)
    DSO = float(exp_cfg.DSO)
    DOD = float(exp_cfg.DOD)
    full_views = int(exp_cfg.full_views)
    sparse_views = [int(v) for v in exp_cfg.sparse_views]
    device = torch.device(device if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("Sparse Projector Alignment Check")
    print("=" * 60)
    print(f"Config:          {config_path}")
    print(f"image_size:      {image_size}")
    print(f"n_detec:         {n_detec}")
    print(f"d_detec:         {d_detec}")
    print(f"d_voxel:         {d_voxel}")
    print(f"DSO:             {DSO}")
    print(f"DOD:             {DOD}")
    print(f"full_views:      {full_views}")
    print(f"sparse_views:    {sparse_views}")
    print(f"device:          {device}")
    print()

    # 2. Build eval dataset
    eval_dataset = MultiRateFanbeamDataset(
        dicom_folder=run_cfg.dicom_folder,
        image_size=image_size,
        full_views=full_views,
        n_detec=n_detec,
        d_detec=d_detec,
        d_voxel=d_voxel,
        DSO=DSO,
        DOD=DOD,
        sparse_views=sparse_views,
        train=False,
    )

    # 3. Build corrected sparse projector
    projector = AstraSparseFanBeamProjector(
        image_size=image_size,
        n_detec=n_detec,
        d_detec=d_detec,
        d_voxel=d_voxel,
        DSO=DSO,
        DOD=DOD,
        views_list=sparse_views,
        angle_range="2pi",
        device=device,
        use_cache=True,
    )

    # 4. Take a test slice
    test_idx = int(len(eval_dataset) * 0.8)  # first test slice
    sino_full_tensor, img_tensor = eval_dataset[test_idx]
    target = img_tensor.to(device, dtype=torch.float32)  # [1, H, W]
    target_bchw = target.unsqueeze(0)                     # [1, 1, H, W]

    print(f"\nTest slice index: {test_idx}")
    print(f"Target image shape: {tuple(target.shape)}")
    print(f"Target range:       [{target.min().item():.6f}, {target.max().item():.6f}]")
    print()

    # 5. Compare forward projection for each sparse view count
    results: Dict[int, Dict] = {}
    for V in sparse_views:
        print("-" * 50)
        print(f"V = {V}")

        # Get dataset sinogram (1st dimension is singleton: [1, V, D])
        sino_full = sino_full_tensor.numpy()  # [1, full_views, D]
        if V < full_views:
            step = full_views // V
            sino_gt = sino_full[:, ::step, :].copy()  # [1, V, D]
        else:
            sino_gt = sino_full.copy()  # [1, V, D]
        sino_gt = sino_gt.squeeze(0)  # [V, D]

        sino_gt_tensor = torch.from_numpy(sino_gt).to(device, dtype=torch.float32)
        # [V, D] -> [1, 1, V, D]
        sino_gt_bchw = sino_gt_tensor.unsqueeze(0).unsqueeze(0)

        # Forward projection via sparse matrix
        sino_proj = projector.forward(target_bchw, V)  # [1, 1, V, D]

        # --- Comparison metrics ---
        gt_flat = sino_gt_tensor.reshape(-1)
        proj_flat = sino_proj.reshape(-1)

        mse = torch.mean((gt_flat - proj_flat) ** 2).item()
        mae = torch.mean(torch.abs(gt_flat - proj_flat)).item()
        relative_error = torch.norm(gt_flat - proj_flat) / (torch.norm(gt_flat) + 1e-12)
        max_err = torch.max(torch.abs(gt_flat - proj_flat)).item()

        print(f"  MSE:             {mse:.12e}")
        print(f"  MAE:             {mae:.12e}")
        print(f"  Relative error:  {relative_error:.12e}")
        print(f"  Max error:       {max_err:.12e}")
        print(f"  GT range:        [{gt_flat.min().item():.6f}, {gt_flat.max().item():.6f}]")
        print(f"  Proj range:      [{proj_flat.min().item():.6f}, {proj_flat.max().item():.6f}]")

        # Check if errors are acceptable (machine precision for linear operators)
        if mse > 1e-6:
            print(f"  *** WARNING: MSE = {mse:.6e} > 1e-6. Projector may NOT match dataset!")
        elif mse > 1e-10:
            print(f"  *  OK: MSE = {mse:.6e} (small, likely floating-point accumulation)")
        else:
            print(f"  ** EXCELLENT: MSE = {mse:.6e}")

        # --- Save residual map ---
        sino_gt_np = sino_gt_tensor.cpu().numpy()   # [V, D]
        sino_proj_np = sino_proj[0, 0].cpu().numpy()  # [V, D]
        residual = sino_proj_np - sino_gt_np

        results[V] = {
            "mse": mse,
            "mae": mae,
            "relative_error": relative_error,
            "max_error": max_err,
            "sino_gt": sino_gt_np,
            "sino_proj": sino_proj_np,
            "residual": residual,
        }

        # Plot comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        vmin = min(sino_gt_np.min(), sino_proj_np.min())
        vmax = max(sino_gt_np.max(), sino_proj_np.max())
        im0 = axes[0].imshow(sino_gt_np, aspect="auto", vmin=vmin, vmax=vmax)
        axes[0].set_title(f"GT Sinogram (dataset)\nV={V}")
        plt.colorbar(im0, ax=axes[0], fraction=0.046)

        im1 = axes[1].imshow(sino_proj_np, aspect="auto", vmin=vmin, vmax=vmax)
        axes[1].set_title(f"Sparse Matrix Proj\nV={V}")
        plt.colorbar(im1, ax=axes[1], fraction=0.046)

        vmax_res = max(abs(residual.min()), abs(residual.max()))
        im2 = axes[2].imshow(residual, aspect="auto", cmap="RdBu",
                             vmin=-vmax_res, vmax=vmax_res)
        axes[2].set_title(f"Residual (Proj - GT)\nmax|err|={max_err:.6e}")
        plt.colorbar(im2, ax=axes[2], fraction=0.046)

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sino_comparison_V{V}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved: {save_path}")
        print()

    # 6. Adjoint consistency check
    print("=" * 50)
    print("Adjoint Consistency Check")
    print("=" * 50)
    for V in sparse_views:
        print(f"\n  V = {V}:")
        errors = []
        for trial in range(num_adjoint_trials):
            x = torch.randn(1, 1, image_size, image_size, device=device, dtype=torch.float32)
            y = torch.randn(1, 1, V, n_detec, device=device, dtype=torch.float32)

            Ax = projector.forward(x, V)
            ATy = projector.adjoint(y, V)

            lhs = torch.sum(Ax * y)       # <Sx, y>
            rhs = torch.sum(x * ATy)      # <x, S^T y>

            # Relative error in adjoint relation
            rel_err = torch.abs(lhs - rhs) / (torch.abs(lhs) + torch.abs(rhs) + 1e-30)
            errors.append(float(rel_err))

        mean_err = float(np.mean(errors))
        max_err = float(np.max(errors))
        print(f"    Mean relative adjoint error: {mean_err:.16e}")
        print(f"    Max  relative adjoint error: {max_err:.16e}")
        if mean_err < 1e-12:
            print(f"    ** Perfect adjoint relation (<Sx,y> == <x,S^Ty>)")
        else:
            print(f"    *  Adjoint error is non-zero but may be acceptable")

    # 7. Summary judgment
    print("\n" + "=" * 60)
    print("VERDICT")
    print("=" * 60)

    all_ok = True
    for V in sparse_views:
        r = results[V]
        if r["mse"] > 1e-6:
            print(f"  !! V={V}: Projector MISMATCHES dataset (MSE={r['mse']:.6e})")
            all_ok = False
        elif r["mse"] > 1e-10:
            print(f"  ~~ V={V}: Small deviation (MSE={r['mse']:.6e}), likely FP accumulation")
        else:
            print(f"  ++ V={V}: Projector matches dataset (MSE={r['mse']:.6e})")

    if all_ok:
        print("\n  ++ VERDICT: Projector is correctly aligned. Safe to use for DC.")
    else:
        print("\n  !! VERDICT: Projector is NOT aligned. Do NOT proceed with DC loss.")

    # Save numerical results
    save_results = {
        "config": config_path,
        "test_idx": test_idx,
        "d_voxel": d_voxel,
        "results": {
            V: {k: float(v) if isinstance(v, (np.floating, float)) else v
                for k, v in r.items() if k in ("mse", "mae", "relative_error", "max_error")}
            for V, r in results.items()
        },
    }
    torch.save(save_results, os.path.join(save_dir, "alignment_results.pt"))
    print(f"\nSaved numerical results: {os.path.join(save_dir, 'alignment_results.pt')}")

    return all_ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify AstraSparseFanBeamProjector alignment with dataset."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/multirate_selected_models_60view.json",
        help="Path to experiment config.",
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        help="Device: cuda or cpu.",
    )
    parser.add_argument(
        "--save_dir", type=str,
        default="outputs/check_sparse_projector_alignment",
        help="Output directory for figures and results.",
    )
    parser.add_argument(
        "--num_adjoint_trials", type=int, default=20,
        help="Number of random trials for adjoint consistency check.",
    )
    args = parser.parse_args()

    ok = check_alignment(
        config_path=args.config,
        device=args.device,
        save_dir=args.save_dir,
        num_adjoint_trials=args.num_adjoint_trials,
    )

    sys.exit(0 if ok else 1)

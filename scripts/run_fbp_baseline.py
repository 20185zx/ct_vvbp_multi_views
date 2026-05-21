"""
FBP baseline for multi-rate sparse-view fan-beam CT (AAPM setup).

Evaluates FBP reconstruction at sparse_views ∈ {9, 18, 36, 72} using
uniform angular subsampling from a 720-view full sinogram.

Geometry:
    - image_size = 256
    - detector_elements = 672
    - DSD = 1075 mm (DSO=595, DOD=480)
    - fan-beam, uniform angles over [0, 2π)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.dicom_dataset import MultiRateFanbeamDataset
from src.data.subsample import uniform_subsample_views_np
from src.geometry.fanbeam import build_linfbp_geo, load_or_generate_geo
from src.geometry.fbp import LInFBPFixedLinearFBPBatch
from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_images


FULL_VIEWS = 720
SPARSE_VIEWS = [9, 18, 36, 72]
IMAGE_SIZE = 256
N_DETEC = 672
D_DETEC = 1.0
D_VOXEL = 1.0
DSO = 595.0
DOD = 480.0
WATER_MU = 0.0192


def build_fbp_reconstructor(views: int, results_folder: str, device: torch.device):
    """Build a fixed FBP reconstructor for a given number of views."""
    geo = load_or_generate_geo(
        views=views,
        results_folder=results_folder,
        device=device,
        image_size=IMAGE_SIZE,
        n_detec=N_DETEC,
        d_detec=D_DETEC,
        d_voxel=D_VOXEL,
        DSO=DSO,
        DOD=DOD,
    )
    return LInFBPFixedLinearFBPBatch(geo).to(device), geo


@torch.no_grad()
def reconstruct_fbp(
    sino_sparse: torch.Tensor,
    fbp: LInFBPFixedLinearFBPBatch,
    device: torch.device,
) -> np.ndarray:
    """Run FBP on a sparse-view sinogram, return numpy array."""
    sino = sino_sparse.unsqueeze(0).to(device)  # [1, 1, V, D]
    recon = fbp(sino)
    return recon[0, 0].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(
        description="FBP baseline for multi-rate sparse-view fan-beam CT"
    )
    parser.add_argument(
        "--dicom_folder",
        type=str,
        default=None,
        help="Path to DICOM folder (default: auto-detect from project).",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="Results",
        help="Folder for geometry index cache files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/fbp_baseline_multirate",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--slice_idx",
        type=int,
        default=None,
        help="DICOM slice index to evaluate (default: 80%% train split).",
    )
    parser.add_argument(
        "--sparse_views",
        type=int,
        nargs="+",
        default=SPARSE_VIEWS,
        help="Sparse view counts to evaluate.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Discover DICOM folder ---
    dicom_folder = args.dicom_folder
    if dicom_folder is None:
        candidate = os.path.join(PROJECT_ROOT, "full_1mm", "L067", "full_1mm")
        if os.path.isdir(candidate):
            dicom_folder = candidate
        else:
            raise FileNotFoundError(
                "DICOM folder not found. Provide --dicom_folder."
            )
    print(f"DICOM folder: {dicom_folder}")

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.results_folder, exist_ok=True)

    # --- Load dataset (eval mode: returns full 720-view sinogram) ---
    print("\n=== Loading dataset ===")
    dataset = MultiRateFanbeamDataset(
        dicom_folder=dicom_folder,
        image_size=IMAGE_SIZE,
        full_views=FULL_VIEWS,
        n_detec=N_DETEC,
        d_detec=D_DETEC,
        d_voxel=D_VOXEL,
        DSO=DSO,
        DOD=DOD,
        sparse_views=args.sparse_views,
        train=False,
    )

    # Use the first test slice.
    n_slices = len(dataset)
    split = int(0.8 * n_slices)
    if args.slice_idx is not None:
        test_idx = args.slice_idx
    else:
        test_idx = split  # first test slice
    print(f"Evaluating slice index: {test_idx} / {n_slices}")

    sino_full_tensor, img_target_tensor = dataset[test_idx]
    # sino_full_tensor: [1, 720, 672]
    # img_target_tensor: [1, 256, 256]
    sino_full = sino_full_tensor.squeeze(0).numpy()  # [720, 672]
    target = img_target_tensor.squeeze(0).numpy()     # [256, 256]

    print(f"Full sinogram shape: {tuple(sino_full.shape)}")
    print(f"Target image shape:  {tuple(target.shape)}")
    print(f"Target range: [{target.min():.6f}, {target.max():.6f}]")

    # --- Pre-build all FBP reconstructors ---
    fbp_recons = {}
    geo_dicts = {}
    for V in args.sparse_views:
        print(f"\n=== Building FBP reconstructor for V={V} ===")
        fbp, geo = build_fbp_reconstructor(V, args.results_folder, device)
        fbp.eval()
        fbp_recons[V] = fbp
        geo_dicts[V] = geo

    # --- Evaluate each sparse view count ---
    all_metrics = {}
    all_recons = {}
    all_titles = [
        "Target",
    ]
    all_images = [
        target,
    ]

    print("\n" + "=" * 70)
    print("FBP BASELINE — MULTI-RATE SPARSE-VIEW EVALUATION")
    print("=" * 70)

    for V in args.sparse_views:
        print(f"\n--- V = {V} views ---")
        t0 = time.time()

        sino_sparse = uniform_subsample_views_np(sino_full, V)  # [V, 672]
        sino_tensor = torch.from_numpy(sino_sparse.astype(np.float32)).unsqueeze(0)

        recon = reconstruct_fbp(sino_tensor, fbp_recons[V], device)

        metrics = compute_metrics_np(recon, target)
        all_metrics[V] = metrics
        all_recons[V] = recon

        elapsed = time.time() - t0
        print(f"  PSNR: {metrics['PSNR']:.4f} dB")
        print(f"  SSIM: {metrics['SSIM']:.6f}")
        print(f"  MSE:  {metrics['MSE']:.8f}")
        print(f"  MAE:  {metrics['MAE']:.8f}")
        print(f"  Time: {elapsed:.2f}s")

        all_images.append(recon)
        all_titles.append(f"FBP {V}-view\nPSNR={metrics['PSNR']:.2f} dB")

    # --- Save metrics ---
    metrics_df = pd.DataFrame(all_metrics).T
    metrics_df.index.name = "views"
    metrics_path = os.path.join(args.output_dir, "fbp_multirate_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(f"\nSaved metrics: {metrics_path}")
    print(metrics_df[["PSNR", "SSIM"]])

    # --- Save comparison figure ---
    fig_path = os.path.join(args.output_dir, "fbp_multirate_comparison.png")
    plot_comparison_images(all_images, all_titles, save_path=fig_path, show=False)
    print(f"Saved figure: {fig_path}")

    # --- Save individual reconstructions ---
    recons_dict = {
        "target": target,
        "sino_full": sino_full,
        "sparse_views": args.sparse_views,
        **{f"recon_{V}": all_recons[V] for V in args.sparse_views},
        "all_metrics": all_metrics,
    }
    recons_path = os.path.join(args.output_dir, "fbp_multirate_reconstructions.pt")
    torch.save(recons_dict, recons_path)
    print(f"Saved reconstructions: {recons_path}")

    # --- Summary table ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for V in args.sparse_views:
        m = all_metrics[V]
        print(f"  V={V:3d}  |  PSNR={m['PSNR']:.4f} dB  |  SSIM={m['SSIM']:.6f}")

    return all_metrics, all_recons


if __name__ == "__main__":
    main()

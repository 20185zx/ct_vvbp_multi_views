#!/usr/bin/env python3
"""Evaluate original LR Integral MLP with data-consistency refinement.

Pipeline:
  1. Load pre-trained LR Integral MLP checkpoint.
  2. For each test slice and each V ∈ sparse_views:
     a. Full-image MLP prediction (VVBP → pixel-wise MLP, assembled).
     b. Optional: DC refinement via A^T(Ax - y) using corrected sparse projector.
     c. Crop region, compute PSNR/SSIM.
  3. Report metrics with/without DC.

Usage:
    python scripts/run_lr_integral_dc.py --config configs/multirate_selected_models_60view_local_rank_integral_dc.json
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config, save_run_config
from src.experiments.project_setup import prepare_multirate_context
from src.geometry import LInFBPFixedLinearFBPBatch
from src.geometry.astra_sparse_projector import AstraSparseFanBeamProjector
from src.models import LocalRankCenterIntegralMLPNet, DCRefinement
from src.data.local_vvbp import gather_sorted_vvbp_patch
from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_images, plot_comparison_grid
from src.training import estimate_multirate_stats


@torch.no_grad()
def predict_full_image_mlp(
    model: torch.nn.Module,
    vvbp: torch.Tensor,
    stats: dict,
    image_size: int,
    chunk_size: int = 8192,
    patch_size: int = 3,
    device: str = "cuda",
) -> torch.Tensor:
    """Run MLP on every non-border pixel to produce a full image.

    The border of width ``patch_size // 2`` is zero-filled.
    """
    r = patch_size // 2
    H, W = image_size, image_size

    full_pred = torch.zeros(H, W, device=device, dtype=torch.float32)

    # Generate all pixel coordinates within the valid interior
    xs = torch.arange(r, H - r, device=device)
    ys = torch.arange(r, W - r, device=device)
    yy, xx = torch.meshgrid(ys, xs, indexing="xy")  # [H-2r, W-2r] each
    xs_all = xx.reshape(-1)  # [N]
    ys_all = yy.reshape(-1)  # [N]

    n_total = xs_all.numel()

    for start in range(0, n_total, chunk_size):
        end = min(start + chunk_size, n_total)
        chunk_xs = xs_all[start:end]
        chunk_ys = ys_all[start:end]

        values_sorted = gather_sorted_vvbp_patch(
            vvbp, chunk_xs, chunk_ys, patch_size=patch_size, mode="3x3",
        )  # [1, P, J, K]
        P = chunk_xs.numel()
        values_sorted = values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3])

        pred_norm = model(values_sorted, stats)
        pred = pred_norm * stats["target_std"] + stats["target_mean"]

        # Place into full image
        for i in range(P):
            full_pred[chunk_xs[i].long(), chunk_ys[i].long()] = pred[i, 0]

    return full_pred.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]


@torch.no_grad()
def evaluate_slice(
    model,
    dc_refine: DCRefinement | None,
    sino_full: np.ndarray,
    target: np.ndarray,
    extractor,
    v_stats,
    target_stats,
    V: int,
    full_views: int,
    region: tuple,
    image_size: int,
    chunk_size: int,
    patch_size: int,
    device: str,
):
    """Evaluate one slice at one view count."""
    x0, x1, y0, y1 = region

    # Subsample sinogram
    step = full_views // V
    sino_sparse = sino_full[::step, :].copy()
    sino_tensor = torch.from_numpy(sino_sparse).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,V,D]

    # --- VVBP extraction ---
    vvbp = extractor(sino_tensor)  # [1, 1, H, W, V]

    # Build stats dict
    vs = v_stats[V]
    batch_stats = {
        "v_mean": vs["v_mean"].to(device),
        "v_std": vs["v_std"].to(device),
        "target_mean": target_stats["target_mean"].to(device),
        "target_std": target_stats["target_std"].to(device),
    }

    # --- Full-image MLP prediction ---
    mlp_full = predict_full_image_mlp(
        model, vvbp, batch_stats,
        image_size=image_size,
        chunk_size=chunk_size,
        patch_size=patch_size,
        device=device,
    )  # [1, 1, H, W]

    # --- DC refinement ---
    if dc_refine is not None:
        dc_full = dc_refine(mlp_full, sino_tensor, V)
    else:
        dc_full = mlp_full

    # Crop region
    mlp_region = mlp_full[0, 0, x0:x1, y0:y1].cpu().numpy()
    dc_region = dc_full[0, 0, x0:x1, y0:y1].cpu().numpy()
    target_region = target[x0:x1, y0:y1]

    mlp_metrics = compute_metrics_np(mlp_region, target_region)
    dc_metrics = compute_metrics_np(dc_region, target_region) if dc_refine is not None else mlp_metrics

    return {
        "mlp_region": mlp_region,
        "dc_region": dc_region,
        "mlp_full": mlp_full[0, 0].cpu().numpy(),
        "dc_full": dc_full[0, 0].cpu().numpy(),
        "target_region": target_region,
        "target_full": target,
        "mlp_metrics": mlp_metrics,
        "dc_metrics": dc_metrics,
    }


def main():
    parser = argparse.ArgumentParser(
        description="LR Integral MLP + data-consistency refinement."
    )
    parser.add_argument(
        "--config", type=str,
        default="configs/multirate_selected_models_60view_local_rank_integral_dc.json",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained MLP checkpoint. Default: uses save_dir from config.")
    parser.add_argument("--dc_steps", type=int, default=None,
                        help="Override DC steps from config.")
    parser.add_argument("--dc_step_size", type=float, default=None,
                        help="Override DC step size from config.")
    parser.add_argument("--num_test_slices", type=int, default=5,
                        help="Number of test slices to evaluate.")
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment
    exp_cfg.ensure_dirs()
    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Context ---
    ctx = prepare_multirate_context(
        dicom_folder=run_cfg.dicom_folder,
        results_folder=run_cfg.results_folder,
        save_dir=exp_cfg.save_dir,
        cache_dir=exp_cfg.cache_dir,
        cfg=exp_cfg,
        device=device,
        seed=exp_cfg.seed,
    )
    geo_dict = ctx["geo_dict"]
    extractors = ctx["extractors"]
    eval_dataset = ctx["eval_dataset"]
    test_indices = ctx["test_indices"]

    sparse_views = [int(v) for v in exp_cfg.sparse_views]
    full_views = int(exp_cfg.full_views)
    image_size = int(exp_cfg.image_size)
    region = tuple(run_cfg.region)
    chunk_size = int(getattr(exp_cfg, "chunk_size_eval", 8192))
    patch_size = int(getattr(exp_cfg, "patch_size", 3))

    # DC config
    dc_cfg = getattr(exp_cfg, "dc_refinement", {})
    use_dc = bool(dc_cfg.get("enabled", False))
    dc_steps = args.dc_steps if args.dc_steps is not None else int(dc_cfg.get("n_steps", 5))
    dc_step_size = args.dc_step_size if args.dc_step_size is not None else float(dc_cfg.get("step_size", 1e-7))

    print(f"\nDC refinement: enabled={use_dc}, steps={dc_steps}, step_size={dc_step_size:.2e}")

    # --- Sparse projector (with d_voxel!) ---
    projector = AstraSparseFanBeamProjector(
        image_size=image_size,
        n_detec=int(exp_cfg.n_detec),
        d_detec=float(exp_cfg.d_detec),
        d_voxel=float(exp_cfg.d_voxel),
        DSO=float(exp_cfg.DSO),
        DOD=float(exp_cfg.DOD),
        views_list=sparse_views if use_dc else None,
        device=device,
        use_cache=True,
    )
    dc_refine = DCRefinement(projector, n_steps=dc_steps, step_size=dc_step_size) if use_dc else None

    # --- Load model ---
    model = LocalRankCenterIntegralMLPNet().to(device)
    # Default: load from the original (non-DC) multirate_vvbp save_dir
    original_save_dir = exp_cfg.save_dir.replace("_dc", "")
    ckpt_path = args.checkpoint or os.path.join(
        original_save_dir, "local_rank_center_integral_mlp_10_epochs.pt"
    )
    print(f"Loading checkpoint: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()
    print("Model loaded successfully.")

    # --- Estimate normalization stats ---
    print("\nEstimating normalization statistics from training data ...")
    target_stats_raw, v_stats_raw = estimate_multirate_stats(
        ctx["train_loader"], ctx["extractors"], exp_cfg, device,
        num_stats_batches=4,
    )
    target_stats = {k: v.to(device) for k, v in target_stats_raw.items()}
    v_stats = {
        V: {k: v.to(device) for k, v in s.items()}
        for V, s in v_stats_raw.items()
    }

    # --- Evaluate test slices ---
    all_rows = []
    n_test = min(args.num_test_slices, len(test_indices))

    for slice_idx in range(n_test):
        test_idx = int(test_indices[slice_idx])
        print(f"\n{'=' * 60}")
        print(f"Test slice {slice_idx + 1}/{n_test} (index {test_idx})")

        sino_full_tensor, img_tensor = eval_dataset[test_idx]
        sino_full = sino_full_tensor.squeeze(0).numpy()  # [full_views, D]
        target = img_tensor.squeeze(0).numpy()            # [H, W]

        for V in sparse_views:
            print(f"\n  V = {V}")
            t0 = time.time()

            out = evaluate_slice(
                model=model,
                dc_refine=dc_refine,
                sino_full=sino_full,
                target=target,
                extractor=extractors[V],
                v_stats=v_stats,
                target_stats=target_stats,
                V=V,
                full_views=full_views,
                region=region,
                image_size=image_size,
                chunk_size=chunk_size,
                patch_size=patch_size,
                device=device,
            )
            elapsed = time.time() - t0

            print(f"    MLP: PSNR={out['mlp_metrics']['PSNR']:.4f}  SSIM={out['mlp_metrics']['SSIM']:.6f}")
            if use_dc:
                gain_psnr = out['dc_metrics']['PSNR'] - out['mlp_metrics']['PSNR']
                print(f"    DC : PSNR={out['dc_metrics']['PSNR']:.4f}  SSIM={out['dc_metrics']['SSIM']:.6f}  "
                      f"gain={gain_psnr:+.4f} dB")
            print(f"    time: {elapsed:.1f}s")

            row = {
                "test_slice": slice_idx,
                "test_idx": test_idx,
                "V": V,
                "MLP_PSNR": out['mlp_metrics']['PSNR'],
                "MLP_SSIM": out['mlp_metrics']['SSIM'],
                "MLP_MAE": out['mlp_metrics']['MAE'],
                "MLP_MSE": out['mlp_metrics']['MSE'],
            }
            if use_dc:
                row.update({
                    "DC_PSNR": out['dc_metrics']['PSNR'],
                    "DC_SSIM": out['dc_metrics']['SSIM'],
                    "DC_MAE": out['dc_metrics']['MAE'],
                    "DC_MSE": out['dc_metrics']['MSE'],
                    "DC_PSNR_gain": out['dc_metrics']['PSNR'] - out['mlp_metrics']['PSNR'],
                    "DC_SSIM_gain": out['dc_metrics']['SSIM'] - out['mlp_metrics']['SSIM'],
                })
            all_rows.append(row)

    # --- Summary ---
    metrics_df = pd.DataFrame(all_rows)
    metrics_path = os.path.join(exp_cfg.save_dir, "lr_integral_dc_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"\nSaved metrics: {metrics_path}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    avg_cols = ["MLP_PSNR", "MLP_SSIM"]
    if use_dc:
        avg_cols += ["DC_PSNR", "DC_SSIM", "DC_PSNR_gain", "DC_SSIM_gain"]
    print(metrics_df[["V"] + avg_cols].groupby("V").mean())
    print(f"\nFull metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()

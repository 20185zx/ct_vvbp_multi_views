"""Shared FBP baseline computation for multi-rate sparse-view CT evaluation."""

from __future__ import annotations

import torch

from src.evaluation.metrics import compute_metrics_np
from src.geometry.fbp import LInFBPFixedLinearFBPBatch


@torch.no_grad()
def compute_fbp_baselines(eval_dataset, geo_dict, sparse_views, test_idx, region, device):
    """Precompute FBP baselines once for all sparse-view counts.

    FBP is a deterministic reconstruction method, so its results are
    independent of any trained model and only need to be computed once.

    Returns:
        fbp_metrics: dict V -> {MSE, MAE, PSNR, SSIM} (region-level).
        fbp_preds_region: dict V -> ndarray (cropped to region).
        full_fbp_preds: dict V -> ndarray (full image reconstruction).
    """
    sino_full_tensor, img_tensor = eval_dataset[test_idx]
    sino_full = sino_full_tensor.squeeze(0)  # [N_full, D]
    target = img_tensor.squeeze(0).numpy()
    x0, x1, y0, y1 = region
    target_region = target[x0:x1, y0:y1]

    fbp_metrics = {}
    fbp_preds_region = {}
    full_fbp_preds = {}

    for V in sparse_views:
        print(f"[FBP BASELINE] V={V}")
        step = sino_full.shape[0] // int(V)
        sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)
        fbp = LInFBPFixedLinearFBPBatch(geo_dict[int(V)]).to(device).eval()
        fbp_img = fbp(sino_sparse)[0, 0].detach().cpu().numpy()
        fbp_region = fbp_img[x0:x1, y0:y1]
        fbp_metrics[V] = compute_metrics_np(fbp_region, target_region)
        fbp_preds_region[V] = fbp_region
        full_fbp_preds[V] = fbp_img
        print(f"  FBP PSNR={fbp_metrics[V]['PSNR']:.4f} dB SSIM={fbp_metrics[V]['SSIM']:.6f}")

    return {
        "fbp_metrics": fbp_metrics,
        "fbp_preds_region": fbp_preds_region,
        "full_fbp_preds": full_fbp_preds,
    }

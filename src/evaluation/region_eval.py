"""Region-level model prediction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.cached_dataset import CachedSortedVVBPDataset
from src.data.feature_builder import make_model_features_from_values
from src.data.local_rank import compute_local_rank_closed_integral
from src.data.local_vvbp import gather_sorted_vvbp_patch
from src.evaluation.metrics import compute_metrics_np


compute_local_rank_closed_from_values = compute_local_rank_closed_integral


@torch.no_grad()
def predict_region_from_cache(
    model,
    region_cache,
    stats,
    batch_size: int = 8192,
    patch_size: int = 3,
    device: str = "cuda",
):
    """Predict a rectangular evaluation region from a cached VVBP patch tensor."""
    model.eval()
    dataset = CachedSortedVVBPDataset(region_cache)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    preds = []
    for values_sorted, *_ in loader:
        values_sorted = values_sorted.to(device, non_blocking=True)

        if getattr(model, "input_mode", "features") == "values_sorted":
            pred_norm = model(values_sorted, stats)
        else:
            features = make_model_features_from_values(
                values_sorted=values_sorted,
                stats=stats,
                use_coord=model.use_coord,
                patch_size=patch_size,
            )
            pred_norm = model(features)

        pred = pred_norm * stats["target_std"] + stats["target_mean"]
        preds.append(pred.cpu())

    h_reg = region_cache["metadata"]["Hreg"]
    w_reg = region_cache["metadata"]["Wreg"]

    return {
        "target": region_cache["target"].numpy().reshape(h_reg, w_reg),
        "center_base": region_cache["center_base"].numpy().reshape(h_reg, w_reg),
        "local_3x3_base": region_cache["local_3x3_base"].numpy().reshape(h_reg, w_reg),
        "pred": torch.cat(preds, dim=0).numpy().reshape(h_reg, w_reg),
    }


@torch.no_grad()
def evaluate_multirate(
    model,
    eval_dataset,
    extractors,
    geo_dict,
    target_stats,
    v_stats,
    sparse_views,
    test_idx,
    region,
    patch_size=3,
    chunk_size=8192,
    device="cuda",
):
    """Evaluate model + parameter-free baselines at each V on a fixed test region.

    For each V:
      1. Subsample full 720-view sinogram → V views.
      2. Extract VVBP with the V-specific extractor.
      3. Gather 3×3 VVBP patches for every pixel in the region.
      4. Compute:
         - Model prediction (normalized → denormalized).
         - Center base: pi * mean of center pixel's V VVBP values.
         - Local-rank closed: closed-interval non-uniform trapezoid integral.
      5. Compute PSNR/SSIM against ground truth.

    Returns:
        model_metrics_df: DataFrame indexed by V with model MSE, MAE, PSNR, SSIM.
        baseline_metrics: dict {"Center base": {V: {metrics}}, "Local-rank closed": {V: {metrics}}}.
        preds: dict V → model pred ndarray.
        baseline_preds: dict {"Center base": {V: ndarray}, "Local-rank closed": {V: ndarray}}.
        target_arr: ground truth ndarray for the region.
    """
    model.eval()

    sino_full_tensor, img_tensor = eval_dataset[test_idx]
    sino_full = sino_full_tensor.squeeze(0)  # [720, D]
    target_full = img_tensor.squeeze(0)       # [H, W]

    target_mean = target_stats["target_mean"].to(device)
    target_std = target_stats["target_std"].to(device)

    x_start, x_end, y_start, y_end = region
    Hreg = x_end - x_start
    Wreg = y_end - y_start

    coords = [(x, y) for x in range(x_start, x_end) for y in range(y_start, y_end)]
    xs_all = torch.tensor([c[0] for c in coords], dtype=torch.long)
    ys_all = torch.tensor([c[1] for c in coords], dtype=torch.long)

    target_region = target_full[x_start:x_end, y_start:y_end].reshape(-1, 1)

    # Model outputs
    model_metrics = {}
    model_preds = {}

    # Baseline outputs
    baseline_preds = {"Local-rank closed": {}}
    baseline_metrics = {"Local-rank closed": {}}

    for V in sparse_views:
        print(f"\n[EVAL] V={V} ...")

        N_full = sino_full.shape[0]
        step = N_full // V
        sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)  # [1, 1, V, D]

        extractor = extractors[V]
        vvbp = extractor(sino_sparse)  # [1, 1, H, W, V]

        vs = v_stats[V]
        batch_stats = {
            "target_mean": target_mean,
            "target_std": target_std,
            "v_mean": vs["v_mean"].to(device),
            "v_std": vs["v_std"].to(device),
        }

        model_chunks = []
        local_rank_closed_chunks = []

        for start in range(0, len(coords), chunk_size):
            chunk_xs = xs_all[start : start + chunk_size].to(device)
            chunk_ys = ys_all[start : start + chunk_size].to(device)

            values_sorted = gather_sorted_vvbp_patch(
                vvbp, chunk_xs, chunk_ys, patch_size=patch_size, mode="3x3",
            )  # [1, P, J, K]
            P = chunk_xs.numel()
            values_sorted = values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3])

            # --- Model prediction ---
            if getattr(model, "input_mode", "features") == "values_sorted":
                pred_norm = model(values_sorted, batch_stats)
            else:
                features = make_model_features_from_values(
                    values_sorted=values_sorted,
                    stats=batch_stats,
                    use_coord=model.use_coord,
                    patch_size=patch_size,
                )
                pred_norm = model(features)

            pred = pred_norm * target_std + target_mean
            model_chunks.append(pred.cpu())

            # --- Parameter-free baseline ---
            local_rank_closed_chunks.append(
                compute_local_rank_closed_from_values(values_sorted).cpu()
            )

        # Concatenate chunks → region images
        model_pred = torch.cat(model_chunks, dim=0).numpy().reshape(Hreg, Wreg)
        local_rank_closed_pred = torch.cat(local_rank_closed_chunks, dim=0).numpy().reshape(Hreg, Wreg)
        target_arr = target_region.numpy().reshape(Hreg, Wreg)

        # Metrics
        model_metrics[V] = compute_metrics_np(model_pred, target_arr)
        baseline_metrics["Local-rank closed"][V] = compute_metrics_np(
            local_rank_closed_pred, target_arr
        )

        model_preds[V] = model_pred
        baseline_preds["Local-rank closed"][V] = local_rank_closed_pred

        print(f"  Model PSNR={model_metrics[V]['PSNR']:.4f} dB  SSIM={model_metrics[V]['SSIM']:.6f}")
        print(f"  Local-rank closed PSNR={baseline_metrics['Local-rank closed'][V]['PSNR']:.4f} dB  "
              f"SSIM={baseline_metrics['Local-rank closed'][V]['SSIM']:.6f}")

    model_metrics_df = pd.DataFrame(model_metrics).T
    model_metrics_df.index.name = "V"

    return model_metrics_df, baseline_metrics, model_preds, baseline_preds, target_arr

#!/usr/bin/env python3
"""Train original local-rank center integral MLP with training-time DC loss.

L_total = L_img + lambda_dc * L_dc_norm

L_img = MSE(pred_region, target_region)
L_dc_norm = MSE( (S(full_pred) - sino_sparse) / sino_std )

where:
  - pred_region is the MLP prediction on region [128:384, 128:384]
  - full_pred = target_full.detach() outside region, pred_region inside region
  - S is the AstraSparseFanBeamProjector (verified aligned with dataset)
  - sino_std = sinogram std from the current batch, for scale normalization

Usage:
    python scripts/run_local_rank_integral_dc_loss.py --config configs/multirate_selected_models_60view_local_rank_integral_dc_loss.json --device cuda
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config, save_run_config
from src.experiments.project_setup import prepare_multirate_context
from src.geometry.astra_sparse_projector import AstraSparseFanBeamProjector
from src.geometry import LInFBPFixedLinearFBPBatch
from src.models import LocalRankCenterIntegralMLPNet
from src.data.local_vvbp import gather_sorted_vvbp_patch
from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_grid
from src.training import estimate_multirate_stats


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _build_region_coords(region, device="cpu"):
    x0, x1, y0, y1 = region
    coords = [(x, y) for x in range(x0, x1) for y in range(y0, y1)]
    xs = torch.tensor([c[0] for c in coords], dtype=torch.long, device=device)
    ys = torch.tensor([c[1] for c in coords], dtype=torch.long, device=device)
    return xs, ys  # each [N_reg]


@torch.no_grad()
def sanity_projector_alignment(projector, target_full, sino_sparse, V, eps=1e-12):
    """Verify projector matches dataset:  MSE(S(target), sino)  should be ~1e-12."""
    sino_test = projector.forward(target_full, V)
    mse = F.mse_loss(sino_test, sino_sparse).item()
    mae = F.l1_loss(sino_test, sino_sparse).item()
    return mse, mae, sino_test


def predict_region(model, vvbp, stats, xs_all, ys_all, chunk_size, patch_size=3):
    """Run MLP on all region pixels (chunked). Returns [1,1,Hreg,Wreg].

    Keeps the computation graph for gradient flow during training / sanity checks.
    """
    all_preds = []
    N = xs_all.numel()
    for start in range(0, N, chunk_size):
        end = min(start + chunk_size, N)
        chunk_xs = xs_all[start:end]
        chunk_ys = ys_all[start:end]
        values_sorted = gather_sorted_vvbp_patch(
            vvbp, chunk_xs, chunk_ys, patch_size=patch_size, mode="3x3",
        )
        P = chunk_xs.numel()
        values_sorted = values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3])
        pred_norm = model(values_sorted, stats)
        pred = pred_norm * stats["target_std"] + stats["target_mean"]
        all_preds.append(pred)
    preds = torch.cat(all_preds, dim=0)  # [N, 1]
    Hreg = int(np.sqrt(N))
    Wreg = Hreg
    return preds.reshape(1, 1, Hreg, Wreg)


def build_full_pred(pred_region, target_full, region, device):
    """Compose full image: target_full.detach() outside region, pred_region inside.

    Uses a binary mask so gradient flows only through the region.
    """
    x0, x1, y0, y1 = region
    H, W = target_full.shape[2], target_full.shape[3]

    pred_padded = F.pad(pred_region, [y0, W - y1, x0, H - x1], mode="constant", value=0.0)

    mask = torch.zeros(1, 1, H, W, device=device)
    mask[:, :, x0:x1, y0:y1] = 1.0

    full_pred = target_full.detach() * (1.0 - mask) + pred_padded * mask
    return full_pred


# ---------------------------------------------------------------------------
# train one epoch
# ---------------------------------------------------------------------------

def train_one_epoch(
    model, train_loader, extractors, v_stats, target_stats,
    optimizer, projector, dc_cfg,
    xs_all, ys_all, region,
    chunk_size, patch_size, device, epoch,
):
    model.train()

    total_L_img_norm = 0.0
    total_L_img_phys = 0.0
    total_L_dc_raw = 0.0
    total_L_dc_norm = 0.0
    total_L_total = 0.0
    n_batches = 0

    target_mean = target_stats["target_mean"]
    target_std = target_stats["target_std"]
    lambda_dc = float(dc_cfg["lambda_dc"])
    eps = float(dc_cfg.get("eps", 1e-8))
    use_dc = bool(dc_cfg.get("enabled", False))

    first_batch_printed = False

    t0 = time.time()

    for sino_batch, img_batch in train_loader:
        sino_sparse = sino_batch.to(device, non_blocking=True)
        target_full = img_batch.to(device, non_blocking=True)
        V = int(sino_sparse.shape[2])

        target_region = target_full[:, :, region[0]:region[1], region[2]:region[3]]

        # --- VVBP extraction ---
        extractor = extractors[V]
        vvbp = extractor(sino_sparse)  # [1, 1, H, W, V]

        vs = v_stats[V]
        batch_stats = {
            "v_mean": vs["v_mean"].to(device),
            "v_std": vs["v_std"].to(device),
            "target_mean": target_mean,
            "target_std": target_std,
        }

        # --- Chunked MLP forward over full region ---
        # Collect pred_norm (normalized) for L_img, matching original train_multirate_model.
        # Denormalize only for DC loss.
        all_pred_norms = []
        Hreg = region[1] - region[0]
        Wreg = region[3] - region[2]
        N = xs_all.numel()
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk_xs = xs_all[start:end]
            chunk_ys = ys_all[start:end]
            values_sorted = gather_sorted_vvbp_patch(
                vvbp, chunk_xs, chunk_ys, patch_size=patch_size, mode="3x3",
            )
            P = chunk_xs.numel()
            values_sorted = values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3])
            pred_norm = model(values_sorted, batch_stats)
            all_pred_norms.append(pred_norm)

        pred_norm_region = torch.cat(all_pred_norms, dim=0).reshape(1, 1, Hreg, Wreg)

        # --- L_img (normalized space — matches original train_multirate_model) ---
        target_region_norm = (target_region - target_mean) / target_std
        L_img_norm = F.mse_loss(pred_norm_region, target_region_norm)

        # --- DC loss (physical space) ---
        if use_dc:
            pred_phys_region = pred_norm_region * target_std + target_mean
            full_pred = build_full_pred(pred_phys_region, target_full, region, device)
            sino_pred = projector.forward(full_pred, V)

            sino_std = sino_sparse.detach().std().clamp_min(eps)
            L_dc_raw = F.mse_loss(sino_pred, sino_sparse)
            residual_norm = (sino_pred - sino_sparse) / sino_std
            L_dc_norm = (residual_norm ** 2).mean()

            L_total = L_img_norm + lambda_dc * L_dc_norm
        else:
            L_dc_raw = torch.tensor(0.0, device=device)
            L_dc_norm = torch.tensor(0.0, device=device)
            L_total = L_img_norm

        # --- Diagnostic: first batch only, print both spaces ---
        if not first_batch_printed:
            with torch.no_grad():
                L_img_phys_diag = F.mse_loss(
                    pred_norm_region * target_std + target_mean, target_region
                )
                L_img_norm_diag = L_img_norm.item()
                ratio = L_img_norm_diag / (L_img_phys_diag.item() + eps)
                print(f"\n  [DIAGNOSTIC first batch epoch {epoch}]")
                print(f"    target_mean={float(target_mean):.8f}  target_std={float(target_std):.8f}")
                print(f"    pred_norm_region  min={float(pred_norm_region.min()):.6f}  max={float(pred_norm_region.max()):.6f}  "
                      f"mean={float(pred_norm_region.mean()):.6f}  std={float(pred_norm_region.std()):.6f}")
                print(f"    target_norm_region min={float(target_region_norm.min()):.6f}  max={float(target_region_norm.max()):.6f}  "
                      f"mean={float(target_region_norm.mean()):.6f}  std={float(target_region_norm.std()):.6f}")
                pred_phys_diag = pred_norm_region * target_std + target_mean
                print(f"    pred_phys_region   min={float(pred_phys_diag.min()):.8f}  max={float(pred_phys_diag.max()):.8f}  "
                      f"mean={float(pred_phys_diag.mean()):.8f}  std={float(pred_phys_diag.std()):.8f}")
                print(f"    target_phys_region min={float(target_region.min()):.8f}  max={float(target_region.max()):.8f}  "
                      f"mean={float(target_region.mean()):.8f}  std={float(target_region.std()):.8f}")
                print(f"    L_img_norm    = {L_img_norm_diag:.8e}")
                print(f"    L_img_phys    = {L_img_phys_diag.item():.8e}")
                print(f"    expected_ratio = 1/target_std^2 = {1.0/(float(target_std)**2):.2f}")
                print(f"    actual_ratio   = L_img_norm / L_img_phys = {ratio:.2f}")
                print(f"  [END DIAGNOSTIC]\n")
                first_batch_printed = True

        # --- Backward ---
        optimizer.zero_grad(set_to_none=True)
        L_total.backward()
        optimizer.step()

        total_L_img_norm += float(L_img_norm.detach())
        total_L_img_phys += float(F.mse_loss(pred_norm_region * target_std + target_mean, target_region).detach())
        total_L_dc_raw += float(L_dc_raw.detach()) if use_dc else 0.0
        total_L_dc_norm += float(L_dc_norm.detach()) if use_dc else 0.0
        total_L_total += float(L_total.detach())
        n_batches += 1

    avg_img_norm = total_L_img_norm / max(n_batches, 1)
    avg_img_phys = total_L_img_phys / max(n_batches, 1)
    avg_dc_raw = total_L_dc_raw / max(n_batches, 1)
    avg_dc_norm = total_L_dc_norm / max(n_batches, 1)
    avg_total = total_L_total / max(n_batches, 1)
    dc_ratio = (lambda_dc * avg_dc_norm) / (avg_img_norm + eps) if use_dc else 0.0
    elapsed = time.time() - t0

    print(
        f"Epoch {epoch:03d} | "
        f"L_img_norm={avg_img_norm:.6e}  "
        f"L_img_phys={avg_img_phys:.6e}  "
        f"L_dc_raw={avg_dc_raw:.6e}  "
        f"L_dc_norm={avg_dc_norm:.6e}  "
        f"lambda*L_dc_norm={lambda_dc * avg_dc_norm:.6e}  "
        f"dc_ratio={dc_ratio:.4f}  "
        f"L_total={avg_total:.6e}  "
        f"time={elapsed:.1f}s"
    )

    return {
        "L_img_norm": avg_img_norm,
        "L_img_phys": avg_img_phys,
        "L_dc_raw": avg_dc_raw,
        "L_dc_norm": avg_dc_norm,
        "lambda_dc_norm": lambda_dc * avg_dc_norm,
        "dc_ratio": dc_ratio,
        "L_total": avg_total,
        "seconds": elapsed,
    }


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_one_slice(
    model, eval_dataset, extractors, geo_dict, v_stats, target_stats,
    projector, dc_cfg,
    sparse_views, region, test_idx,
    chunk_size, patch_size, device,
):
    model.eval()
    sino_full_tensor, img_tensor = eval_dataset[test_idx]
    sino_full = sino_full_tensor.squeeze(0)  # [full_views, D]
    target_full = img_tensor.to(device, dtype=torch.float32)  # [1, H, W]
    target_bchw = target_full.unsqueeze(0)  # [1, 1, H, W]

    x0, x1, y0, y1 = region
    target_region_np = target_full[0, x0:x1, y0:y1].cpu().numpy()

    xs_all, ys_all = _build_region_coords(region, device=device)

    results = {}
    for V in sparse_views:
        step = sino_full.shape[0] // V
        sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)

        extractor = extractors[V]
        vvbp = extractor(sino_sparse)

        vs = v_stats[V]
        batch_stats = {
            "v_mean": vs["v_mean"],
            "v_std": vs["v_std"],
            "target_mean": target_stats["target_mean"],
            "target_std": target_stats["target_std"],
        }

        # MLP prediction
        pred_region = predict_region(
            model, vvbp, batch_stats,
            xs_all, ys_all, chunk_size, patch_size,
        )
        pred_region_np = pred_region[0, 0].cpu().numpy()

        # FBP baseline
        fbp = LInFBPFixedLinearFBPBatch(geo_dict[V]).to(device).eval()
        fbp_img = fbp(sino_sparse)[0, 0].cpu().numpy()
        fbp_region = fbp_img[x0:x1, y0:y1]

        # Projection metrics
        full_pred = build_full_pred(pred_region, target_bchw, region, device)
        sino_pred = projector.forward(full_pred, V)
        proj_mse = F.mse_loss(sino_pred, sino_sparse).item()
        proj_mae = F.l1_loss(sino_pred, sino_sparse).item()
        proj_residual = sino_pred - sino_sparse
        proj_res_mean = float(proj_residual.mean())
        proj_res_std = float(proj_residual.std())

        # Image metrics
        img_metrics = compute_metrics_np(pred_region_np, target_region_np)
        fbp_metrics = compute_metrics_np(fbp_region, target_region_np)

        results[V] = {
            "pred_region": pred_region_np,
            "fbp_region": fbp_region,
            "target_region": target_region_np,
            "img_metrics": img_metrics,
            "fbp_metrics": fbp_metrics,
            "proj_mse": proj_mse,
            "proj_mae": proj_mae,
            "proj_res_mean": proj_res_mean,
            "proj_res_std": proj_res_std,
            "sino_pred": sino_pred[0, 0].cpu().numpy(),
            "sino_gt": sino_sparse[0, 0].cpu().numpy(),
            "sino_residual": proj_residual[0, 0].cpu().numpy(),
        }

        print(f"  V={V}: PSNR={img_metrics['PSNR']:.4f} SSIM={img_metrics['SSIM']:.6f} | "
              f"FBP PSNR={fbp_metrics['PSNR']:.4f} | "
              f"proj MSE={proj_mse:.6e} res_mean={proj_res_mean:.4e}")

    return results


def save_eval_figures(results, target, sparse_views, save_dir, epoch=None):
    """Save image comparison (standard plot_comparison_grid) + projection comparison."""
    epoch_tag = f"_epoch{epoch:03d}" if epoch is not None else ""

    # Build preds_by_method / psnr_by_method for plot_comparison_grid
    preds_by_method = {"FBP": {}, "MLP": {}}
    psnr_by_method = {"FBP": {}, "MLP": {}}
    for V in sparse_views:
        r = results[V]
        preds_by_method["FBP"][V] = r["fbp_region"]
        preds_by_method["MLP"][V] = r["pred_region"]
        psnr_by_method["FBP"][V] = r["fbp_metrics"]["PSNR"]
        psnr_by_method["MLP"][V] = r["img_metrics"]["PSNR"]

    # Image comparison (standard grid style)
    fig_path = os.path.join(save_dir, f"comparison_grid{epoch_tag}.png")
    plot_comparison_grid(
        target=target,
        preds_by_method=preds_by_method,
        psnr_by_method=psnr_by_method,
        col_labels=["FBP", "MLP"],
        sparse_views=sparse_views,
        save_path=fig_path,
        show=False,
    )
    plt.close("all")

    # Projection comparison (unique to DC loss script)
    for V in sparse_views:
        r = results[V]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        vmin = min(r["sino_gt"].min(), r["sino_pred"].min())
        vmax = max(r["sino_gt"].max(), r["sino_pred"].max())
        im0 = axes[0].imshow(r["sino_gt"], aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        axes[0].set_title("sino_sparse (GT)")
        im1 = axes[1].imshow(r["sino_pred"], aspect="auto", cmap="gray", vmin=vmin, vmax=vmax)
        axes[1].set_title("sino_pred")
        vmax_res = max(abs(r["sino_residual"].min()), abs(r["sino_residual"].max()))
        im2 = axes[2].imshow(r["sino_residual"], aspect="auto", cmap="RdBu",
                             vmin=-vmax_res, vmax=vmax_res)
        axes[2].set_title(f"residual  mean={r['proj_res_mean']:.4e}")
        for ax in axes:
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"projection_V{V}{epoch_tag}.png"),
                    dpi=450, bbox_inches="tight")
        plt.close()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train LR Integral MLP with training-time DC loss."
    )
    parser.add_argument("--config", type=str,
                        default="configs/multirate_selected_models_60view_local_rank_integral_dc_loss.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max_train_batches", type=int, default=None,
                        help="Limit train batches per epoch (debug).")
    parser.add_argument("--skip_sanity", action="store_true",
                        help="Skip pre-training sanity checks.")
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment
    if args.epochs is not None:
        exp_cfg.num_epochs = int(args.epochs)
    exp_cfg.ensure_dirs()
    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Read data_consistency from raw JSON (not filtered by ExperimentConfig dataclass)
    _raw = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dc_cfg = dict(_raw.get("experiment", {}).get("data_consistency", {}))
    use_dc = bool(dc_cfg.get("enabled", False))
    lambda_dc = float(dc_cfg.get("lambda_dc", 1e-4))
    eps = float(dc_cfg.get("eps", 1e-8))

    print("=" * 60)
    print("Training LR Integral MLP with DC loss")
    print("=" * 60)
    print(f"DC enabled:    {use_dc}")
    print(f"lambda_dc:     {lambda_dc}")
    print(f"Config:        {args.config}")
    print(f"Save dir:      {exp_cfg.save_dir}")
    print()

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
    train_loader = ctx["train_loader"]
    eval_dataset = ctx["eval_dataset"]
    test_indices = ctx["test_indices"]

    sparse_views = [int(v) for v in exp_cfg.sparse_views]
    full_views = int(exp_cfg.full_views)
    image_size = int(exp_cfg.image_size)
    region = tuple(run_cfg.region)
    chunk_size_train = int(getattr(exp_cfg, "chunk_size_train", 8192))
    chunk_size_eval = int(getattr(exp_cfg, "chunk_size_eval", 8192))
    patch_size = int(getattr(exp_cfg, "patch_size", 3))
    num_epochs = int(exp_cfg.num_epochs)

    # --- Region coordinates ---
    xs_all, ys_all = _build_region_coords(region, device=device)
    n_reg = xs_all.numel()
    print(f"Region pixels:  {n_reg} ({region[1]-region[0]}x{region[3]-region[2]})")

    # --- Stats ---
    print("\nEstimating normalization stats ...")
    target_stats_raw, v_stats_raw = estimate_multirate_stats(
        train_loader, extractors, exp_cfg, device, num_stats_batches=4,
    )
    target_stats = {k: v.to(device) for k, v in target_stats_raw.items()}
    v_stats = {V: {k: v.to(device) for k, v in s.items()} for V, s in v_stats_raw.items()}

    # --- Model ---
    model = LocalRankCenterIntegralMLPNet().to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=exp_cfg.lr,
        weight_decay=exp_cfg.weight_decay,
    )

    # --- Sparse projector ---
    projector = None
    if use_dc:
        projector = AstraSparseFanBeamProjector(
            image_size=image_size,
            n_detec=int(exp_cfg.n_detec),
            d_detec=float(exp_cfg.d_detec),
            d_voxel=float(exp_cfg.d_voxel),
            DSO=float(exp_cfg.DSO),
            DOD=float(exp_cfg.DOD),
            views_list=sparse_views,
            device=device,
            use_cache=True,
        )

    # --- Sanity checks ---
    if not args.skip_sanity and use_dc:
        print("\n" + "=" * 60)
        print("SANITY CHECK 1: Projector alignment")
        print("=" * 60)
        sino_full_tensor, img_tensor = eval_dataset[test_indices[0]]
        sino_full = sino_full_tensor.squeeze(0)
        target_full = img_tensor.to(device, dtype=torch.float32)
        target_bchw = target_full.unsqueeze(0)

        for V in sparse_views:
            step = full_views // V
            sino_sparse = sino_full[::step, :].unsqueeze(0).unsqueeze(0).to(device)
            mse, mae, sino_test = sanity_projector_alignment(
                projector, target_bchw, sino_sparse, V,
            )
            print(f"  V={V}: S(target) vs sino_sparse: MSE={mse:.6e}  MAE={mae:.6e}")
            if mse > 1e-6:
                print(f"  *** WARNING: MSE={mse:.6e} > 1e-6. Projector may not be aligned!")
            elif mse > 1e-10:
                print(f"  *  OK (floating-point accumulation)")
            else:
                print(f"  ++ EXCELLENT alignment")

        print("\n" + "=" * 60)
        print("SANITY CHECK 2: Gradient flow")
        print("=" * 60)
        model.train()
        sino_sparse_test = sino_sparse.clone().detach().requires_grad_(False)
        target_full_test = target_bchw.clone().detach().requires_grad_(False)

        x0, x1, y0, y1 = region

        extractor = extractors[V]
        vvbp = extractor(sino_sparse_test)
        vs = v_stats[V]
        batch_stats = {
            "v_mean": vs["v_mean"],
            "v_std": vs["v_std"],
            "target_mean": target_stats["target_mean"],
            "target_std": target_stats["target_std"],
        }

        # Full-region chunked forward (raw pred_norm, matching training loop)
        Hreg = region[1] - region[0]
        Wreg = region[3] - region[2]
        pred_norm_chunks = []
        N = xs_all.numel()
        for start in range(0, N, chunk_size_eval):
            end = min(start + chunk_size_eval, N)
            chunk_xs = xs_all[start:end]
            chunk_ys = ys_all[start:end]
            values_sorted = gather_sorted_vvbp_patch(
                vvbp, chunk_xs, chunk_ys, patch_size=patch_size, mode="3x3",
            )
            P = chunk_xs.numel()
            values_sorted = values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3])
            pred_norm_chunks.append(model(values_sorted, batch_stats))
        pred_norm_test = torch.cat(pred_norm_chunks, dim=0).reshape(1, 1, Hreg, Wreg)
        pred_phys_test = pred_norm_test * target_stats["target_std"] + target_stats["target_mean"]

        target_phys_test = target_full_test[:, :, x0:x1, y0:y1]
        target_norm_test = (target_phys_test - target_stats["target_mean"]) / target_stats["target_std"]

        L_img_norm_test = F.mse_loss(pred_norm_test, target_norm_test)
        L_img_phys_test = F.mse_loss(pred_phys_test, target_phys_test)

        full_pred_test = build_full_pred(pred_phys_test, target_full_test, region, device)
        sino_pred_test = projector.forward(full_pred_test, V)
        sino_std_test = sino_sparse_test.detach().std().clamp_min(eps)
        residual_norm_test = (sino_pred_test - sino_sparse_test) / sino_std_test
        L_dc_norm_test = (residual_norm_test ** 2).mean()

        L_total_test = L_img_norm_test + lambda_dc * L_dc_norm_test

        L_total_test.backward()

        grad_norms = {}
        has_none_grad = False
        for name, param in model.named_parameters():
            if param.grad is None:
                print(f"  !! {name}: grad is None")
                has_none_grad = True
            else:
                gn = param.grad.norm().item()
                grad_norms[name] = gn
                if gn < 1e-16:
                    print(f"  ?? {name}: grad norm = {gn:.4e} (very small)")

        if has_none_grad:
            print("  *** FAILED: some parameters have None gradient")
        else:
            print(f"  ++ All parameters have gradients")
            print(f"  grad norms: min={min(grad_norms.values()):.4e}  "
                  f"max={max(grad_norms.values()):.4e}  "
                  f"mean={np.mean(list(grad_norms.values())):.4e}")

        print(f"\n  target_mean={float(target_stats['target_mean']):.8f}  target_std={float(target_stats['target_std']):.8f}")
        print(f"  pred_norm:  min={float(pred_norm_test.min()):.6f} max={float(pred_norm_test.max()):.6f} "
              f"mean={float(pred_norm_test.mean()):.6f} std={float(pred_norm_test.std()):.6f}")
        print(f"  pred_phys:  min={float(pred_phys_test.min()):.8f} max={float(pred_phys_test.max()):.8f} "
              f"mean={float(pred_phys_test.mean()):.8f} std={float(pred_phys_test.std()):.8f}")
        print(f"  L_img_norm    = {L_img_norm_test.item():.6e}")
        print(f"  L_img_phys    = {L_img_phys_test.item():.6e}")
        print(f"  L_dc_norm     = {L_dc_norm_test.item():.6e}")
        print(f"  lambda*L_dc   = {lambda_dc * L_dc_norm_test.item():.6e}")
        print(f"  L_total = L_img_norm + lambda*L_dc_norm = {L_total_test.item():.6e}")

        if has_none_grad or min(grad_norms.values()) < 1e-16:
            print("\n  *** ABORT: gradient check failed")
            return
        else:
            print("\n  ++ Gradient check passed")

        # Reset model state
        model = LocalRankCenterIntegralMLPNet().to(device)
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=exp_cfg.lr,
            weight_decay=exp_cfg.weight_decay,
        )

        print("\n" + "=" * 60)
        print("SANITY CHECK 3: Initial loss scales (with random-init model)")
        print("=" * 60)
        Hreg = region[1] - region[0]
        Wreg = region[3] - region[2]
        model.eval()
        with torch.no_grad():
            # Get predictions in both spaces
            pred_norm_chunks = []
            N = xs_all.numel()
            for start in range(0, N, chunk_size_eval):
                end = min(start + chunk_size_eval, N)
                chunk_xs = xs_all[start:end]
                chunk_ys = ys_all[start:end]
                values_sorted = gather_sorted_vvbp_patch(
                    vvbp, chunk_xs, chunk_ys, patch_size=patch_size, mode="3x3",
                )
                P = chunk_xs.numel()
                values_sorted = values_sorted.reshape(P, values_sorted.shape[2], values_sorted.shape[3])
                pred_norm_chunks.append(model(values_sorted, batch_stats))
            pred_norm_init = torch.cat(pred_norm_chunks, dim=0).reshape(1, 1, Hreg, Wreg)
            pred_phys_init = pred_norm_init * target_stats["target_std"] + target_stats["target_mean"]

            target_phys_init = target_full_test[:, :, x0:x1, y0:y1]
            target_norm_init = (target_phys_init - target_stats["target_mean"]) / target_stats["target_std"]

            L_img_norm_init = F.mse_loss(pred_norm_init, target_norm_init).item()
            L_img_phys_init = F.mse_loss(pred_phys_init, target_phys_init).item()

            full_pred_init = build_full_pred(pred_phys_init, target_full_test, region, device)
            sino_pred_init = projector.forward(full_pred_init, V)
            sino_std_init = sino_sparse_test.std().clamp_min(eps)
            L_dc_raw_init = F.mse_loss(sino_pred_init, sino_sparse_test).item()
            residual_norm_init = (sino_pred_init - sino_sparse_test) / sino_std_init
            L_dc_norm_init = (residual_norm_init ** 2).mean().item()

        ratio_init = L_img_norm_init / (L_img_phys_init + eps)
        expected_ratio = 1.0 / (float(target_stats["target_std"]) ** 2)
        print(f"  target_mean={float(target_stats['target_mean']):.8f}  target_std={float(target_stats['target_std']):.8f}")
        print(f"  pred_norm  min={float(pred_norm_init.min()):.6f} max={float(pred_norm_init.max()):.6f} "
              f"mean={float(pred_norm_init.mean()):.6f} std={float(pred_norm_init.std()):.6f}")
        print(f"  target_norm min={float(target_norm_init.min()):.6f} max={float(target_norm_init.max()):.6f} "
              f"mean={float(target_norm_init.mean()):.6f} std={float(target_norm_init.std()):.6f}")
        print(f"  pred_phys  min={float(pred_phys_init.min()):.8f} max={float(pred_phys_init.max()):.8f} "
              f"mean={float(pred_phys_init.mean()):.8f} std={float(pred_phys_init.std()):.8f}")
        print(f"  target_phys min={float(target_phys_init.min()):.8f} max={float(target_phys_init.max()):.8f} "
              f"mean={float(target_phys_init.mean()):.8f} std={float(target_phys_init.std()):.8f}")
        print(f"  L_img_norm    = {L_img_norm_init:.6e}")
        print(f"  L_img_phys    = {L_img_phys_init:.6e}")
        print(f"  L_dc_raw      = {L_dc_raw_init:.6e}")
        print(f"  L_dc_norm     = {L_dc_norm_init:.6e}")
        print(f"  expected_ratio (1/target_std^2) = {expected_ratio:.2f}")
        print(f"  actual_ratio   = {ratio_init:.2f}")
        print(f"  lambda*L_dc_norm = {lambda_dc * L_dc_norm_init:.6e}")

        if abs(ratio_init - expected_ratio) / expected_ratio > 0.1:
            print(f"  *** WARNING: actual_ratio deviates from expected_ratio by >10%")
            print(f"  Check if pred_norm_region is already de-normalized before L_img_norm")

        if lambda_dc * L_dc_norm_init > 10.0 * L_img_norm_init:
            print(f"  *** WARNING: lambda*L_dc_norm ({lambda_dc * L_dc_norm_init:.2e}) >> L_img_norm ({L_img_norm_init:.2e})")

    # --- Training ---
    print("\n" + "=" * 60)
    print("TRAINING")
    print("=" * 60)
    print(f"Epochs: {num_epochs}")
    print(f"DC lambda: {lambda_dc}")
    print(f"Region: {region}")
    print()

    train_log = []
    global_test_idx = int(test_indices[0])
    best_psnr = -1.0
    best_ssim = -1.0
    best_epoch = 0

    for epoch in range(1, num_epochs + 1):
        info = train_one_epoch(
            model=model,
            train_loader=train_loader,
            extractors=extractors,
            v_stats=v_stats,
            target_stats=target_stats,
            optimizer=optimizer,
            projector=projector,
            dc_cfg=dc_cfg,
            xs_all=xs_all,
            ys_all=ys_all,
            region=region,
            chunk_size=chunk_size_train,
            patch_size=patch_size,
            device=device,
            epoch=epoch,
        )
        info["epoch"] = epoch
        train_log.append(info)

        # --- Epoch-end evaluation ---
        print(f"  Evaluating on test slice {global_test_idx} ...")
        model.eval()
        eval_results = evaluate_one_slice(
            model=model,
            eval_dataset=eval_dataset,
            extractors=extractors,
            geo_dict=geo_dict,
            v_stats=v_stats,
            target_stats=target_stats,
            projector=projector,
            dc_cfg=dc_cfg,
            sparse_views=sparse_views,
            region=region,
            test_idx=global_test_idx,
            chunk_size=chunk_size_eval,
            patch_size=patch_size,
            device=device,
        )
        target_region_ref = eval_results[sparse_views[0]]["target_region"]
        save_eval_figures(eval_results, target_region_ref, sparse_views, exp_cfg.save_dir, epoch=epoch)

        # Add per-V PSNR/SSIM to train log
        for V in sparse_views:
            r = eval_results[V]
            info[f"PSNR_V{V}"] = r["img_metrics"]["PSNR"]
            info[f"SSIM_V{V}"] = r["img_metrics"]["SSIM"]

        # Save best model (track PSNR on the first sparse view)
        current_psnr = eval_results[sparse_views[0]]["img_metrics"]["PSNR"]
        current_ssim = eval_results[sparse_views[0]]["img_metrics"]["SSIM"]
        if current_psnr > best_psnr:
            best_psnr = current_psnr
            best_ssim = current_ssim
            best_epoch = epoch
            model_path = os.path.join(exp_cfg.save_dir, "local_rank_center_integral_mlp_best.pt")
            torch.save(model.state_dict(), model_path)
            print(f"  ++ New best PSNR={best_psnr:.4f} SSIM={best_ssim:.6f} at epoch {best_epoch}, saved")

        # Save train log incrementally
        pd.DataFrame(train_log).to_csv(
            os.path.join(exp_cfg.save_dir, "train_log.csv"), index=False,
        )

    # --- Final evaluation ---
    print("\n" + "=" * 60)
    print("FINAL EVALUATION")
    print("=" * 60)
    final_rows = []
    for V in sparse_views:
        r = eval_results[V]
        row = {
            "V": V,
            "PSNR": r["img_metrics"]["PSNR"],
            "SSIM": r["img_metrics"]["SSIM"],
            "MSE": r["img_metrics"]["MSE"],
            "MAE": r["img_metrics"]["MAE"],
            "FBP_PSNR": r["fbp_metrics"]["PSNR"],
            "FBP_SSIM": r["fbp_metrics"]["SSIM"],
            "proj_MSE": r["proj_mse"],
            "proj_MAE": r["proj_mae"],
            "proj_res_mean": r["proj_res_mean"],
            "proj_res_std": r["proj_res_std"],
        }
        final_rows.append(row)
        print(f"  V={V}: PSNR={r['img_metrics']['PSNR']:.4f} SSIM={r['img_metrics']['SSIM']:.6f} | "
              f"FBP PSNR={r['fbp_metrics']['PSNR']:.4f} | "
              f"proj MSE={r['proj_mse']:.2e}")

    final_df = pd.DataFrame(final_rows)
    final_path = os.path.join(exp_cfg.save_dir, "final_metrics.csv")
    final_df.to_csv(final_path, index=False)
    print(f"\nSaved final metrics: {final_path}")

    # Save full eval results
    torch.save(eval_results, os.path.join(exp_cfg.save_dir, "eval_results.pt"))

    # Save final figures
    target_region_ref = eval_results[sparse_views[0]]["target_region"]
    save_eval_figures(eval_results, target_region_ref, sparse_views, exp_cfg.save_dir)

    # Save best metrics
    best_metrics = {
        "best_epoch": best_epoch,
        "best_psnr": best_psnr,
        "best_ssim": best_ssim,
        "lambda_dc": lambda_dc,
        "num_epochs": num_epochs,
    }
    import json as _json
    with open(os.path.join(exp_cfg.save_dir, "best_metrics.json"), "w") as f:
        _json.dump(best_metrics, f, indent=2)
    print(f"Saved best metrics: {os.path.join(exp_cfg.save_dir, 'best_metrics.json')}")

    print(f"\nDone. Best PSNR={best_psnr:.4f} SSIM={best_ssim:.6f} at epoch {best_epoch}/{num_epochs}")


if __name__ == "__main__":
    main()

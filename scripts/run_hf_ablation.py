"""Experiment A: high-frequency loss ablation.

Trains ``local rank center integral mlp`` with optional Sobel / Laplacian
high-frequency losses on 2D spatial patches.

The 2D patches are extracted from a region cache built for a single slice,
ensuring genuine spatial adjacency for gradient / Laplacian convolution.

Run:
    python scripts/run_hf_ablation.py \\
        --config configs/multirate_selected_models_60view_hfloss.json

    python scripts/run_hf_ablation.py \\
        --config configs/multirate_selected_models_60view_hfloss.json \\
        --baseline   # train without HF loss for comparison
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config, save_run_config, ExperimentConfig
from src.geometry import load_or_generate_geo, FanBeamVVBPExtractor
from src.models.local_rank_center_integral_mlp import LocalRankCenterIntegralMLPNet
from src.data.cache_builder import (
    build_or_load_region_cache, estimate_stats_from_train_cache,
    build_or_load_train_cache,
)
from src.data.cached_dataset import CachedSortedVVBPDataset
from src.data.dicom_dataset import build_dataloaders
from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_images
from src.training import train_direct_model_cached  # baseline
from src.training.trainer import train_direct_model_cached_hf  # HF trainer


def safe_model_name(name: str) -> str:
    return name.replace(", ", "_").replace(" ", "_")


def main():
    parser = argparse.ArgumentParser(
        description="Experiment A: HF loss ablation.",
    )
    parser.add_argument("--config", type=str,
                        default="configs/multirate_selected_models_60view_hfloss.json")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--eval_only", action="store_true",
                        help="Skip training, load and evaluate only.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Model checkpoint path for eval_only.")
    parser.add_argument("--baseline", action="store_true",
                        help="Override: train without HF loss (same as lambda_grad=0, lambda_lap=0).")
    parser.add_argument("--train_slice_idx", type=int, default=None,
                        help="Slice index for building region cache (default: first train slice).")
    parser.add_argument("--test_slice_idx", type=int, default=None,
                        help="Slice index for evaluation (default: first test slice).")
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment

    # ---- Load HF loss config ----
    loss_cfg = getattr(exp_cfg, "loss", None)
    # ExperimentConfig is a dataclass — loss is not an official field, may be in raw dict
    if loss_cfg is None:
        # Try reading from the raw JSON directly
        import json
        raw = json.loads(open(args.config, encoding="utf-8").read())
        raw_exp = raw.get("experiment", {})
        loss_cfg = raw_exp.get("loss", None)
    if loss_cfg is None:
        loss_cfg = {}
    loss_cfg = dict(loss_cfg)  # copy

    if args.baseline:
        loss_cfg["lambda_grad"] = 0.0
        loss_cfg["lambda_lap"] = 0.0
        print("[BASELINE MODE] HF loss disabled.")

    lambda_grad = float(loss_cfg.get("lambda_grad", 0.0))
    lambda_lap = float(loss_cfg.get("lambda_lap", 0.0))
    hf_patch_dim = int(getattr(exp_cfg, "hf_patch_dim", 32))
    hf_steps = int(getattr(exp_cfg, "hf_steps_per_epoch", 128))

    print(f"[HF] lambda_grad={lambda_grad}, lambda_lap={lambda_lap}")
    print(f"[HF] patch_dim={hf_patch_dim}, hf_steps/epoch={hf_steps}")
    hf_enabled = (lambda_grad > 0 or lambda_lap > 0)

    exp_cfg.ensure_dirs()
    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ---- Single-V dataset (V=60, same as build_dataloaders convention) ----
    region = tuple(run_cfg.region)

    # build_dataloaders: LInFBPAlignedDataset with views=60, iterates ALL slices
    dataset, train_indices, test_indices, train_loader = build_dataloaders(
        dicom_folder=str(run_cfg.dicom_folder or "full_1mm/L067/full_1mm"),
        cfg=exp_cfg,
    )
    print(f"Train slices: {len(train_indices)}, Test slices: {len(test_indices)}")

    # ---- Geometry & extractor ----
    V = 60  # fixed; config uses sparse_views=[60]
    geo = load_or_generate_geo(
        V, str(run_cfg.results_folder or "cache/fanbeam_geometry"),
        device,
        image_size=exp_cfg.image_size,
        n_detec=exp_cfg.n_detec,
        d_detec=exp_cfg.d_detec,
        d_voxel=exp_cfg.d_voxel,
        DSO=exp_cfg.DSO,
        DOD=exp_cfg.DOD,
    )
    extractor = FanBeamVVBPExtractor(geo).to(device).eval()

    # ---- Build DIVERSE train cache (multi-slice random pixels) ----
    # Same as original: samples random pixels from ALL training slices.
    exp_cfg.rebuild_train_cache = True
    train_cache = build_or_load_train_cache(
        loader=train_loader,
        extractor=extractor,
        cfg=exp_cfg,
        device=device,
    )
    print(f"Train cache: {train_cache['target'].shape[0]} pixels from {len(train_indices)} slices")

    # ---- Build region cache for ONE training slice (for HF 2D patches) ----
    train_slice_idx = args.train_slice_idx if args.train_slice_idx is not None else train_indices[0]
    test_slice_idx = args.test_slice_idx if args.test_slice_idx is not None else test_indices[0]
    print(f"HF region cache from train slice: {train_slice_idx}")
    print(f"Eval region cache from test slice: {test_slice_idx}")

    orig_cache_dir = exp_cfg.cache_dir
    hf_cache_dir = os.path.join(exp_cfg.save_dir, "hf_region_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    exp_cfg.cache_dir = hf_cache_dir

    hf_region_cache = build_or_load_region_cache(
        dataset=dataset,
        extractor=extractor,
        global_idx=train_slice_idx,
        region_name="hf_train_region",
        region=region,
        cfg=exp_cfg,
        device=device,
    )
    hf_N = hf_region_cache["target"].shape[0]
    print(f"HF region cache: {hf_N} pixels")

    # Build eval region cache
    eval_region_cache = build_or_load_region_cache(
        dataset=dataset,
        extractor=extractor,
        global_idx=test_slice_idx,
        region_name="eval_region",
        region=region,
        cfg=exp_cfg,
        device=device,
    )
    exp_cfg.cache_dir = orig_cache_dir  # restore

    # ---- Stats ----
    stats = estimate_stats_from_train_cache(train_cache, device=device)

    # ---- Model ----
    model_name = run_cfg.model_names[0] if run_cfg.model_names else "local rank center integral mlp, 10 epochs"
    model = LocalRankCenterIntegralMLPNet().to(device)

    if args.eval_only:
        ckpt_path = args.checkpoint or os.path.join(exp_cfg.save_dir,
                                                      f"{safe_model_name(model_name)}.pt")
        print(f"Loading checkpoint: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        # ---- Training ----
        # Pixel-wise MSE: uses DIVERSE train cache (many slices, random pixels)
        # 2D HF patches:   uses hf_region_cache (one slice, spatially coherent)
        train_dataset = CachedSortedVVBPDataset(train_cache)
        train_cached_loader = DataLoader(
            train_dataset,
            batch_size=exp_cfg.cached_batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
        )

        print(f"\n{'=' * 60}")
        print(f"Training: {model_name}")
        print(f"{'=' * 60}")
        print(f"Epochs: {exp_cfg.num_epochs}, LR: {exp_cfg.lr}")
        print(f"Train cache: {train_cache['target'].shape[0]} pixels from multi-slice")
        print(f"HF patches from slice {train_slice_idx}, region {region}, {hf_N} pixels")
        print(f"Eval on slice {test_slice_idx}, region {region}")

        t_start = time.time()

        if hf_enabled:
            train_log, comp_log = train_direct_model_cached_hf(
                model=model,
                cached_loader=train_cached_loader,
                stats=stats,
                loss_cfg=loss_cfg,
                model_name=model_name,
                num_epochs=exp_cfg.num_epochs,
                patch_size=exp_cfg.patch_size,
                lr=exp_cfg.lr,
                weight_decay=exp_cfg.weight_decay,
                grad_clip=exp_cfg.grad_clip,
                device=device,
                hf_patch_dim=hf_patch_dim,
                hf_steps_per_epoch=hf_steps,
                hf_region_cache=hf_region_cache,
            )
        else:
            train_log = train_direct_model_cached(
                model=model,
                cached_loader=train_cached_loader,
                stats=stats,
                model_name=model_name,
                num_epochs=exp_cfg.num_epochs,
                patch_size=exp_cfg.patch_size,
                lr=exp_cfg.lr,
                weight_decay=exp_cfg.weight_decay,
                grad_clip=exp_cfg.grad_clip,
                device=device,
            )
            comp_log = None

        train_time = time.time() - t_start
        print(f"Training time: {train_time / 60:.1f} min")

        # Save model
        model_path = os.path.join(exp_cfg.save_dir, f"{safe_model_name(model_name)}.pt")
        torch.save(model.state_dict(), model_path)
        print(f"Saved model: {model_path}")

        # Save train log
        if comp_log is not None:
            log_df = pd.DataFrame(comp_log)
        else:
            log_df = pd.DataFrame({
                "epoch": range(1, len(train_log) + 1),
                "loss_img": train_log,
                "loss_grad": 0.0,
                "loss_lap": 0.0,
                "total_loss": train_log,
                "lambda_grad": 0.0,
                "lambda_lap": 0.0,
            })
        log_path = os.path.join(exp_cfg.save_dir,
                                 f"train_log_{safe_model_name(model_name)}.csv")
        log_df.to_csv(log_path, index=False)
        print(f"Saved train log: {log_path}")

    # ---- Evaluate on eval slice ----
    model.eval()
    print(f"\n{'=' * 60}")
    print(f"Evaluation on slice {test_slice_idx}, region {region}")
    print(f"{'=' * 60}")

    from src.evaluation.region_eval import predict_region_from_cache
    recon = predict_region_from_cache(
        model=model,
        region_cache=eval_region_cache,
        stats=stats,
        batch_size=exp_cfg.chunk_size_eval,
        patch_size=exp_cfg.patch_size,
        device=device,
    )
    target_arr = recon["target"]
    pred_arr = recon["pred"]
    center_arr = recon["center_base"]
    local_3x3_arr = recon["local_3x3_base"]

    metrics_model = compute_metrics_np(pred_arr, target_arr)
    metrics_center = compute_metrics_np(center_arr, target_arr)
    metrics_local = compute_metrics_np(local_3x3_arr, target_arr)

    print(f"\nModel         PSNR={metrics_model['PSNR']:.4f} dB  SSIM={metrics_model['SSIM']:.6f}")
    print(f"Center base    PSNR={metrics_center['PSNR']:.4f} dB  SSIM={metrics_center['SSIM']:.6f}")
    print(f"Local-3x3 base PSNR={metrics_local['PSNR']:.4f} dB  SSIM={metrics_local['SSIM']:.6f}")

    # ---- Save metrics ----
    metrics_rows = [
        {"method": "center_base", **metrics_center},
        {"method": "local_3x3_base", **metrics_local},
        {"method": model_name, **metrics_model},
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(exp_cfg.save_dir, "hf_ablation_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    print(f"Saved metrics: {metrics_path}")

    # ---- HF-specific evaluation metrics (optional) ----
    if hf_enabled:
        from src.training.losses import gradient_loss_2d, laplacian_loss_2d
        pred_t = torch.from_numpy(pred_arr)[None, None, :, :].float()
        targ_t = torch.from_numpy(target_arr)[None, None, :, :].float()

        grad_mae = float(gradient_loss_2d(pred_t, targ_t).item())
        lap_mae = float(laplacian_loss_2d(pred_t, targ_t).item())

        print(f"\nHF eval metrics:")
        print(f"  Grad-MAE: {grad_mae:.6f}")
        print(f"  Lap-MAE : {lap_mae:.6f}")

        hf_metrics_row = {
            "method": model_name,
            "Grad_MAE": grad_mae,
            "Lap_MAE": lap_mae,
            "PSNR": metrics_model["PSNR"],
            "SSIM": metrics_model["SSIM"],
        }
        pd.DataFrame([hf_metrics_row]).to_csv(
            os.path.join(exp_cfg.save_dir, "hf_ablation_hf_metrics.csv"), index=False,
        )

    # ---- Comparison figure ----
    images = [target_arr, center_arr, local_3x3_arr, pred_arr]
    titles = [
        "Target",
        f"Center base\nPSNR={metrics_center['PSNR']:.2f}",
        f"Local-3x3 base\nPSNR={metrics_local['PSNR']:.2f}",
        f"{model_name}\nPSNR={metrics_model['PSNR']:.2f}",
    ]
    fig_path = os.path.join(exp_cfg.save_dir, "hf_ablation_comparison.png")
    plot_comparison_images(images, titles, save_path=fig_path, show=False)
    print(f"Saved figure: {fig_path}")

    # ---- Error map & gradient maps ----
    abs_error = np.abs(pred_arr - target_arr)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    im0 = axes[0].imshow(abs_error, cmap="hot")
    axes[0].set_title(f"Abs Error | MAE={float(np.mean(abs_error)):.6f}")
    plt.colorbar(im0, ax=axes[0], fraction=0.046)

    pred_grad_x = np.abs(np.gradient(pred_arr, axis=1))
    targ_grad_x = np.abs(np.gradient(target_arr, axis=1))
    pred_grad = np.sqrt(np.gradient(pred_arr, axis=0)**2 + np.gradient(pred_arr, axis=1)**2)
    targ_grad = np.sqrt(np.gradient(target_arr, axis=0)**2 + np.gradient(target_arr, axis=1)**2)

    im1 = axes[1].imshow(pred_grad, cmap="viridis")
    axes[1].set_title("Pred Gradient Magnitude")
    plt.colorbar(im1, ax=axes[1], fraction=0.046)

    grad_diff = np.abs(pred_grad - targ_grad)
    im2 = axes[2].imshow(grad_diff, cmap="hot")
    axes[2].set_title(f"Grad Diff | MAE={float(np.mean(grad_diff)):.6f}")
    plt.colorbar(im2, ax=axes[2], fraction=0.046)

    plt.tight_layout()
    err_path = os.path.join(exp_cfg.save_dir, "hf_ablation_error_maps.png")
    fig.savefig(err_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved error maps: {err_path}")

    # ---- Summary ----
    print(f"\n{'=' * 60}")
    print("EXPERIMENT A COMPLETE")
    print(f"{'=' * 60}")
    print(f"Config: {args.config}")
    print(f"HF enabled: {hf_enabled}")
    print(f"lambda_grad: {lambda_grad}")
    print(f"lambda_lap:  {lambda_lap}")
    print(f"Output dir:  {exp_cfg.save_dir}")


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    main()

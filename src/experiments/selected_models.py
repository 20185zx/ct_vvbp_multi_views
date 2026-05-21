"""Run the cleaned selected-model VVBP comparison experiment."""

from __future__ import annotations

import math
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import CachedSortedVVBPDataset
from src.data.cache_builder import (
    build_or_load_region_cache,
    build_or_load_train_cache,
    estimate_stats_from_train_cache,
)
from src.evaluation import compute_metrics_np, plot_comparison_images, predict_region_from_cache
from src.models import build_model
from src.training import train_direct_model_cached


DEFAULT_MODEL_NAMES = ["local rank center integral mlp, 10 epochs"]
LOCAL_RANK_CLOSED_NAME = "local-rank center integral closed"


def safe_model_name(name: str) -> str:
    """Convert a display model name into a safe checkpoint filename stem."""
    return name.replace(", ", "_").replace(" ", "_")


def array_stats(name: str, arr):
    """Return basic distribution statistics for a reconstructed image array."""
    return {
        "array": name,
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def print_array_stats(name: str, arr) -> None:
    stats = array_stats(name, arr)
    print(f"{name}:")
    print(f"  min  = {stats['min']:.8f}")
    print(f"  max  = {stats['max']:.8f}")
    print(f"  mean = {stats['mean']:.8f}")
    print(f"  std  = {stats['std']:.8f}")
    print()


@torch.no_grad()
def local_rank_center_integral_closed_from_cache(region_cache):
    """Compute the parameter-free local-rank closed integral baseline."""
    values_sorted = region_cache["values_sorted"].float()  # [N, J, K]
    n_pixels, n_patch, n_views = values_sorted.shape

    center_idx = n_patch // 2
    n_local_values = n_patch * n_views

    center_values = values_sorted[:, center_idx, :].contiguous()  # [N, K]
    merged_sorted = torch.sort(values_sorted.reshape(n_pixels, n_local_values), dim=-1).values

    left = torch.searchsorted(merged_sorted.contiguous(), center_values, right=False)
    right = torch.searchsorted(merged_sorted.contiguous(), center_values, right=True) - 1
    left = left.clamp(min=0, max=n_local_values - 1)
    right = right.clamp(min=0, max=n_local_values - 1)

    local_rank = 0.5 * (left.float() + right.float())
    q = local_rank / float(n_local_values - 1)

    q_sorted, order = torch.sort(q, dim=-1)
    center_sorted = torch.gather(center_values, dim=-1, index=order)

    q0 = torch.zeros(n_pixels, 1, device=q_sorted.device, dtype=q_sorted.dtype)
    q1 = torch.ones(n_pixels, 1, device=q_sorted.device, dtype=q_sorted.dtype)
    q_ext = torch.cat([q0, q_sorted, q1], dim=-1)
    v_ext = torch.cat([center_sorted[:, :1], center_sorted, center_sorted[:, -1:]], dim=-1)

    dq = (q_ext[:, 1:] - q_ext[:, :-1]).clamp_min(0.0)
    area = 0.5 * dq * (v_ext[:, 1:] + v_ext[:, :-1])
    pred = math.pi * area.sum(dim=-1)

    h_reg = region_cache["metadata"]["Hreg"]
    w_reg = region_cache["metadata"]["Wreg"]
    return pred.cpu().numpy().reshape(h_reg, w_reg)


def _prepare_training(model_names, train_loader, extractor, cfg, device):
    train_cache = build_or_load_train_cache(train_loader, extractor, cfg, device=device)
    stats_cached = estimate_stats_from_train_cache(train_cache, device=device)

    stats_path = os.path.join(cfg.save_dir, "stats_cached.pt")
    stats_to_save = {
        key: value.detach().cpu() if torch.is_tensor(value) else value
        for key, value in stats_cached.items()
    }
    torch.save(stats_to_save, stats_path)
    print("Saved stats:", stats_path)

    train_cached_dataset = CachedSortedVVBPDataset(train_cache)
    train_cached_loader = DataLoader(
        train_cached_dataset,
        batch_size=cfg.cached_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    return train_cache, stats_cached, train_cached_loader


def _load_models(model_names, cfg, device):
    checkpoint_dir = getattr(cfg, "checkpoint_dir", cfg.save_dir)
    stats_path = os.path.join(checkpoint_dir, "stats_cached.pt")
    print("Loading stats:", stats_path)
    stats_loaded = torch.load(stats_path, map_location=device, weights_only=False)
    stats_cached = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in stats_loaded.items()
    }

    models = {}
    for model_name in model_names:
        print("\n" + "=" * 80)
        print("Loading model:", model_name)
        print("=" * 80)
        model = build_model(model_name).to(device)
        model_path = os.path.join(checkpoint_dir, f"{safe_model_name(model_name)}.pt")
        print("Loading checkpoint:", model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        models[model_name] = model
    return models, stats_cached


def _train_models(model_names, train_cached_loader, stats_cached, cfg, device):
    models, logs = {}, {}
    for model_name in model_names:
        print("\n" + "=" * 80)
        print("Training:", model_name)
        print("=" * 80)

        model = build_model(model_name).to(device)
        logs[model_name] = train_direct_model_cached(
            model=model,
            cached_loader=train_cached_loader,
            stats=stats_cached,
            model_name=model_name,
            num_epochs=cfg.num_epochs,
            patch_size=cfg.patch_size,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            grad_clip=cfg.grad_clip,
            device=device,
        )
        models[model_name] = model

        model_path = os.path.join(cfg.save_dir, f"{safe_model_name(model_name)}.pt")
        torch.save(model.state_dict(), model_path)
        print("Saved:", model_path)
    return models, logs


def run_selected_models_experiment(
    dataset,
    train_loader,
    extractor,
    test_indices,
    cfg,
    device,
    model_names=None,
    region_name="single_center",
):
    """Train/evaluate selected models against center and local-rank baselines."""
    os.makedirs(cfg.save_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)

    model_names = model_names or DEFAULT_MODEL_NAMES
    base_only = "base_only" in model_names
    if base_only:
        model_names = []

    models, logs = {}, {}
    train_cache, stats_cached = None, None

    if not base_only:
        if getattr(cfg, "eval_only", False):
            print("\n[MODE] Evaluation only. Skip training.")
            models, stats_cached = _load_models(model_names, cfg, device)
        else:
            train_cache, stats_cached, train_cached_loader = _prepare_training(
                model_names, train_loader, extractor, cfg, device
            )
            models, logs = _train_models(
                model_names, train_cached_loader, stats_cached, cfg, device
            )

    global_test_idx = int(test_indices[0])
    region_cache = build_or_load_region_cache(
        dataset=dataset,
        extractor=extractor,
        global_idx=global_test_idx,
        region_name=region_name,
        region=cfg.region,
        cfg=cfg,
        device=device,
    )

    h_reg = region_cache["metadata"]["Hreg"]
    w_reg = region_cache["metadata"]["Wreg"]
    target_arr = region_cache["target"].numpy().reshape(h_reg, w_reg)
    center_arr = region_cache["center_base"].numpy().reshape(h_reg, w_reg)
    local_rank_closed_arr = local_rank_center_integral_closed_from_cache(region_cache)

    metrics = {
        "center base": compute_metrics_np(center_arr, target_arr),
        LOCAL_RANK_CLOSED_NAME: compute_metrics_np(local_rank_closed_arr, target_arr),
    }
    recons = {}
    distribution_rows = {}

    for model_name, model in models.items():
        print("Evaluating:", model_name)
        recon = predict_region_from_cache(
            model=model,
            region_cache=region_cache,
            stats=stats_cached,
            batch_size=cfg.cached_batch_size,
            patch_size=cfg.patch_size,
            device=device,
        )
        recons[model_name] = recon
        metrics[model_name] = compute_metrics_np(recon["pred"], recon["target"])

        print("\n[Distribution]", model_name)
        print_array_stats("Target", recon["target"])
        print_array_stats("Center base", recon["center_base"])
        print_array_stats("Local-rank center integral closed", local_rank_closed_arr)
        print_array_stats("Pred", recon["pred"])

        distribution_rows[model_name] = [
            array_stats("target", recon["target"]),
            array_stats("center_base", recon["center_base"]),
            array_stats("local_rank_closed", local_rank_closed_arr),
            array_stats("pred", recon["pred"]),
        ]

    method_order = ["center base", LOCAL_RANK_CLOSED_NAME] + model_names
    full_metrics_df = pd.DataFrame(metrics).T.loc[method_order]
    summary_table = full_metrics_df[["PSNR", "SSIM"]]

    full_metrics_path = os.path.join(cfg.save_dir, "selected_model_full_metrics.csv")
    summary_path = os.path.join(cfg.save_dir, "selected_model_psnr_ssim.csv")
    full_metrics_df.to_csv(full_metrics_path)
    summary_table.to_csv(summary_path)
    print("Saved:", full_metrics_path)
    print("Saved:", summary_path)

    if distribution_rows:
        records = []
        for model_name, rows in distribution_rows.items():
            for row in rows:
                records.append({"model": model_name, **row})
        dist_path = os.path.join(cfg.save_dir, "prediction_distribution_stats.csv")
        pd.DataFrame(records).to_csv(dist_path, index=False)
        print("Saved:", dist_path)

    if base_only:
        best_model_name = None
        images = [target_arr, center_arr, local_rank_closed_arr]
        titles = [
            "Target",
            f"Center base\nPSNR={metrics['center base']['PSNR']:.2f} dB",
            f"Local-rank closed\nPSNR={metrics[LOCAL_RANK_CLOSED_NAME]['PSNR']:.2f} dB",
        ]
        fig_path = os.path.join(cfg.save_dir, "base_only_comparison.png")
    else:
        best_model_name = summary_table.loc[model_names, "PSNR"].idxmax()
        images = [target_arr, center_arr, local_rank_closed_arr, recons[best_model_name]["pred"]]
        titles = [
            "Target",
            f"Center base\nPSNR={metrics['center base']['PSNR']:.2f} dB",
            f"Local-rank closed\nPSNR={metrics[LOCAL_RANK_CLOSED_NAME]['PSNR']:.2f} dB",
            f"{best_model_name}\nPSNR={metrics[best_model_name]['PSNR']:.2f} dB",
        ]
        fig_path = os.path.join(cfg.save_dir, "selected_model_comparison.png")

    plot_comparison_images(images, titles, save_path=fig_path)

    return {
        "models": models,
        "logs": logs,
        "stats_cached": stats_cached,
        "train_cache": train_cache,
        "region_cache": region_cache,
        "metrics": metrics,
        "full_metrics_df": full_metrics_df,
        "summary_table": summary_table,
        "recons": recons,
        "best_model_name": best_model_name,
    }

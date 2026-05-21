import argparse
import os
import sys
import time

import torch
import torch.nn as nn
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.config import load_run_config, save_run_config
from src.experiments.project_setup import prepare_project_context
from src.geometry import load_or_generate_geo
from src.models.sino_upsample_residual_fbp import SinoUpsampleResidualFBPNet
from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_images


def safe_model_name(name: str):
    return name.replace(", ", "_").replace(" ", "_").replace("-", "_")


def crop_region(img, region):
    x0, x1, y0, y1 = region
    return img[x0:x1, y0:y1]


@torch.no_grad()
def estimate_target_stats(train_loader, device):
    values = []

    for _, img in train_loader:
        img = img.to(device)
        values.append(img.flatten())

    values = torch.cat(values)

    return {
        "target_mean": values.mean(),
        "target_std": values.std().clamp_min(1e-8),
    }


def train_baseline(
    model,
    train_loader,
    target_stats,
    num_epochs=10,
    lr=1e-3,
    weight_decay=1e-6,
    grad_clip=None,
    device="cuda",
):
    model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    criterion = nn.MSELoss()

    target_mean = target_stats["target_mean"].to(device)
    target_std = target_stats["target_std"].to(device)

    logs = []

    for epoch in range(1, num_epochs + 1):
        model.train()

        t0 = time.time()
        total_loss = 0.0
        total_count = 0

        for sino60, target in train_loader:
            sino60 = sino60.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            pred = model(sino60)

            pred_norm = (pred - target_mean) / target_std
            target_norm = (target - target_mean) / target_std

            loss = criterion(pred_norm, target_norm)

            optimizer.zero_grad()
            loss.backward()

            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

            optimizer.step()

            bs = target.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / max(total_count, 1)
        logs.append(avg_loss)

        print(
            f"Epoch [{epoch:03d}/{num_epochs}] "
            f"Loss={avg_loss:.8f} "
            f"Time={time.time() - t0:.1f}s"
        )

    return logs


@torch.no_grad()
def evaluate_on_region(
    model,
    dataset,
    global_idx,
    region,
    device,
):
    model.eval()

    sino60, target = dataset[global_idx]

    sino60 = sino60.unsqueeze(0).to(device)
    target = target.unsqueeze(0).to(device)

    pred = model(sino60)

    pred_np = pred[0, 0].detach().cpu().numpy()
    target_np = target[0, 0].detach().cpu().numpy()

    pred_region = crop_region(pred_np, region)
    target_region = crop_region(target_np, region)

    metrics = compute_metrics_np(pred_region, target_region)

    return {
        "pred_full": pred_np,
        "target_full": target_np,
        "pred": pred_region,
        "target": target_region,
        "metrics": metrics,
    }


def main():
    parser = argparse.ArgumentParser(description="Run 60-to-240 sinogram residual FBP baseline.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/sino_60_to_240_baseline.json",
        help="Path to the baseline JSON/YAML config.",
    )
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment

    if run_cfg.save_dir is not None:
        exp_cfg.save_dir = run_cfg.save_dir
    if run_cfg.cache_dir is not None:
        exp_cfg.cache_dir = run_cfg.cache_dir

    exp_cfg.region = tuple(run_cfg.region)
    exp_cfg.ensure_dirs()

    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))

    ctx = prepare_project_context(
        dicom_folder=run_cfg.dicom_folder,
        results_folder=run_cfg.results_folder,
        save_dir=exp_cfg.save_dir,
        cache_dir=exp_cfg.cache_dir,
        cfg=exp_cfg,
        seed=exp_cfg.seed,
    )

    device = ctx["device"]
    dataset_sparse = ctx["dataset_sparse"]
    train_loader_sparse = ctx["train_loader_sparse"]
    test_indices = ctx["test_indices"]
    results_folder = str(ctx["paths"]["results_folder"])

    # Important: build 240-view geometry for the fixed FBP baseline
    print("\nBuilding/loading 240-view geometry for baseline FBP...")
    geo_full = load_or_generate_geo(
        views=exp_cfg.full_views,
        results_folder=results_folder,
        device=device,
        image_size=exp_cfg.image_size,
        n_detec=exp_cfg.n_detec,
        d_detec=exp_cfg.d_detec,
        d_voxel=exp_cfg.d_voxel,
        DSO=exp_cfg.DSO,
        DOD=exp_cfg.DOD,
    )

    model_name = "sino 60 to 240 residual fbp baseline"

    # ------------------------------------------------------------
    # Load or estimate target normalization stats
    # ------------------------------------------------------------
    target_stats_path = os.path.join(
        exp_cfg.cache_dir,
        "baseline_target_stats_train60.pt"
    )

    if os.path.exists(target_stats_path):
        print("\nLoading cached target stats...")
        target_stats_cpu = torch.load(target_stats_path, map_location="cpu", weights_only=False)
        target_stats = {
            k: v.to(device) if torch.is_tensor(v) else v
            for k, v in target_stats_cpu.items()
        }
        print("Loaded target stats:", target_stats_path)
    else:
        print("\nEstimating target stats...")
        target_stats = estimate_target_stats(train_loader_sparse, device=device)

        os.makedirs(exp_cfg.cache_dir, exist_ok=True)
        torch.save(
            {k: v.detach().cpu() for k, v in target_stats.items()},
            target_stats_path,
        )
        print("Saved target stats:", target_stats_path)

    print("target_mean:", float(target_stats["target_mean"]))
    print("target_std :", float(target_stats["target_std"]))

    model = SinoUpsampleResidualFBPNet(
        geo_full=geo_full,
        input_views=exp_cfg.sparse_views,
        full_views=240,
        sino_channels=32,
        num_blocks=5,
        residual_scale=0.1,
        use_sino_norm=True,
    )

    print("\nTraining baseline:", model_name)
    logs = train_baseline(
        model=model,
        train_loader=train_loader_sparse,
        target_stats=target_stats,
        num_epochs=exp_cfg.num_epochs,
        lr=exp_cfg.lr,
        weight_decay=exp_cfg.weight_decay,
        grad_clip=exp_cfg.grad_clip,
        device=device,
    )

    model_path = os.path.join(exp_cfg.save_dir, safe_model_name(model_name) + ".pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model:", model_path)

    log_path = os.path.join(exp_cfg.save_dir, "baseline_train_log.csv")
    pd.DataFrame(
        {
            "epoch": list(range(1, len(logs) + 1)),
            "loss": logs,
        }
    ).to_csv(log_path, index=False)
    print("Saved train log:", log_path)

    global_test_idx = int(test_indices[0])
    print("\nEvaluating global test idx:", global_test_idx)
    print("Region:", exp_cfg.region)

    result = evaluate_on_region(
        model=model,
        dataset=dataset_sparse,
        global_idx=global_test_idx,
        region=exp_cfg.region,
        device=device,
    )

    metrics_df = pd.DataFrame({model_name: result["metrics"]}).T

    metrics_path = os.path.join(exp_cfg.save_dir, "baseline_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print("Saved metrics:", metrics_path)
    print(metrics_df[["PSNR", "SSIM"]])

    fig_path = os.path.join(exp_cfg.save_dir, "sino_60_to_240_baseline_comparison.png")

    plot_comparison_images(
        [result["target"], result["pred"]],
        [
            "Target",
            f"{model_name}\nPSNR={result['metrics']['PSNR']:.2f} dB",
        ],
        save_path=fig_path,
    )

    print("Saved figure:", fig_path)


if __name__ == "__main__":
    main()
"""
Multi-rate VVBP experiment: train multiple models on mixed {9,18,36,72}-view data,
then evaluate separately at each sparse view count.

Usage:
    python scripts/run_multirate_vvbp.py --config configs/multirate_selected_models.json
    python scripts/run_multirate_vvbp.py  # uses defaults
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

from src.utils.config import load_run_config, save_run_config
from src.experiments.project_setup import prepare_multirate_context
from src.evaluation.region_eval import evaluate_multirate
from src.evaluation.fbp_baseline import compute_fbp_baselines
from src.evaluation.visualization import plot_comparison_grid
from src.models import build_model
from src.training import estimate_multirate_stats
from src.data.local_vvbp import sample_random_coords, gather_sorted_vvbp_patch
from src.data.feature_builder import make_model_features_from_values


def safe_model_name(name: str) -> str:
    return name.replace(", ", "_").replace(" ", "_")


def short_label(name: str) -> str:
    """Strip _10_epochs suffix for display."""
    return safe_model_name(name).replace("_10_epochs", "")


def train_one_multirate_epoch(model, train_loader, extractors, v_stats,
                             target_mean, target_std, optimizer, criterion,
                             patch_size, pixels_per_batch, device,
                             train_region=None, grad_clip=None):
    """Run one training epoch over the multi-rate loader. Returns avg loss and elapsed time."""
    model.train()
    t0 = time.time()
    total_loss = 0.0
    total_count = 0

    for sino_batch, img_batch in train_loader:
        sino_batch = sino_batch.to(device, non_blocking=True)
        img_batch = img_batch.to(device, non_blocking=True)

        V = sino_batch.shape[-2]
        extractor = extractors[V]
        vs = v_stats[V]
        batch_stats = {
            "target_mean": target_mean,
            "target_std": target_std,
            "v_mean": vs["v_mean"].to(device),
            "v_std": vs["v_std"].to(device),
        }

        vvbp = extractor(sino_batch)
        B, _, H, W, _ = vvbp.shape

        xs, ys = sample_random_coords(
            H, W, num_pixels=pixels_per_batch,
            margin=patch_size // 2, device=device,
            train_region=train_region,
        )
        values_sorted = gather_sorted_vvbp_patch(
            vvbp, xs, ys, patch_size=patch_size, mode="3x3",
        )
        N = B * pixels_per_batch
        values_sorted = values_sorted.reshape(N, values_sorted.shape[2], values_sorted.shape[3])
        target = img_batch[:, 0, xs, ys].reshape(N, 1)

        y_norm = (target - batch_stats["target_mean"]) / batch_stats["target_std"]

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

        loss = criterion(pred_norm, y_norm)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        total_loss += loss.item() * y_norm.shape[0]
        total_count += y_norm.shape[0]

    avg_loss = total_loss / max(total_count, 1)
    return avg_loss, time.time() - t0


def main():
    parser = argparse.ArgumentParser(
        description="Multi-rate VVBP training + per-V evaluation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multirate_selected_models.json",
        help="Path to JSON config.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override num_epochs from config.",
    )
    parser.add_argument(
        "--eval_only",
        action="store_true",
        help="Skip training, load checkpoint and evaluate.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint for eval_only.",
    )
    args = parser.parse_args()

    run_cfg = load_run_config(args.config)
    exp_cfg = run_cfg.experiment
    if args.epochs is not None:
        exp_cfg.num_epochs = args.epochs

    exp_cfg.ensure_dirs()
    save_run_config(run_cfg, os.path.join(exp_cfg.save_dir, "config_used.json"))

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # --- Multi-rate context (shared across models) ---
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
    sparse_views = exp_cfg.sparse_views
    region = tuple(run_cfg.region)

    print("\nSparse views:", sparse_views)
    print("Region:", region)

    # --- Estimate normalization stats (once, shared) ---
    print("\n" + "=" * 60)
    print("Estimating normalization statistics ...")
    print("=" * 60)
    target_stats_raw, v_stats_raw = estimate_multirate_stats(
        train_loader, extractors, exp_cfg, device,
        num_stats_batches=4,
    )
    target_stats = {k: v.to(device) for k, v in target_stats_raw.items()}
    v_stats = {
        V: {k: v.to(device) for k, v in s.items()}
        for V, s in v_stats_raw.items()
    }

    # --- Models to train/evaluate ---
    model_names = run_cfg.model_names
    if model_names is None or len(model_names) == 0:
        model_names = ["local rank center integral mlp, 10 epochs"]

    global_test_idx = int(test_indices[0])

    # Per-model results storage
    all_model_preds = {}    # model_name -> {V: ndarray}
    all_model_metrics = {}  # model_name -> DataFrame

    # Baselines (model-independent, captured from first evaluation)
    baseline_preds = None
    baseline_metrics_raw = None
    target_arr = None

    for model_name in model_names:
        print(f"\n{'=' * 60}")
        print(f"Model: {model_name}")
        print(f"{'=' * 60}")

        model = build_model(model_name)

        if args.eval_only:
            model = model.to(device)
            ckpt_path = args.checkpoint or os.path.join(
                exp_cfg.save_dir, f"{safe_model_name(model_name)}.pt"
            )
            print(f"Loading checkpoint: {ckpt_path}")
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        else:
            num_epochs = exp_cfg.num_epochs
            print(f"\nTraining on slices [0, {len(ctx['train_indices'])}), "
                  f"test slice {global_test_idx}")
            print(f"Epochs: {num_epochs}, LR: {exp_cfg.lr}")

            model = model.to(device)
            optimizer = torch.optim.Adam(
                model.parameters(), lr=exp_cfg.lr, weight_decay=exp_cfg.weight_decay,
            )
            criterion = torch.nn.MSELoss()
            target_mean = target_stats["target_mean"]
            target_std = target_stats["target_std"]

            # Precompute FBP baselines once (model-independent)
            print("\nPrecomputing FBP baselines ...")
            precomputed_fbp = compute_fbp_baselines(
                eval_dataset=eval_dataset,
                geo_dict=geo_dict,
                sparse_views=sparse_views,
                test_idx=global_test_idx,
                region=region,
                device=device,
            )

            train_log = []
            epoch_eval_rows = []
            best_avg_psnr = -1e9
            safe_name = safe_model_name(model_name)

            t_train_start = time.time()
            for epoch in range(1, num_epochs + 1):
                avg_loss, epoch_time = train_one_multirate_epoch(
                    model=model,
                    train_loader=train_loader,
                    extractors=extractors,
                    v_stats=v_stats,
                    target_mean=target_mean,
                    target_std=target_std,
                    optimizer=optimizer,
                    criterion=criterion,
                    patch_size=exp_cfg.patch_size,
                    pixels_per_batch=exp_cfg.cache_pixels_per_slice,
                    device=device,
                    train_region=exp_cfg.train_region,
                    grad_clip=exp_cfg.grad_clip,
                )
                train_log.append(avg_loss)
                print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] "
                      f"loss={avg_loss:.6f} time={epoch_time:.1f}s")

                # --- Evaluate after each epoch ---
                if getattr(exp_cfg, 'eval_every_epoch', True):
                    model.eval()
                    model_metrics_df, base_metrics, model_preds, base_preds, target_arr = \
                        evaluate_multirate(
                            model=model,
                            eval_dataset=eval_dataset,
                            extractors=extractors,
                            geo_dict=geo_dict,
                            target_stats=target_stats,
                            v_stats=v_stats,
                            sparse_views=sparse_views,
                            test_idx=global_test_idx,
                            region=region,
                            patch_size=exp_cfg.patch_size,
                            chunk_size=exp_cfg.chunk_size_eval,
                            device=device,
                        )

                    # Log per-epoch metrics
                    for V in sparse_views:
                        fbp_m = precomputed_fbp["fbp_metrics"][V]
                        lr_m = base_metrics["Local-rank closed"][V]
                        epoch_eval_rows.append({
                            "epoch": epoch,
                            "V": V,
                            "PSNR": model_metrics_df.loc[V, "PSNR"],
                            "SSIM": model_metrics_df.loc[V, "SSIM"],
                            "FBP_PSNR": fbp_m["PSNR"],
                            "FBP_SSIM": fbp_m["SSIM"],
                            "LR_closed_PSNR": lr_m["PSNR"],
                            "LR_closed_SSIM": lr_m["SSIM"],
                        })

                    # Save epoch eval log
                    epoch_eval_df = pd.DataFrame(epoch_eval_rows)
                    epoch_eval_path = os.path.join(
                        exp_cfg.save_dir, f"multirate_epoch_eval_log_{safe_name}.csv",
                    )
                    epoch_eval_df.to_csv(epoch_eval_path, index=False)

                    # Track best by average PSNR across sparse views
                    avg_psnr = model_metrics_df["PSNR"].mean()
                    if avg_psnr > best_avg_psnr:
                        best_avg_psnr = avg_psnr
                        best_path = os.path.join(
                            exp_cfg.save_dir, f"{safe_name}_best_avg_psnr.pt",
                        )
                        torch.save(model.state_dict(), best_path)
                        print(f"  -> New best avg PSNR={best_avg_psnr:.4f}, "
                              f"saved: {best_path}")

                        # Save best-epoch visualization
                        best_fig_path = os.path.join(
                            exp_cfg.save_dir,
                            f"{safe_name}_best_epoch_{epoch:03d}_comparison.png",
                        )
                        preds_by_method = {
                            "FBP": {
                                V: precomputed_fbp["fbp_preds_region"][V]
                                for V in sparse_views
                            },
                            "Local-rank closed": base_preds["Local-rank closed"],
                            short_label(model_name): model_preds,
                        }
                        psnr_by_method = {
                            "FBP": {V: precomputed_fbp["fbp_metrics"][V]["PSNR"] for V in sparse_views},
                            "Local-rank closed": {
                                V: base_metrics["Local-rank closed"][V]["PSNR"] for V in sparse_views
                            },
                            short_label(model_name): {
                                V: model_metrics_df.loc[V, "PSNR"] for V in sparse_views
                            },
                        }
                        plot_comparison_grid(
                            target=target_arr,
                            preds_by_method=preds_by_method,
                            psnr_by_method=psnr_by_method,
                            col_labels=["FBP", "Local-rank closed", short_label(model_name)],
                            sparse_views=sparse_views,
                            save_path=best_fig_path,
                            show=False,
                        )
                        print(f"  -> Saved best-epoch visualization: {best_fig_path}")

            train_time = time.time() - t_train_start
            print(f"Training time: {train_time / 60:.1f} min")

            model_path = os.path.join(exp_cfg.save_dir, f"{safe_name}.pt")
            torch.save(model.state_dict(), model_path)
            print(f"Saved final model: {model_path}")

            log_path = os.path.join(exp_cfg.save_dir, f"train_log_{safe_name}.csv")
            pd.DataFrame({"epoch": range(1, len(train_log) + 1), "loss": train_log}).to_csv(
                log_path, index=False
            )
            print(f"Saved train log: {log_path}")

        # --- Evaluate per V ---
        model = model.to(device)
        model.eval()

        print(f"\n{'=' * 60}")
        print(f"Per-V evaluation on test slice {global_test_idx}, region {region}")
        print(f"{'=' * 60}")

        model_metrics_df, base_metrics, model_preds, base_preds, target_arr = \
            evaluate_multirate(
                model=model,
                eval_dataset=eval_dataset,
                extractors=extractors,
                geo_dict=geo_dict,
                target_stats=target_stats,
                v_stats=v_stats,
                sparse_views=sparse_views,
                test_idx=global_test_idx,
                region=region,
                patch_size=exp_cfg.patch_size,
                chunk_size=exp_cfg.chunk_size_eval,
                device=device,
            )

        all_model_preds[model_name] = model_preds
        all_model_metrics[model_name] = model_metrics_df

        # Capture baselines from first model (they are model-independent).
        if baseline_preds is None:
            baseline_preds = base_preds
            baseline_metrics_raw = base_metrics

        # Save per-model metrics
        safe_name = safe_model_name(model_name)
        metrics_path = os.path.join(exp_cfg.save_dir, f"multirate_per_v_metrics_{safe_name}.csv")
        model_metrics_df.to_csv(metrics_path)
        print(f"Saved per-V metrics: {metrics_path}")
        print(model_metrics_df[["PSNR", "SSIM"]])

        # Save per-model results
        results_path = os.path.join(exp_cfg.save_dir, f"multirate_results_{safe_name}.pt")
        torch.save(
            {
                "target": target_arr,
                "preds": model_preds,
                "baseline_preds": base_preds,
                "metrics": model_metrics_df.to_dict(),
                "baseline_metrics": base_metrics,
                "sparse_views": sparse_views,
                "region": region,
                "model_name": model_name,
            },
            results_path,
        )
        print(f"Saved results: {results_path}")

    # --- FBP baselines ---
    x0, x1, y0, y1 = region
    precomputed_fbp = compute_fbp_baselines(
        eval_dataset=eval_dataset,
        geo_dict=geo_dict,
        sparse_views=sparse_views,
        test_idx=global_test_idx,
        region=region,
        device=device,
    )
    fbp_metrics = precomputed_fbp["fbp_metrics"]
    fbp_preds = precomputed_fbp["full_fbp_preds"]

    # --- Build all_methods_metrics.csv (long format) ---
    rows = []
    method_order = ["FBP", "Local-rank closed"]

    for V in sparse_views:
        for method in method_order:
            if method == "FBP":
                m = fbp_metrics[V]
            else:
                m = baseline_metrics_raw[method][V]
            rows.append({
                "V": V,
                "method": method,
                "MSE": m["MSE"],
                "MAE": m["MAE"],
                "PSNR": m["PSNR"],
                "SSIM": m["SSIM"],
            })

    for model_name in model_names:
        method_label = short_label(model_name)
        method_order.append(method_label)
        mdf = all_model_metrics[model_name]
        for V in sparse_views:
            rows.append({
                "V": V,
                "method": method_label,
                "MSE": mdf.loc[V, "MSE"],
                "MAE": mdf.loc[V, "MAE"],
                "PSNR": mdf.loc[V, "PSNR"],
                "SSIM": mdf.loc[V, "SSIM"],
            })

    all_methods_df = pd.DataFrame(rows)
    all_methods_path = os.path.join(exp_cfg.save_dir, "multirate_all_methods_metrics.csv")
    all_methods_df.to_csv(all_methods_path, index=False)
    print(f"\nSaved all-methods metrics: {all_methods_path}")

    # --- Grid comparison figure ---
    # Rows = V, Cols = [Target, FBP, Center base, Local-rank closed, model_1, ...]
    preds_by_method = {}
    psnr_by_method = {}

    # FBP
    fbp_region_preds = {V: fbp_preds[V][x0:x1, y0:y1] for V in sparse_views}
    preds_by_method["FBP"] = fbp_region_preds
    psnr_by_method["FBP"] = {V: fbp_metrics[V]["PSNR"] for V in sparse_views}

    col_labels = ["FBP"]

    # Parameter-free baseline
    preds_by_method["Local-rank closed"] = baseline_preds["Local-rank closed"]
    psnr_by_method["Local-rank closed"] = {
        V: baseline_metrics_raw["Local-rank closed"][V]["PSNR"] for V in sparse_views
    }
    col_labels.append("Local-rank closed")

    # Learned models
    for model_name in model_names:
        short = short_label(model_name)
        preds_by_method[short] = all_model_preds[model_name]
        psnr_by_method[short] = {V: all_model_metrics[model_name].loc[V, "PSNR"]
                                 for V in sparse_views}
        col_labels.append(short)

    fig_path = os.path.join(exp_cfg.save_dir, "multirate_comparison.png")
    plot_comparison_grid(
        target=target_arr,
        preds_by_method=preds_by_method,
        psnr_by_method=psnr_by_method,
        col_labels=col_labels,
        sparse_views=sparse_views,
        save_path=fig_path,
        show=False,
    )
    print(f"Saved comparison figure: {fig_path}")

    # --- Save all results ---
    all_results = {
        "target": target_arr,
        "fbp_preds": fbp_preds,
        "fbp_metrics": fbp_metrics,
        "baseline_preds": baseline_preds,
        "baseline_metrics": baseline_metrics_raw,
        "model_preds": all_model_preds,
        "model_metrics": {n: df.to_dict() for n, df in all_model_metrics.items()},
        "sparse_views": sparse_views,
        "region": region,
        "model_names": model_names,
        "method_order": method_order,
    }
    results_path = os.path.join(exp_cfg.save_dir, "multirate_all_results.pt")
    torch.save(all_results, results_path)
    print(f"Saved all results: {results_path}")

    # --- Final summary ---
    print("\n" + "=" * 100)
    print("FINAL SUMMARY")
    print("=" * 100)

    # Header: FBP | Local-rank closed | model_1 | model_2 | ...
    header = f"{'V':>5s}  {'FBP PSNR':>10s}  {'FBP SSIM':>10s}"
    header += f"  {'LR-closed PSNR':>15s}  {'LR-closed SSIM':>15s}"
    for model_name in model_names:
        label = short_label(model_name)
        label = label[:16] if len(label) > 16 else label
        header += f"  {label + ' PSNR':>14s}  {label + ' SSIM':>14s}"
    print(header)
    print("-" * len(header.expandtabs()))

    for V in sparse_views:
        line = f"{V:5d}  {fbp_metrics[V]['PSNR']:10.4f}  {fbp_metrics[V]['SSIM']:10.6f}"
        line += f"  {baseline_metrics_raw['Local-rank closed'][V]['PSNR']:15.4f}"
        line += f"  {baseline_metrics_raw['Local-rank closed'][V]['SSIM']:15.6f}"
        for model_name in model_names:
            mdf = all_model_metrics[model_name]
            line += f"  {mdf.loc[V, 'PSNR']:14.4f}  {mdf.loc[V, 'SSIM']:14.6f}"
        print(line)


if __name__ == "__main__":
    main()

import os
import sys
import math
import argparse

import torch
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from src.evaluation.metrics import compute_metrics_np
from src.evaluation.visualization import plot_comparison_images

def local_rank_center_integral(values_sorted, use_closed_interval=False, normalize_weight_sum=False):
    """
    Local-rank center VVBP integral.

    values_sorted:
        [N, J, K]
        J = 9 for 3x3 patch
        K = number of sparse views

    Idea:
        For each center pixel:
        1. Merge all J×K VVBP values -> L = J*K values.
        2. Sort the L local values.
        3. Take the center pixel's original K values.
        4. Find their ranks inside the merged 540-value distribution.
        5. Use the local rank coordinates for non-uniform trapezoid integration.

    Return:
        pred: [N]
    """
    if values_sorted.ndim != 3:
        raise ValueError(f"values_sorted should be [N, J, K], got {values_sorted.shape}")

    N, J, K = values_sorted.shape

    if J != 9:
        raise ValueError(f"This experiment expects patch_size=3, so J=9. Got J={J}")

    center_idx = J // 2
    L = J * K

    # Center pixel's K VVBP values.
    # Since the cache stores sorted values for each spatial position,
    # this is already sorted ascending.
    center_values = values_sorted[:, center_idx, :]  # [N, K]

    # Merge all 3x3 local VVBP values and sort globally.
    merged_values = values_sorted.reshape(N, L)      # [N, 540]
    merged_sorted = torch.sort(merged_values, dim=-1).values

    # Find local rank position of each center value inside merged_sorted.
    # To handle duplicate values, use the average of left and right insertion ranks.
    left = torch.searchsorted(merged_sorted, center_values, right=False)
    right = torch.searchsorted(merged_sorted, center_values, right=True) - 1

    local_rank = 0.5 * (left.float() + right.float())  # [N, K]

    # Normalize rank coordinate into [0, 1].
    q = local_rank / float(L - 1)  # [N, K]

    # The center_values are sorted ascending, and q should also be non-decreasing.
    # However, for numerical safety, sort by q again.
    q_sorted, order = torch.sort(q, dim=-1)
    center_sorted = torch.gather(center_values, dim=-1, index=order)

    if use_closed_interval:
        # Optional version:
        # add boundary points (0, first value) and (1, last value),
        # so that the integration covers the full [0, 1] interval.
        q0 = torch.zeros(N, 1, device=values_sorted.device, dtype=values_sorted.dtype)
        q1 = torch.ones(N, 1, device=values_sorted.device, dtype=values_sorted.dtype)

        v0 = center_sorted[:, :1]
        v1 = center_sorted[:, -1:]

        q_ext = torch.cat([q0, q_sorted, q1], dim=-1)
        v_ext = torch.cat([v0, center_sorted, v1], dim=-1)
    else:
        # Strict version:
        # integrate only on the local-rank support occupied by center values.
        q_ext = q_sorted
        v_ext = center_sorted

    dq = q_ext[:, 1:] - q_ext[:, :-1]  # [N, M-1]

    # Convert non-uniform trapezoid integration into point weights.
    # For points v_0, ..., v_{M-1}:
    # weight_0     = 0.5 * (q_1 - q_0)
    # weight_i     = 0.5 * (q_i - q_{i-1}) + 0.5 * (q_{i+1} - q_i)
    # weight_{M-1} = 0.5 * (q_{M-1} - q_{M-2})
    point_w = torch.zeros_like(v_ext)

    point_w[:, :-1] += 0.5 * dq
    point_w[:, 1:] += 0.5 * dq

    if normalize_weight_sum:
        point_w = point_w / (point_w.sum(dim=-1, keepdim=True) + 1e-12)

    pred = math.pi * (point_w * v_ext).sum(dim=-1)

    return pred


def reshape_region(x, metadata):
    """
    x: [N]
    reshape to [Hreg, Wreg] if metadata is available.
    """
    Hreg = metadata.get("Hreg", None)
    Wreg = metadata.get("Wreg", None)

    if Hreg is not None and Wreg is not None and x.numel() == Hreg * Wreg:
        return x.reshape(Hreg, Wreg)

    return x


def compute_and_collect_metrics(name, pred, target):
    pred_np = pred.detach().cpu().numpy()
    target_np = target.detach().cpu().numpy()
    m = compute_metrics_np(pred_np, target_np)
    return name, m




def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--eval_cache",
        type=str,
        default="cache/vvbp_patches/eval_cache_g448_center_256_x128-384_y128-384_patch3.pt",
        help="Path to eval cache containing values_sorted and target.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/local_rank_center_integral",
        help="Output directory.",
    )

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Loading eval cache:")
    print(args.eval_cache)

    cache = torch.load(args.eval_cache, map_location="cpu", weights_only=False)

    print("Cache keys:", list(cache.keys()))

    values_sorted = cache["values_sorted"].float()  # [N, J, K]
    target = cache["target"].float()                # [N]

    metadata = cache.get("metadata", {})

    print("values_sorted:", tuple(values_sorted.shape))
    print("target:", tuple(target.shape))
    print("metadata:", metadata)

    # Existing no-learning bases, if available.
    results = {}

    if "center_base" in cache:
        results["center base"] = cache["center_base"].float()
    else:
        # center base = pi * mean of center pixel K VVBP values
        center_idx = values_sorted.shape[1] // 2
        results["center base"] = math.pi * values_sorted[:, center_idx, :].mean(dim=-1)

    if "local_3x3_base" in cache:
        results["local 3x3 mean base"] = cache["local_3x3_base"].float()
    else:
        # local 3x3 mean base = pi * mean of all J×K values
        results["local 3x3 mean base"] = math.pi * values_sorted.mean(dim=(1, 2))

    results["local-rank center integral"] = local_rank_center_integral(
    values_sorted,
    use_closed_interval=False,
    normalize_weight_sum=False,
    )

    results["local-rank center integral norm"] = local_rank_center_integral(
        values_sorted,
        use_closed_interval=False,
        normalize_weight_sum=True,
    )

    results["local-rank center integral closed"] = local_rank_center_integral(
        values_sorted,
        use_closed_interval=True,
        normalize_weight_sum=False,
    )

    # Metrics
    metric_rows = {}

    target_img = reshape_region(target, metadata)

    for name, pred in results.items():
        pred_img = reshape_region(pred, metadata)
        _, metrics = compute_and_collect_metrics(name, pred_img, target_img)
        metric_rows[name] = metrics

    metrics_df = pd.DataFrame(metric_rows).T

    metrics_path = os.path.join(args.output_dir, "local_rank_center_integral_metrics.csv")
    metrics_df.to_csv(metrics_path)

    print("\n===== Metrics =====")
    print(metrics_df[["MSE", "MAE", "PSNR", "SSIM"]])
    print(f"\nSaved metrics: {metrics_path}")

    # Save predictions
    pred_path = os.path.join(args.output_dir, "local_rank_center_integral_predictions.pt")
    torch.save(
        {
            "results": results,
            "target": target,
            "metadata": metadata,
        },
        pred_path,
    )
    print(f"Saved predictions: {pred_path}")

    # Visualization: use the same plotting function as previous experiments
    images = [
    target_img.detach().cpu().numpy(),
    reshape_region(results["center base"], metadata).detach().cpu().numpy(),
    reshape_region(results["local 3x3 mean base"], metadata).detach().cpu().numpy(),
    reshape_region(results["local-rank center integral"], metadata).detach().cpu().numpy(),
    reshape_region(results["local-rank center integral norm"], metadata).detach().cpu().numpy(),
    reshape_region(results["local-rank center integral closed"], metadata).detach().cpu().numpy(),
    ]

    titles = [
        "Target",
        f"Center base\nPSNR={metric_rows['center base']['PSNR']:.2f} dB",
        f"3x3 mean base\nPSNR={metric_rows['local 3x3 mean base']['PSNR']:.2f} dB",
        f"Local-rank integral\nPSNR={metric_rows['local-rank center integral']['PSNR']:.2f} dB",
        f"Local-rank norm\nPSNR={metric_rows['local-rank center integral norm']['PSNR']:.2f} dB",
        f"Local-rank closed\nPSNR={metric_rows['local-rank center integral closed']['PSNR']:.2f} dB",
    ]

    fig_path = os.path.join(args.output_dir, "local_rank_center_integral_comparison.png")

    plot_comparison_images(
        images,
        titles,
        save_path=fig_path,
    )

    print(f"Saved figure: {fig_path}")


if __name__ == "__main__":
    main()
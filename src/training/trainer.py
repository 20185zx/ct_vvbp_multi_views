import time
import math

import torch
import torch.nn as nn

from src.data.feature_builder import make_model_features_from_values
from src.data.local_vvbp import sample_random_coords, gather_sorted_vvbp_patch


def train_direct_model_cached(
    model,
    cached_loader,
    stats,
    model_name="model",
    num_epochs=20,
    patch_size=3,
    lr=1e-3,
    weight_decay=1e-6,
    grad_clip=None,
    device="cuda",
):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_log = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0
        for values_sorted, target, _, _ in cached_loader:
            values_sorted = values_sorted.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            y_norm = (target - stats["target_mean"]) / stats["target_std"]

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
            loss = criterion(pred_norm, y_norm)
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            bs = y_norm.shape[0]
            total_loss += loss.item() * bs
            total_count += bs
        avg_loss = total_loss / max(total_count, 1)
        train_log.append(avg_loss)
        print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] loss={avg_loss:.6f} time={time.time() - t0:.1f}s")
    return train_log


@torch.no_grad()
def estimate_multirate_stats(
    train_loader,
    extractors,
    cfg,
    device,
    num_stats_batches=4,
):
    """Estimate target mean/std and per-V v_mean/v_std from training data.

    Iterates up to num_stats_batches; continues until every V in extractors
    has at least one sample or the loader is exhausted.

    Returns:
        target_stats: dict with target_mean, target_std (V-independent).
        v_stats: dict mapping V -> {v_mean, v_std}.
    """
    target_values = []
    v_values = {V: [] for V in extractors.keys()}
    max_batches = max(num_stats_batches, len(extractors) * 2)

    for bi, (sino_batch, img_batch) in enumerate(train_loader):
        sino_batch = sino_batch.to(device)
        img_batch = img_batch.to(device)
        V = sino_batch.shape[-2]

        target_values.append(img_batch.flatten().cpu())

        extractor = extractors[V]
        vvbp = extractor(sino_batch)
        v_values[V].append(vvbp.flatten().cpu())

        all_covered = all(len(v_values[v]) > 0 for v in extractors.keys())
        if bi >= max_batches - 1 and all_covered:
            break

    all_targets = torch.cat(target_values)
    target_mean = all_targets.mean()
    target_std = all_targets.std().clamp_min(1e-8)

    print(f"[STATS] target_mean={float(target_mean):.6f}  target_std={float(target_std):.6f}")

    v_stats = {}
    for V, values in v_values.items():
        if len(values) == 0:
            raise RuntimeError(f"No stats collected for V={V}. Increase num_stats_batches.")
        all_v = torch.cat(values)
        v_stats[V] = {
            "v_mean": all_v.mean(),
            "v_std": all_v.std().clamp_min(1e-8),
        }
        print(f"[STATS] V={V}: v_mean={float(v_stats[V]['v_mean']):.6f}  v_std={float(v_stats[V]['v_std']):.6f}")

    return {"target_mean": target_mean, "target_std": target_std}, v_stats


def train_multirate_model(
    model,
    train_loader,
    extractors,
    target_stats,
    v_stats,
    model_name="multirate_model",
    num_epochs=10,
    patch_size=3,
    pixels_per_batch=8192,
    lr=1e-3,
    weight_decay=1e-6,
    grad_clip=None,
    device="cuda",
    train_region=None,
):
    """Train one model with on-the-fly multi-rate VVBP extraction.

    Each batch comes from a MultiRateFanbeamDataset (train=True) and
    contains a sinogram subsampled to a random V ∈ sparse_views.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    train_log = []

    target_mean = target_stats["target_mean"].to(device)
    target_std = target_stats["target_std"].to(device)

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        total_count = 0

        for sino_batch, img_batch in train_loader:
            sino_batch = sino_batch.to(device, non_blocking=True)
            img_batch = img_batch.to(device, non_blocking=True)

            V = sino_batch.shape[-2]

            # Select per-V extractor and stats.
            extractor = extractors[V]
            vs = v_stats[V]
            batch_stats = {
                "target_mean": target_mean,
                "target_std": target_std,
                "v_mean": vs["v_mean"].to(device),
                "v_std": vs["v_std"].to(device),
            }

            # On-the-fly VVBP extraction.
            vvbp = extractor(sino_batch)
            B, _, H, W, _ = vvbp.shape

            xs, ys = sample_random_coords(
                H, W, num_pixels=pixels_per_batch,
                margin=patch_size // 2, device=device,
                train_region=train_region,
            )
            values_sorted = gather_sorted_vvbp_patch(
                vvbp, xs, ys, patch_size=patch_size, mode="3x3",
            )  # [B, P, J, K] with B=1 (single sinogram)
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

            bs = y_norm.shape[0]
            total_loss += loss.item() * bs
            total_count += bs

        avg_loss = total_loss / max(total_count, 1)
        train_log.append(avg_loss)
        print(f"{model_name} | Epoch [{epoch:03d}/{num_epochs}] loss={avg_loss:.6f} time={time.time() - t0:.1f}s")

    return train_log

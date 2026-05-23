import math
import torch


def sample_random_coords(H, W, num_pixels, margin=1, device="cuda", train_region=None):
    """
    Sample random coordinates.

    If train_region is provided:
        train_region = [x0, x1, y0, y1]
        sample only from that region.
    Otherwise:
        sample from the whole image.
    """
    if train_region is None:
        x0, x1 = margin, H - margin
        y0, y1 = margin, W - margin
    else:
        x0, x1, y0, y1 = train_region
        x0 = max(int(x0), margin)
        x1 = min(int(x1), H - margin)
        y0 = max(int(y0), margin)
        y1 = min(int(y1), W - margin)

    xs = torch.randint(low=x0, high=x1, size=(num_pixels,), device=device)
    ys = torch.randint(low=y0, high=y1, size=(num_pixels,), device=device)
    return xs, ys

def get_relative_coords(patch_size=3, device="cuda"):
    """For 3x3: (-1,-1), (-1,0), ..., (1,1), normalized by radius."""
    r = patch_size // 2
    coords = []
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            coords.append([du / max(r, 1), dv / max(r, 1)])
    return torch.tensor(coords, dtype=torch.float32, device=device)


def gather_sorted_vvbp_patch(vvbp, xs, ys, patch_size=3, mode="3x3"):
    """Gather sorted VVBP values.

    vvbp: [B, 1, H, W, K]
    xs, ys: [P]
    return:
        mode="single": [B, P, 1, K]
        mode="3x3":   [B, P, patch_size*patch_size, K]
    """
    B, C, H, W, K = vvbp.shape
    assert C == 1
    assert mode == "3x3", f"Only mode='3x3' supported, got {mode}"
    if mode == "3x3":
        r = patch_size // 2
        patch_list = []
        for du in range(-r, r + 1):
            for dv in range(-r, r + 1):
                vals = vvbp[:, 0, xs + du, ys + dv, :]
                patch_list.append(torch.sort(vals, dim=-1).values)
        return torch.stack(patch_list, dim=2)
    raise ValueError(f"Unknown mode: {mode}")


def gather_raw_vvbp_patch(vvbp, xs, ys, patch_size=3):
    """Gather raw (unsorted) per-view VVBP values for each pixel's patch.

    Unlike ``gather_sorted_vvbp_patch``, this preserves the original view
    ordering so that ``values[b, p, j, v]`` is the VVBP value at view *v* for
    patch position *j* of pixel *p*.  Required for detector-coordinate-based
    G / R² computation where per-view correspondence must be exact.

    Args:
        vvbp: [B, 1, H, W, V]  VVBP tensor.
        xs, ys: [P] pixel coordinates.
        patch_size: odd integer (3 → 3×3).

    Returns:
        raw: [B, P, J, V]  where J = patch_size², views in original order.
    """
    r = patch_size // 2
    B = vvbp.shape[0]
    P = xs.shape[0]
    J = patch_size * patch_size
    V = vvbp.shape[4]
    out = torch.zeros(B, P, J, V, dtype=vvbp.dtype, device=vvbp.device)
    idx = 0
    for du in range(-r, r + 1):
        for dv in range(-r, r + 1):
            out[:, :, idx, :] = vvbp[:, 0, xs + du, ys + dv, :]
            idx += 1
    return out


@torch.no_grad()
def extract_cache_items_from_batch(
    sino_batch,
    img_batch,
    extractor,
    pixels_per_slice=8192,
    patch_size=3,
    device="cuda",
    train_region=None,
):
    """Extract cache tensors from one batch. All outputs are CPU tensors."""
    sino_batch = sino_batch.to(device, non_blocking=True)
    img_batch = img_batch.to(device, non_blocking=True)
    vvbp = extractor(sino_batch)
    B, _, H, W, K = vvbp.shape
    xs, ys = sample_random_coords(
        H,
        W,
        num_pixels=pixels_per_slice,
        margin=patch_size // 2,
        device=device,
        train_region=train_region,
    )
    values_sorted = gather_sorted_vvbp_patch(vvbp, xs, ys, patch_size=patch_size, mode="3x3")
    target = img_batch[:, 0, xs, ys].reshape(B * pixels_per_slice, 1)
    center_vals = vvbp[:, 0, xs, ys, :]
    center_base = center_vals.mean(dim=-1).reshape(B * pixels_per_slice, 1) * math.pi
    local_3x3_base = values_sorted.mean(dim=(2, 3)).reshape(B * pixels_per_slice, 1) * math.pi
    out = {
        "values_sorted": values_sorted.reshape(B * pixels_per_slice, values_sorted.shape[2], K).detach().cpu().float(),
        "target": target.detach().cpu().float(),
        "center_base": center_base.detach().cpu().float(),
        "local_3x3_base": local_3x3_base.detach().cpu().float(),
    }
    del vvbp, values_sorted, target, center_base, local_3x3_base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out

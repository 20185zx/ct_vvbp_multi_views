import os
import pickle
import math
from typing import Dict, Union, Optional

import numpy as np
import torch



def build_linfbp_geo(
    views: int = 720,
    device: Union[str, torch.device] = "cuda",
    image_size: int = 256,
    n_detec: int = 672,
    d_detec: float = 1.0,
    d_voxel: float = 1.0,
    DSO: float = 595.0,
    DOD: float = 480.0,
    start_angle: float = 0.0,
    end_angle: float = 2 * np.pi,
) -> Dict:
    """Build LInFBP-style fan-beam geometry and precompute w1/filter/w2.

    Defaults target the AAPM fan-beam multi-rate sparse-view setup:
      - 256×256 image, 672 detector elements
      - DSD = DSO + DOD = 1075 mm
      - Uniform angular sampling over [0, 2π)
    """
    DSD = DSO + DOD
    s_voxel = image_size * d_voxel
    s_detec = n_detec * d_detec

    geo = {
        "nVoxelX": image_size,
        "sVoxelX": s_voxel,
        "dVoxelX": d_voxel,
        "nVoxelY": image_size,
        "sVoxelY": s_voxel,
        "dVoxelY": d_voxel,
        "nDetecU": n_detec,
        "sDetecU": s_detec,
        "dDetecU": d_detec,
        "offOriginX": 0.0,
        "offOriginY": 0.0,
        "views": views,
        "slices": 1,
        "DSD": DSD,
        "DSO": DSO,
        "DOD": DOD,
        "start_angle": start_angle,
        "end_angle": end_angle,
        "mode": "fanflat",
        "extent": 1,
    }

    w = (geo["nDetecU"] - 1) / 2
    s = geo["dDetecU"] * (np.arange(geo["nDetecU"]) - w)
    gam = np.arctan(s / geo["DSD"])
    w1 = np.abs(geo["DSO"] * np.cos(gam)) / geo["DSD"]
    geo["w1"] = torch.from_numpy(w1.astype(np.float32)).to(device)

    npad = int(2 ** np.ceil(np.log2(2 * geo["nDetecU"] - 1)))
    nnp = np.arange(-(npad // 2), npad // 2)
    h = np.zeros_like(nnp, dtype=np.float64)
    h[npad // 2] = 1 / 4
    odd = (nnp % 2 == 1)
    h[odd] = -1 / (np.pi * nnp[odd]) ** 2
    h = h / (geo["dDetecU"] ** 2)

    Hk = np.real(np.fft.fft(np.fft.fftshift(h)))
    window = np.fft.fftshift(np.ones(npad, dtype=np.float64))
    Hk = Hk * window
    geo["filter"] = torch.from_numpy((Hk * geo["dDetecU"]).astype(np.float32)).to(device)

    betas = np.linspace(geo["start_angle"], geo["end_angle"], geo["views"], False)
    betas = np.expand_dims(np.expand_dims(betas, 0), 0)

    xc = np.arange(1, geo["nVoxelX"] + 1) - (geo["nVoxelX"] + 1) / 2
    yc = np.arange(1, geo["nVoxelY"] + 1) - (geo["nVoxelY"] + 1) / 2
    yc = np.flip(yc)
    xc = np.expand_dims(np.expand_dims(xc, -1), 0) * geo["dVoxelX"]
    yc = np.expand_dims(np.expand_dims(yc, -1), -1) * geo["dVoxelY"]
    d_loop = geo["DSO"] - xc * np.sin(betas) + yc * np.cos(betas)
    mag = geo["DSD"] / d_loop
    geo["w2"] = torch.from_numpy((mag ** 2).astype(np.float32)).to(device)
    return geo


def compute_deltas_cube_np(geo: Dict, alpha: float):
    P0 = {
        "x": -(geo["sVoxelX"] / 2 - geo["dVoxelX"] / 2) + geo["offOriginX"],
        "y": -(geo["sVoxelY"] / 2 - geo["dVoxelY"] / 2) + geo["offOriginY"],
    }
    Px0 = {"x": P0["x"] + geo["dVoxelX"], "y": P0["y"]}
    Py0 = {"x": P0["x"], "y": P0["y"] + geo["dVoxelY"]}

    P = {
        "x": P0["x"] * math.cos(alpha) - P0["y"] * math.sin(alpha),
        "y": P0["x"] * math.sin(alpha) + P0["y"] * math.cos(alpha),
    }
    Px = {
        "x": Px0["x"] * math.cos(alpha) - Px0["y"] * math.sin(alpha),
        "y": Px0["x"] * math.sin(alpha) + Px0["y"] * math.cos(alpha),
    }
    Py = {
        "x": Py0["x"] * math.cos(alpha) - Py0["y"] * math.sin(alpha),
        "y": Py0["x"] * math.sin(alpha) + Py0["y"] * math.cos(alpha),
    }
    P["y"] /= geo["dDetecU"]
    Px["y"] /= geo["dDetecU"]
    Py["y"] /= geo["dDetecU"]
    deltaX = {"x": Px["x"] - P["x"], "y": Px["y"] - P["y"]}
    deltaY = {"x": Py["x"] - P["x"], "y": Py["y"] - P["y"]}
    return P, deltaX, deltaY


def pixel_index_cal_numpy(geo: Dict, device: Union[str, torch.device] = "cuda", save_path: Optional[str] = None, verbose: bool = True):
    """Generate fixed linear backprojection indices without PyCUDA."""
    assert geo["extent"] == 1, "This implementation currently assumes extent=1."
    nVoxelX = geo["nVoxelX"]
    nVoxelY = geo["nVoxelY"]
    nDetecU = geo["nDetecU"]
    views = geo["views"]
    alphas = np.linspace(geo["start_angle"], geo["end_angle"], views, False)
    sino_indices = np.zeros((nVoxelX * nVoxelY, views), dtype=np.float32)
    indX, indY = np.meshgrid(np.arange(nVoxelX, dtype=np.float32), np.arange(nVoxelY, dtype=np.float32), indexing="ij")
    indX_flat = indX.reshape(-1)
    indY_flat = indY.reshape(-1)

    for angle_idx in range(views):
        alpha = -alphas[angle_idx]
        origin, deltaX, deltaY = compute_deltas_cube_np(geo, alpha)
        P_x = origin["x"] + indX_flat * deltaX["x"] + indY_flat * deltaY["x"]
        P_y = origin["y"] + indX_flat * deltaX["y"] + indY_flat * deltaY["y"]
        S_x = geo["DSO"]
        S_y = 0.0 if geo["mode"] == "fanflat" else P_y
        vectX = P_x - S_x
        vectY = P_y - S_y
        t = (geo["DSO"] - geo["DSD"] - S_x) / vectX
        y = vectY * t + S_y
        detindx = y + nDetecU / 2 - 0.5
        detindx = np.clip(detindx, 1, nDetecU - 2)
        tmp_index = detindx + nDetecU * angle_idx
        sino_indices[:, angle_idx] = tmp_index.astype(np.float32)
        if verbose and angle_idx % 10 == 0:
            print(f"Generated indices for view {angle_idx}/{views}")

    indices = torch.from_numpy(sino_indices.reshape(-1)).float()
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(indices, save_path)
        print("Saved indices to:", save_path)
    return indices.to(device)


def load_or_generate_geo(
    views: int,
    results_folder: str,
    device: Union[str, torch.device],
    image_size: int = 256,
    n_detec: int = 672,
    d_detec: float = 1.0,
    d_voxel: float = 1.0,
    DSO: float = 595.0,
    DOD: float = 480.0,
):
    """Build LInFBP-style fan-beam geo and load/generate backprojection indices."""
    geo = build_linfbp_geo(
        views=views,
        device=device,
        image_size=image_size,
        n_detec=n_detec,
        d_detec=d_detec,
        d_voxel=d_voxel,
        DSO=DSO,
        DOD=DOD,
    )
    indices_path = os.path.join(results_folder, f"test_{image_size}_{views}_fan_numpy.dat")
    if os.path.exists(indices_path):
        print(f"Loading indices for {views} views:", indices_path)
        try:
            indices = torch.load(indices_path, map_location=device, weights_only=True)
        except (pickle.UnpicklingError, RuntimeError):
            # Fallback for legacy pickle-format cache files
            with open(indices_path, "rb") as f:
                indices = pickle.load(f).to(device)
        geo["indices"] = indices
    else:
        print(f"Generating indices for {views} views:", indices_path)
        geo["indices"] = pixel_index_cal_numpy(geo, device=device, save_path=indices_path, verbose=True)
    print(f"geo {views} indices:", geo["indices"].shape)
    return geo

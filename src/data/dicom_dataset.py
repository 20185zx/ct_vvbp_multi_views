from __future__ import annotations

import os
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
import pydicom
import astra


def _sort_dicom_files(files):
    items = []
    for f in files:
        ds = pydicom.dcmread(f, stop_before_pixels=True)
        if hasattr(ds, "ImagePositionPatient"):
            key = float(ds.ImagePositionPatient[2])
        else:
            key = float(getattr(ds, "InstanceNumber", 0))
        items.append((key, f))
    return [x[1] for x in sorted(items, key=lambda x: x[0])]


def _read_hu(file_path):
    ds = pydicom.dcmread(file_path)
    img = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return img * slope + intercept


def _hu_to_attenuation(hu, water_mu=0.0192):
    atten = (hu / 1000.0 + 1.0) * water_mu
    atten = np.clip(atten, 0.0, None)
    return atten.astype(np.float32)


class LInFBPAlignedDataset(Dataset):
    """DICOM/IMA -> HU -> attenuation -> ASTRA fan-beam sinogram.

    The geometry matches the LInFBP-style fan-beam geometry used in the notebook.
    When ``geo`` is None, defaults are built from the config-style keyword arguments.
    """
    def __init__(
        self,
        dicom_folder,
        views=100,
        water_mu=0.0192,
        geo=None,
        image_size=512,
        n_detec=736,
        d_detec=0.6848 * 2,
        d_voxel=0.6641,
        DSO=595.0,
        DOD=490.6,
    ):
        self.dicom_folder = str(dicom_folder)
        self.views = views
        self.water_mu = water_mu
        self.image_size = image_size

        if geo is not None:
            self.geo = geo.copy()
        else:
            s_voxel = image_size * d_voxel
            s_detec = n_detec * d_detec
            self.geo = {
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
                "DSD": DSO + DOD,
                "DSO": DSO,
                "DOD": DOD,
                "start_angle": 0.0,
                "end_angle": 2 * np.pi,
                "mode": "fanflat",
                "extent": 1,
            }

        self.files = [
            os.path.join(self.dicom_folder, f)
            for f in os.listdir(self.dicom_folder)
            if f.lower().endswith((".ima", ".dcm"))
        ]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .IMA or .dcm files found in: {self.dicom_folder}")
        self.files = _sort_dicom_files(self.files)

        self.vol_geom = astra.create_vol_geom(
            self.geo["nVoxelY"],
            self.geo["nVoxelX"],
            -self.geo["sVoxelY"] / 2 + self.geo["offOriginY"],
            self.geo["sVoxelY"] / 2 + self.geo["offOriginY"],
            -self.geo["sVoxelX"] / 2 + self.geo["offOriginX"],
            self.geo["sVoxelX"] / 2 + self.geo["offOriginX"],
        )
        self.proj_geom = astra.create_proj_geom(
            self.geo["mode"],
            self.geo["dDetecU"],
            self.geo["nDetecU"],
            np.linspace(self.geo["start_angle"], self.geo["end_angle"], self.geo["views"], False),
            self.geo["DSO"],
            self.geo["DOD"],
        )
        self.proj_id = astra.create_projector("line_fanflat", self.proj_geom, self.vol_geom)

        print("LInFBP-aligned dataset initialized.")
        print("Number of slices:", len(self.files))
        print("Views:", self.geo["views"])
        print("Image:", image_size, "x", image_size)
        print("Detector:", self.geo["nDetecU"])
        print("Geometry:", self.geo["mode"])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hu = _read_hu(self.files[idx])
        img = _hu_to_attenuation(hu, self.water_mu)

        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
            img_t = F.interpolate(
                img_t, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            img = img_t.squeeze().numpy()

        sino_id, sino = astra.create_sino(img, self.proj_id)
        astra.data2d.delete(sino_id)
        sino_tensor = torch.from_numpy(sino.astype(np.float32)).unsqueeze(0)
        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return sino_tensor, img_tensor


def build_dataloaders(dicom_folder, cfg):
    sv = cfg.sparse_views
    views = sv[0] if isinstance(sv, list) else sv
    dataset = LInFBPAlignedDataset(
        dicom_folder=dicom_folder,
        views=views,
        image_size=cfg.image_size,
        n_detec=cfg.n_detec,
        d_detec=cfg.d_detec,
        d_voxel=cfg.d_voxel,
        DSO=cfg.DSO,
        DOD=cfg.DOD,
    )
    indices = np.arange(len(dataset))
    split = int(0.8 * len(dataset))
    train_indices = indices[:split]
    test_indices = indices[split:]
    train_loader = DataLoader(
        Subset(dataset, train_indices),
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    return dataset, train_indices, test_indices, train_loader


class MultiRateFanbeamDataset(Dataset):
    """DICOM → resize → 720-view full sinogram → on-the-fly subsample to V views.

    Training mode (``train=True``): each ``__getitem__`` randomly chooses
    ``V`` uniformly from ``sparse_views``.

    Evaluation mode (``train=False``): returns the full 720-view sinogram;
    subsampling is done externally so the caller can iterate over V.
    """

    def __init__(
        self,
        dicom_folder: str,
        image_size: int = 256,
        full_views: int = 720,
        n_detec: int = 672,
        d_detec: float = 1.0,
        d_voxel: float = 1.0,
        DSO: float = 595.0,
        DOD: float = 480.0,
        water_mu: float = 0.0192,
        sparse_views: list | None = None,
        train: bool = True,
    ):
        self.dicom_folder = str(dicom_folder)
        self.image_size = image_size
        self.full_views = full_views
        self.water_mu = water_mu
        self.train = train
        self.sparse_views = sparse_views or [9, 18, 36, 72]

        # Build geometry dict for ASTRA forward projection (full 720 views).
        s_voxel = image_size * d_voxel
        s_detec = n_detec * d_detec
        self.geo = {
            "nVoxelX": image_size,
            "nVoxelY": image_size,
            "sVoxelX": s_voxel,
            "sVoxelY": s_voxel,
            "dVoxelX": d_voxel,
            "dVoxelY": d_voxel,
            "nDetecU": n_detec,
            "dDetecU": d_detec,
            "sDetecU": s_detec,
            "offOriginX": 0.0,
            "offOriginY": 0.0,
            "views": full_views,
            "DSO": DSO,
            "DOD": DOD,
            "DSD": DSO + DOD,
            "start_angle": 0.0,
            "end_angle": 2 * np.pi,
            "mode": "fanflat",
        }

        self.files = [
            os.path.join(self.dicom_folder, f)
            for f in os.listdir(self.dicom_folder)
            if f.lower().endswith((".ima", ".dcm"))
        ]
        if len(self.files) == 0:
            raise FileNotFoundError(f"No .IMA or .dcm files found in: {self.dicom_folder}")
        self.files = _sort_dicom_files(self.files)

        self.vol_geom = astra.create_vol_geom(
            self.geo["nVoxelY"],
            self.geo["nVoxelX"],
            -self.geo["sVoxelY"] / 2 + self.geo["offOriginY"],
            self.geo["sVoxelY"] / 2 + self.geo["offOriginY"],
            -self.geo["sVoxelX"] / 2 + self.geo["offOriginX"],
            self.geo["sVoxelX"] / 2 + self.geo["offOriginX"],
        )
        angles_full = np.linspace(
            self.geo["start_angle"], self.geo["end_angle"], self.geo["views"], False
        )
        self.proj_geom = astra.create_proj_geom(
            self.geo["mode"],
            self.geo["dDetecU"],
            self.geo["nDetecU"],
            angles_full,
            self.geo["DSO"],
            self.geo["DOD"],
        )
        self.proj_id = astra.create_projector("line_fanflat", self.proj_geom, self.vol_geom)

        print("MultiRateFanbeamDataset initialized.")
        print("  Slices:", len(self.files))
        print("  Image:", image_size, "x", image_size)
        print("  Full views:", full_views, "| Detectors:", n_detec)
        print("  DSD:", self.geo["DSD"], "| DSO:", self.geo["DSO"], "| DOD:", self.geo["DOD"])
        print("  Train mode:", train, "| Sparse views:", self.sparse_views)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        hu = _read_hu(self.files[idx])
        img = _hu_to_attenuation(hu, self.water_mu)

        # Resize to target image_size × image_size via bilinear interpolation.
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
            img_t = F.interpolate(
                img_t, size=(self.image_size, self.image_size),
                mode="bilinear", align_corners=False,
            )
            img = img_t.squeeze().numpy()

        sino_id, sino_full = astra.create_sino(img, self.proj_id)
        astra.data2d.delete(sino_id)

        if self.train:
            V = int(np.random.choice(self.sparse_views))
            step = self.full_views // V
            sino_sparse = sino_full[::step, :].copy()
            sino_tensor = torch.from_numpy(sino_sparse.astype(np.float32)).unsqueeze(0)
        else:
            sino_tensor = torch.from_numpy(sino_full.astype(np.float32)).unsqueeze(0)

        img_tensor = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        return sino_tensor, img_tensor


def build_multirate_dataloaders(dicom_folder, cfg):
    """Build dataloaders for multi-rate sparse-view training.

    Training: each batch randomly selects V ∈ cfg.sparse_views.
    Evaluation: returns full 720-view sinograms (subsample externally per V).
    """
    train_dataset = MultiRateFanbeamDataset(
        dicom_folder=str(dicom_folder),
        image_size=cfg.image_size,
        full_views=cfg.full_views,
        n_detec=cfg.n_detec,
        d_detec=cfg.d_detec,
        d_voxel=cfg.d_voxel,
        DSO=cfg.DSO,
        DOD=cfg.DOD,
        sparse_views=cfg.sparse_views,
        train=True,
    )
    eval_dataset = MultiRateFanbeamDataset(
        dicom_folder=str(dicom_folder),
        image_size=cfg.image_size,
        full_views=cfg.full_views,
        n_detec=cfg.n_detec,
        d_detec=cfg.d_detec,
        d_voxel=cfg.d_voxel,
        DSO=cfg.DSO,
        DOD=cfg.DOD,
        sparse_views=cfg.sparse_views,
        train=False,
    )

    indices = np.arange(len(train_dataset))
    split = int(0.8 * len(train_dataset))
    train_indices = indices[:split]
    test_indices = indices[split:]

    train_loader = DataLoader(
        Subset(train_dataset, train_indices),
        batch_size=cfg.train_batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False,
    )
    return train_dataset, eval_dataset, train_indices, test_indices, train_loader

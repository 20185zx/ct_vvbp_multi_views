import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_metrics_np(pred, ref):
    pred = pred.astype(np.float32)
    ref = ref.astype(np.float32)
    mse = float(np.mean((pred - ref) ** 2))
    mae = float(np.mean(np.abs(pred - ref)))
    data_range = float(ref.max() - ref.min()) + 1e-8
    psnr = float(peak_signal_noise_ratio(ref, pred, data_range=data_range))
    ssim = float(structural_similarity(ref, pred, data_range=data_range))
    return {"MSE": mse, "MAE": mae, "PSNR": psnr, "SSIM": ssim}
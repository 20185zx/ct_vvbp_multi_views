# Code Status

## Kept model code

The cleaned `src/models/` folder keeps only the current model candidates:

- `local_rank_center_integral_mlp.py`: current main learnable local-rank integral model.
- `local_rank_center_closed_mlp.py`: learnable closed-interval comparison model.
- `local_rank_center_mlp.py`: non-integral local-rank MLP comparison model.
- `merged_local_sorted.py`: merged 3×3 sorted VVBP comparison model.
- `sino_upsample_residual_fbp.py`: separate sinogram upsampling + fixed FBP baseline.

`model_factory.py` now registers only the current direct VVBP models used by `run_from_config.py`.

## Main experiment output

A selected-model run saves:

- trained checkpoint `.pt`
- `stats_cached.pt`
- `selected_model_full_metrics.csv`
- `selected_model_psnr_ssim.csv`
- `prediction_distribution_stats.csv`
- comparison figure `.png`

## Removed/cleaned

- Old model registry entries for deleted models.
- Broken default config pointing to old `basic coord` models.
- Duplicated commented code in the feature builder and evaluation path.
- Empty/generated folders from the shared zip archive.

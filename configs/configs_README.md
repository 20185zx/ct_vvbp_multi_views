# Configs README

## Local-rank VVBP configs

```bash
python scripts/run_local_rank_center_integral.py --config configs/patch_size_3_local_rank_center_integral_mlp.json
```

Common fields:

- `model_names`: model display names registered in `src/models/model_factory.py`.
- `region`: evaluation region `[x_start, x_end, y_start, y_end]`.
- `cache_dir`: cache folder for train/eval VVBP patch tensors.
- `rebuild_train_cache`: set `true` when data extraction settings change.
- `rebuild_eval_cache`: set `true` when evaluation region or extraction settings change.
- `eval_only`: set `true` to load checkpoints from `checkpoint_dir` instead of training.

Currently registered local-rank VVBP model names:

- `local rank center integral mlp, 10 epochs`
- `local rank center closed mlp, 10 epochs`
- `local rank center mlp, 10 epochs`
- `merged local sorted, 10 epochs`

## CTO-adapted multirate configs

```bash
python scripts/run_cto_multirate.py --config configs/cto_multirate.json
```

Key CTO-specific fields:

- `cto_sino_hidden`, `cto_image_hidden`: hidden channel dimensions for sinogram/image-space operators.
- `cto_cascades`: number of cascade blocks.
- `cto_sino_residual_scale`, `cto_image_reg_scale`: residual scaling factors.
- `cto_dc_step`: data-consistency step size.
- `udno_pools`: number of UDNO pooling layers.
- `udno_radius_cutoff`: spectral radius cutoff.
- `udno_kernel_shape`: convolutional kernel shape `[angle, channel]`.

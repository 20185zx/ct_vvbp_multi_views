# Configs README

Use the current main config first:

```bash
python scripts/run_from_config.py --config configs/patch_size_3_local_rank_center_integral_mlp.json
```

Common fields:

- `model_names`: model display names registered in `src/models/model_factory.py`.
- `region`: evaluation region `[x_start, x_end, y_start, y_end]`.
- `cache_dir`: cache folder for train/eval VVBP patch tensors.
- `rebuild_train_cache`: set `true` when data extraction settings change.
- `rebuild_eval_cache`: set `true` when evaluation region or extraction settings change.
- `eval_only`: set `true` to load checkpoints from `checkpoint_dir` instead of training.

Currently registered direct VVBP model names:

- `local rank center integral mlp, 10 epochs`
- `local rank center closed mlp, 10 epochs`
- `local rank center mlp, 10 epochs`
- `merged local sorted, 10 epochs`

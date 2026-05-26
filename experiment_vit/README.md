# ImageNet-1K DeiT Pretraining

This folder contains a standalone ImageNet-1K training entry for:

- `deit_tiny_patch16_224`
- `deit_small_patch16_224`

It uses `timm.create_model(..., pretrained=False)`, so the model is trained from scratch. Timm initializes DeiT through `VisionTransformer.init_weights()`:

- `pos_embed`: truncated normal, std 0.02
- `cls_token`: normal, std 1e-6
- `nn.Linear`: truncated normal, std 0.02, zero bias
- modules with their own `init_weights()`, such as patch embedding, use their module init

After model creation, `train_imagenet.py` imports local `initial_scheme.py` and calls `apply_initialization(model, args)`. Use `--initialization default` to keep timm initialization. Add SFLI variants in `initial_scheme.py` for future experiments.

## ImageNet-1K Data

Download ImageNet-1K manually from the official source, then arrange it as an ImageFolder tree:

```text
/path/to/imagenet
  train/
    n01440764/
      *.JPEG
    ...
  val/
    n01440764/
      *.JPEG
    ...
```

The script does not download ImageNet automatically because ImageNet requires registration and license acceptance.

## Run

Tiny:

```bash
cd /home/qjw/code/python_code/LLM/mix_LN/MixLN-main
bash experiment_vit/scripts/run_deit_tiny.sh /path/to/imagenet
```

Small:

```bash
cd /home/qjw/code/python_code/LLM/mix_LN/MixLN-main
bash experiment_vit/scripts/run_deit_small.sh /path/to/imagenet
```

Direct torchrun:

```bash
torchrun --nproc_per_node 2 experiment_vit/train_imagenet.py \
  --config experiment_vit/configs/deit_tiny_patch16_224_120ep.json \
  --data-dir /path/to/imagenet
```


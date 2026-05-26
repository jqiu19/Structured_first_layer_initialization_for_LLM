#!/usr/bin/env bash
set -euo pipefail

DATA_DIR=${1:?Usage: bash experiment_vit/scripts/run_deit_small.sh /path/to/imagenet [nproc]}
NPROC=${2:-2}

torchrun --nproc_per_node "$NPROC" experiment_vit/train_imagenet.py \
  --config experiment_vit/configs/deit_small_patch16_224_120ep.json \
  --data-dir "$DATA_DIR"


#!/usr/bin/env bash
set -euo pipefail

# Real PushT + LeWM ViT/projector + Fast-LeWAM DiT Lightning smoke.
# ConstantLR avoids a zero-length cosine interval in a one-step fast_dev_run.
python -u train.py \
  --config-name=fast_lewam \
  data=pusht \
  +trainer.fast_dev_run=true \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  trainer.precision=32 \
  loader.batch_size=2 \
  loader.num_workers=0 \
  loader.persistent_workers=false \
  '~loader.prefetch_factor' \
  latent_head_layers=1 \
  latent_head_dim=48 \
  loss.sigreg.kwargs.num_proj=8 \
  wandb.enabled=false \
  +policy.scheduler.type=ConstantLR \
  +policy.scheduler.factor=1.0 \
  +policy.scheduler.total_iters=1 \
  hydra.run.dir="${FAST_LEWAM_SMOKE_DIR:-/tmp/fast_lewam_smoke_$$}"

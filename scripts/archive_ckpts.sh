#!/bin/bash
# Copy every checkpoint of the given runs into runs/RUN/checkpoints_all/ as soon as it is fully
# written (train.py's default num_ckpt_keep 3 deletes older ones; pair 1 ran with it). Stops when
# every run has checkpoint_000250 archived. Usage: scripts/archive_ckpts.sh RUN...
cd /root/moshi-finetune
while true; do
  all=1
  for R in "$@"; do
    for d in runs/$R/checkpoints/checkpoint_*; do
      [ -d "$d/consolidated" ] || continue
      dst=runs/$R/checkpoints_all/$(basename $d)
      [ -f $dst/consolidated/lora.safetensors ] && continue
      f=$d/consolidated/lora.safetensors
      [ -f $f ] && [ -f $d/consolidated/config.json ] || continue
      s1=$(stat -c %s $f); sleep 15; s2=$(stat -c %s $f 2>/dev/null || echo 0)
      [ "$s1" = "$s2" ] || continue
      mkdir -p $dst && cp -r $d/consolidated $dst/ && echo "$(date +%H:%M:%S) archived $dst"
    done
    [ -f runs/$R/checkpoints_all/checkpoint_000250/consolidated/lora.safetensors ] || all=0
  done
  [ $all = 1 ] && break
  sleep 20
done
echo "$(date +%H:%M:%S) all archived"

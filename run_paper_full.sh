#!/bin/bash
cd /workspace/truongan/crop_nav/xinyanlin
PY=/workspace/truongan/crop_nav/venv310/bin/python
while [ "$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)" -lt 6000 ]; do
  echo "wait: free $(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1) MiB"; sleep 60
done
$PY tools/train.py --experiment full --seed 0 --data-root data/cotton_two_stage --work-dir work_dirs/paper_full_s0

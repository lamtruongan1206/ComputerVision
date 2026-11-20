#!/bin/bash
cd /workspace/truongan/crop_nav/xinyanlin
PY=/workspace/truongan/crop_nav/venv310/bin/python
NEED=16000
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
W=work_dirs/paper_full_s0
tries=0
while [ ! -f $W/iter_20000.pth ]; do
  tries=$((tries+1)); [ $tries -gt 300 ] && { echo "give up"; break; }
  while [ "$(free_mb)" -lt $NEED ]; do echo "$(date +%T) wait: free $(free_mb) MiB"; sleep 60; done
  echo "=== try $tries $(date)"
  $PY tools/train.py --experiment full --seed 0 --data-root data/cotton_two_stage --work-dir $W >> logs/paper_full_s0.txt 2>&1
  if [ ! -f $W/iter_20000.pth ] && ! tail -n 60 logs/paper_full_s0.txt | grep -q "out of memory"; then
    echo "non-OOM error, stop"; exit 1
  fi
  sleep 5
done
echo "DONE $(date)"

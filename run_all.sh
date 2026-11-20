#!/bin/bash
cd /workspace/truongan/crop_nav/xinyanlin
PY=/workspace/truongan/crop_nav/venv310/bin/python
NEED=5000
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
for EXP in baseline full ca_mkir bwr_bag; do
  W=work_dirs/${EXP}_s0
  tries=0
  while [ ! -f $W/iter_10000.pth ]; do
    tries=$((tries+1)); [ $tries -gt 200 ] && { echo "give up $EXP"; break; }
    while [ "$(free_mb)" -lt $NEED ]; do echo "$(date +%T) wait: free $(free_mb) MiB"; sleep 60; done
    echo "=== $EXP try $tries $(date)"
    $PY tools/train.py --config configs/exp.py --experiment $EXP --seed 0 \
      --data-root data/cotton_709 --work-dir $W >> logs/$EXP.txt 2>&1
    if [ ! -f $W/iter_10000.pth ] && ! tail -n 40 logs/$EXP.txt | grep -q "out of memory"; then
      echo "non-OOM error in $EXP, stop"; exit 1
    fi
    sleep 5
  done
done
echo "ALL DONE $(date)"

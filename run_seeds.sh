#!/bin/bash
cd /workspace/truongan/crop_nav/xinyanlin
PY=/workspace/truongan/crop_nav/venv310/bin/python
NEED=5000
free_mb() { nvidia-smi --query-gpu=memory.total,memory.used --format=csv,noheader,nounits | awk -F', ' '{print $1-$2}'; }
for SEED in 1 3407; do
 for EXP in baseline full; do
  W=work_dirs/${EXP}_s${SEED}
  LOG=logs/${EXP}_s${SEED}.txt
  tries=0
  while [ ! -f $W/iter_10000.pth ]; do
    tries=$((tries+1)); [ $tries -gt 200 ] && { echo "give up $EXP s$SEED"; break; }
    while [ "$(free_mb)" -lt $NEED ]; do echo "$(date +%T) wait: free $(free_mb) MiB"; sleep 60; done
    echo "=== $EXP seed $SEED try $tries $(date)"
    $PY tools/train.py --config configs/exp.py --experiment $EXP --seed $SEED \
      --data-root data/cotton_709 --work-dir $W >> $LOG 2>&1
    if [ ! -f $W/iter_10000.pth ] && ! tail -n 40 $LOG | grep -q "out of memory"; then
      echo "non-OOM error in $EXP s$SEED, stop"; exit 1
    fi
    sleep 5
  done
 done
done
echo "SEEDS DONE $(date)"

#!/bin/bash
# Re-run the raw (failure-case) and pca (sensitivity) variants correctly.
# The earlier run failed because estimate wrote _boot_*.npz before the variant
# results dir existed; those dirs now exist, so estimate -> consolidate works.
set -uo pipefail
cd /home/woody/iwi5/iwi5406h/New_Sp
PY=.venv/bin/python
LOG=logs/sensitivity_runs.log
: > "$LOG"
IVS="fluids_sepsis prone_positioning rrt_timing transfusion_threshold"
for V in raw pca; do
  # ensure the variant results dir exists before estimate (write templates first)
  $PY main_causal.py --stage consolidate --variant "$V" >>"$LOG" 2>&1
  for IV in $IVS; do
    echo "## $V estimate $IV" >>"$LOG"
    $PY main_causal.py --stage estimate --variant "$V" --intervention "$IV" >>"$LOG" 2>&1
  done
  echo "## $V consolidate" >>"$LOG"
  $PY main_causal.py --stage consolidate --variant "$V" >>"$LOG" 2>&1
done
echo "SENSITIVITY_DONE" >>"$LOG"

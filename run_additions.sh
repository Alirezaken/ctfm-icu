#!/bin/bash
set -uo pipefail
cd /home/woody/iwi5/iwi5406h/New_Sp
PY=.venv/bin/python
LOG=logs/additions_run.log
: > "$LOG"
for IV in fluids_sepsis transfusion_threshold rrt_timing prone_positioning; do
  echo "## estimate $IV" >>"$LOG"
  $PY main_causal.py --stage estimate --intervention "$IV" >>"$LOG" 2>&1
done
echo "## external eICU" >>"$LOG"
$PY main_causal.py --stage external >>"$LOG" 2>&1
echo "## consolidate" >>"$LOG"
$PY main_causal.py --stage consolidate >>"$LOG" 2>&1
echo "## integrity" >>"$LOG"
$PY main_causal.py --stage integrity_check >>"$LOG" 2>&1
echo "ADDITIONS_DONE" >>"$LOG"

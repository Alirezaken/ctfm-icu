#!/bin/bash -l
# SLURM batch job: build the MIMIC-IV unified event-stream (#2).
#
# The work is CPU/IO-bound (pandas read + Parquet write) and does NOT use a GPU,
# but this is the TinyGPU cluster where every job MUST allocate >=1 GPU, so we
# request one (it sits idle). All six sources run sequentially in one job to avoid
# tying up several GPUs.
#
# Submit:   sbatch pipeline/submit_event_stream.sh
# Monitor:  squeue -u $USER   ;   tail -f logs/event_stream_*.log
#
#SBATCH --job-name=event_stream
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwi5/iwi5406h/New_Sp/logs/%x_%j.log

unset SLURM_EXPORT_ENV
set -euo pipefail

PROJECT=/home/woody/iwi5/iwi5406h/New_Sp
PY="$PROJECT/.venv/bin/python"   # self-contained venv; works under --export=NONE

cd "$PROJECT"
echo "host=$(hostname)  start=$(date)"
"$PY" -u pipeline/build_event_stream.py all
echo "all sources done  end=$(date)"

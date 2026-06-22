#!/bin/bash -l
# SLURM batch job: value-clean the event-stream (raw -> data/event_stream_clean/).
# Reversible: raw data/event_stream/ is never modified.
#
# Submit:   sbatch pipeline/submit_clean_event_stream.sh
# Monitor:  squeue -u $USER ; tail -f logs/clean_event_stream_*.log
#
#SBATCH --job-name=clean_event_stream
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwi5/iwi5406h/New_Sp/logs/%x_%j.log

unset SLURM_EXPORT_ENV
set -euo pipefail

PROJECT=/home/woody/iwi5/iwi5406h/New_Sp
PY="$PROJECT/.venv/bin/python"

cd "$PROJECT"
echo "host=$(hostname)  start=$(date)"
"$PY" -u pipeline/clean_event_stream.py all
echo "clean done; running QC on cleaned stream"
"$PY" -u pipeline/qc_event_stream.py clean
echo "ALL_DONE  end=$(date)"

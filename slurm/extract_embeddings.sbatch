#!/bin/bash -l
# §9.4 extract_embeddings -- the one GPU stage (RAD-DINO + Clinical-Longformer).
# Uses sp_env (torch+transformers+CUDA); the CPU causal .venv has no torch.
#
# Submit:  sbatch slurm/extract_embeddings.sbatch
# Monitor: squeue -u $USER ; tail -f logs/extract_embeddings_*.log
#
#SBATCH --job-name=extract_embeddings
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwi5/iwi5406h/New_Sp/logs/%x_%j.log

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-conda
source /home/hpc/iwi5/iwi5406h/fix10april/Project/sp_env/bin/activate

# compute nodes have NO internet; use the pre-cached models, never hit the network
export HF_HOME=/home/hpc/iwi5/iwi5406h/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/woody/iwi5/iwi5406h/New_Sp
echo "host=$(hostname)  python=$(which python)  start=$(date)"
python -c "import torch; print('cuda available:', torch.cuda.is_available())"
python -u main_causal.py --stage extract_embeddings
echo "extract_embeddings DONE  end=$(date)"

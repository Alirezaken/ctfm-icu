#!/bin/bash -l
# §3 quality gate -- RAD-DINO embeddings for the external CXR sets
# (PadChest, ChestX-ray14, and CheXpert once uploaded). Uses sp_env (torch+CUDA).
#
# Submit:  sbatch slurm/extract_external_cxr.sbatch
# Monitor: squeue -u $USER ; tail -f logs/extract_external_cxr_*.log
#
#SBATCH --job-name=extract_external_cxr
#SBATCH --partition=work
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
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
python -u main_causal.py --stage extract_external_cxr
echo "extract_external_cxr DONE  end=$(date)"

#!/bin/bash -l
# Robustness encoder swap: embed MIMIC frontal CXRs with BiomedCLIP -> images_alt.
# Uses sp_env (torch + open_clip). Compute nodes have no internet, so run offline
# against the pre-cached BiomedCLIP weights.
#
# Submit:  sbatch slurm/extract_images_alt.sbatch
#
#SBATCH --job-name=images_alt
#SBATCH --partition=a100
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=8
#SBATCH --export=NONE
#SBATCH --output=/home/woody/iwi5/iwi5406h/New_Sp/logs/%x_%j.log

unset SLURM_EXPORT_ENV
set -euo pipefail

module load python/3.12-conda
source /home/hpc/iwi5/iwi5406h/fix10april/Project/sp_env/bin/activate
export HF_HOME=/home/hpc/iwi5/iwi5406h/.cache/huggingface
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/woody/iwi5/iwi5406h/New_Sp
echo "host=$(hostname)  start=$(date)"
python -c "import torch,open_clip; print('cuda',torch.cuda.is_available(),'open_clip',open_clip.__version__)"
python -u main_causal.py --stage extract_images_alt
echo "images_alt DONE  end=$(date)"

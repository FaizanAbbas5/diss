#!/bin/bash
#SBATCH --job-name=t2t-train
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Arm-2 training. Submit from the repo root:
#   sbatch hpc/submit_train.sh configs/arm2_e2e_overfit.yaml
# Resumable: re-submitting the same config picks up from the last
# checkpoint in the hashed run dir (checkpoint_every steps). For clean
# bit-reproducibility prefer uninterrupted runs (audit P6: resume regroups
# batches, which can flip greedy tokens later).
set -euo pipefail

CONFIG=${1:?usage: sbatch hpc/submit_train.sh <config.yaml>}

SCRATCH=/scratch/users/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1
# 7B training sits near the 20GB slice ceiling; without this the allocator
# fragments (~4GB reserved-but-unallocated at the first OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

source /opt/software/uoa/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$SCRATCH/envs/t2t"

ENV_PREFIX=$SCRATCH/envs/t2t
NVLIBS=$(echo "$ENV_PREFIX"/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== node: $(hostname)  config: $CONFIG"
# no `| head` here: under pipefail, head's early exit SIGPIPEs nvidia-smi
# and set -e kills the job before training starts
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())"

python experiments/run_training.py --config "$CONFIG"

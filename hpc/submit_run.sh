#!/bin/bash
#SBATCH --job-name=t2t
#SBATCH --output=hpc/logs/%x-%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:3g.20gb:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00

# Generate + evaluate one config. Submit from the repo root:
#   sbatch hpc/submit_run.sh configs/e2e_baseline_colab20.yaml
# Override the walltime per job with e.g.:
#   sbatch --time=00:30:00 hpc/submit_run.sh configs/e2e_baseline_colab20.yaml
# Generation is resumable: resubmitting the same config skips finished items.
set -euo pipefail

CONFIG=${1:?usage: sbatch hpc/submit_run.sh <config.yaml>}

SCRATCH=/scratch/users/$USER
export HF_HOME=$SCRATCH/hf_cache
export HF_HUB_OFFLINE=1        # models were pre-downloaded by hpc/setup_env.sh
export PYTHONUNBUFFERED=1

source /opt/software/uoa/apps/miniconda3/etc/profile.d/conda.sh
conda activate "$SCRATCH/envs/t2t"

# bitsandbytes 0.42 needs (a) the modern libstdc++ installed in the env and
# (b) the CUDA runtime libs that ship inside the torch cu121 wheel — neither
# is on the library path by default on this cluster.
ENV_PREFIX=$SCRATCH/envs/t2t
NVLIBS=$(echo "$ENV_PREFIX"/lib/python3.11/site-packages/nvidia/*/lib | tr ' ' ':')
export LD_LIBRARY_PATH="$ENV_PREFIX/lib:$NVLIBS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "== node: $(hostname)  config: $CONFIG"
nvidia-smi
python -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU ONLY')"

python experiments/run_generation.py --config "$CONFIG"
python experiments/run_eval.py       --config "$CONFIG" --claims --parent
python experiments/show_results.py   --config "$CONFIG" --n 5

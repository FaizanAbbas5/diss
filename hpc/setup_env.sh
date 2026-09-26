#!/bin/bash
# One-time HPC setup. Run on the login node from the repo root:
#   bash hpc/setup_env.sh
# Creates an isolated conda env on scratch (cluster python module is 3.9,
# t2t needs >=3.10), installs the package, and pre-downloads both models so
# compute nodes never need internet access.
set -euo pipefail

SCRATCH=/scratch/users/$USER
ENV_PREFIX=$SCRATCH/envs/t2t
export HF_HOME=$SCRATCH/hf_cache

mkdir -p "$HF_HOME" hpc/logs

source /opt/software/uoa/apps/miniconda3/etc/profile.d/conda.sh

# conda-forge, not defaults: the cluster OS has an old glibc (2.17-era) and
# defaults-channel python builds now require glibc>=2.28 and segfault here.
if [ ! -d "$ENV_PREFIX" ]; then
    conda create -y -p "$ENV_PREFIX" -c conda-forge --override-channels python=3.11
fi
conda activate "$ENV_PREFIX"

# Fail fast if the python build is incompatible with the node's glibc
python --version

# Modern C++ runtime: system libstdc++ is GCC 4.8-era and bitsandbytes'
# CUDA binary needs CXXABI_1.3.9 (GCC >= 5). Jobs put $ENV_PREFIX/lib on
# LD_LIBRARY_PATH so this copy wins over /lib64.
conda install -y -p "$ENV_PREFIX" -c conda-forge --override-channels libstdcxx-ng

# Never build sdists: the OS toolchain (GCC 4.8) is too old (numpy alone
# needs GCC >= 9.3). Prefer an older wheel over a newer source-only release.
export PIP_PREFER_BINARY=1

# glibc 2.17 on ALL nodes (login + gpu, checked 2026-07-21) caps the wheels:
#  - bitsandbytes 0.42.0 is the newest that installs here; it bundles CUDA
#    binaries only up to 12.1, so torch must be the cu121 build (PyPI's
#    default torch wheel bundles CUDA 12.4 -> bnb would fail to load).
#  - torch 2.5.x is the last release with glibc-2.17 wheels.
pip install "torch==2.5.1" --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[dev]"
pip install "bitsandbytes==0.42.0"
# transformers/accelerate must match the bnb 0.42 era: newer transformers
# (>=4.56 hard-requires bnb 0.46.1; 4.55 blocks 4-bit model.to() without
# bnb 0.43.2) can't drive bnb 0.42. This trio was the standard 2024 stack
# for 4-bit NF4 inference with device_map="auto".
pip install "transformers==4.41.2" "accelerate==0.30.1"

# Sanity check: the full unit-test suite (CPU, about 20 s)
pytest

# Pre-download the frozen LLM (~15 GB), the NLI model (~370 MB), and the
# 0.5B smoke-test model (~1 GB; fits a 5 GB MIG slice for pipeline tests)
python - <<'EOF'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-7B-Instruct")
snapshot_download("MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli")
snapshot_download("Qwen/Qwen2.5-0.5B-Instruct")
print("model downloads complete")
EOF

echo "Setup complete. Submit a job with:"
echo "  sbatch hpc/submit_run.sh configs/e2e_baseline_colab20.yaml"

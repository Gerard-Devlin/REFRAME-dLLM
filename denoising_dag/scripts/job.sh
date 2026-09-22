#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
if [[ "$MODE" == download ]]; then
    export HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0
    python -u -m denoising_dag.download --sizes "${MODEL_SIZES:-1.5b,7b}"
    exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
CUDA_VISIBLE_DEVICES="" python -m pytest denoising_dag/tests -q
args=(--mode "$MODE" --gpus "$GPU_IDS" --sizes "${MODEL_SIZES:-1.5b,7b}" --output "$RUN_DIR"
      --limit "${LIMIT:-4}" --batch-size "${BATCH_SIZE:-4}" --depths "${DEPTHS:-1,2,3}"
      --width "${WIDTH:-4}" --repeats "${REPEATS:-3}" --snapshots "${SNAPSHOTS:-2}"
      --max-new-tokens "${MAX_NEW_TOKENS:-128}" --threshold "${THRESHOLD:-0.9}"
      --min-free-gib "${MIN_FREE_GIB:-22}")
if [[ -n "${PROMPTS:-}" ]]; then args+=(--prompts "$PROMPTS"); fi
python -u -m denoising_dag.campaign "${args[@]}"

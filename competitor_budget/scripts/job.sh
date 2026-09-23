#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate GPU $id"; exit 2; }
    seen[$id]=1
    uuids+=("$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
python - <<'PY'
import os, torch
for i in range(torch.cuda.device_count()):
    free, _ = torch.cuda.mem_get_info(i)
    print('logical GPU', i, torch.cuda.get_device_name(i), 'free GiB', free / 2**30, flush=True)
    if free / 2**30 < float(os.environ.get('MIN_FREE_GIB', '8')):
        raise SystemExit('Insufficient free GPU memory')
PY

python -m pytest competitor_budget/tests -q
args=(--dataset "$DATASET" --output "$RUN_DIR/eval" --mode "$MODE"
      --limit "${LIMIT:-32}" --max-new-tokens "${MAX_NEW_TOKENS:-512}"
      --block-size "${BLOCK_SIZE:-32}" --small-block-size "${SMALL_BLOCK_SIZE:-8}"
      --threshold "${THRESHOLD:-0.95}" --margin "${MARGIN:-0.0}")
if [[ "${USE_BLOCK_CACHE:-0}" == 1 ]]; then args+=(--use-block-cache); fi
if [[ ${#physical[@]} -gt 1 ]]; then
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#physical[@]}" \
        -m competitor_budget.evaluate "${args[@]}"
else
    python -u -m competitor_budget.evaluate "${args[@]}"
fi

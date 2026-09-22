#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/home/xuyouwen/hf_home_local
export HF_HUB_CACHE=/home/xuyouwen/hf_hub_local
export HF_DATASETS_CACHE=/home/xuyouwen/hf_home_local/datasets
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate GPU $id"; exit 2; }
    seen[$id]=1
    uuids+=("$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python - <<'PY'
import os, torch
for i in range(torch.cuda.device_count()):
    free,total=torch.cuda.mem_get_info(i)
    print('logical GPU',i,torch.cuda.get_device_name(i),'free GiB',free/2**30,flush=True)
    if free/2**30 < float(os.getenv('MIN_FREE_GIB','24')):
        raise SystemExit('Insufficient free memory; choose idle GPUs')
PY
args=(--data "$DATA_DIR" --output "$RUN_DIR" --global-batch "${GLOBAL_BATCH:-$((2*${#physical[@]}))}"
      --rank "${ADAPTER_RANK:-64}" --lr "${ADAPTER_LR:-1e-4}" --seed "${SEED:-1234}"
      --eval-limit "${EVAL_LIMIT:-32}" --final-eval-limit "${FINAL_EVAL_LIMIT:-256}"
      --heldout-limit "${HELDOUT_LIMIT:-32}")
if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
if [[ "${SMOKE:-0}" == 1 ]]; then args+=(--smoke); fi
python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#physical[@]}" \
    -m relation_adapter.train "${args[@]}"

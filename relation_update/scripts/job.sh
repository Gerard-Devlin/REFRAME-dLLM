#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"

export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

[[ "$MODE" == collect || "$MODE" == train || "$MODE" == oracle ]] || { echo 'MODE must be collect, train or oracle'; exit 2; }
if [[ "$MODE" == train ]]; then
    python - "$TRACE_DIR/manifest.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding='utf-8') as stream:
    manifest = json.load(stream)
if manifest.get('format_version') != 1 or manifest.get('status') != 'complete':
    raise SystemExit('Training requires completed format_version=1 traces')
PY
fi
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS'; exit 2; }
IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate physical GPU: $id"; exit 2; }
    seen[$id]=1
    uuid="$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)"
    [[ "$uuid" == GPU-* && "$uuid" != *$'\n'* ]] || { echo "Cannot resolve physical GPU $id to one UUID"; exit 2; }
    uuids+=("$uuid")
    printf 'physical GPU %s -> %s\n' "$id" "$uuid"
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export EXPECTED_GPU_COUNT="${#physical[@]}"
if [[ "$MODE" != train ]]; then
    export MIN_FREE_GIB="${MIN_FREE_GIB:-8}"
else
    export MIN_FREE_GIB="${MIN_FREE_GIB:-2}"
fi
python - <<'PY'
import os
import torch

expected = int(os.environ['EXPECTED_GPU_COUNT'])
if torch.cuda.device_count() != expected:
    raise SystemExit(f'Expected {expected} visible GPUs, found {torch.cuda.device_count()}')
minimum = float(os.environ['MIN_FREE_GIB'])
for i in range(expected):
    free, total = torch.cuda.mem_get_info(i)
    print(f'logical GPU {i}: {torch.cuda.get_device_name(i)}, '
          f'free={free / 2**30:.2f} GiB, total={total / 2**30:.2f} GiB', flush=True)
    if free / 2**30 < minimum:
        raise SystemExit(f'Insufficient free GPU memory on logical GPU {i}: need {minimum} GiB')
PY

python -m pytest relation_update/tests -q
if [[ "$MODE" == oracle ]]; then
    module=relation_update.oracle
    args=(--data "$DATA_DIR" --output "$TRACE_DIR"
          --prompts "${ORACLE_PROMPTS:-32}" --repeats "${TIMING_REPEATS:-2}"
          --max-new-tokens "${MAX_NEW_TOKENS:-512}"
          --threshold "${THRESHOLD:-0.90}" --seed "${SEED:-1234}")
elif [[ "$MODE" == collect ]]; then
    module=relation_update.collect
    args=(--data "$DATA_DIR" --output "$TRACE_DIR"
          --train-prompts "${TRAIN_PROMPTS:-256}" --heldout-prompts "${HELDOUT_PROMPTS:-64}"
          --max-pairs-per-prompt "${MAX_PAIRS_PER_PROMPT:-32}" --top-k "${TOP_K:-16}"
          --feature-size "${FEATURE_SIZE:-64}" --max-new-tokens "${MAX_NEW_TOKENS:-512}"
          --threshold "${THRESHOLD:-0.90}" --seed "${SEED:-1234}")
else
    module=relation_update.train
    args=(--traces "$TRACE_DIR" --output "$RUN_DIR/train" --epochs "${EPOCHS:-3}"
          --batch-per-gpu "${BATCH_PER_GPU:-32}" --width "${WIDTH:-64}"
          --lr "${LR:-0.001}" --seed "${SEED:-1234}")
    if [[ "${RESUME:-0}" == 1 ]]; then args+=(--resume); fi
fi
if [[ ${#physical[@]} -gt 1 ]]; then
    python -u -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node="${#physical[@]}" -m "$module" "${args[@]}"
else
    python -u -m "$module" "${args[@]}"
fi

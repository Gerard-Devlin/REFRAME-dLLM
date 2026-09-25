#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy

IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=(); uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate GPU: $id"; exit 2; }
    seen[$id]=1
    uuid="$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)"
    [[ "$uuid" == GPU-* && "$uuid" != *$'\n'* ]] || { echo 'GPU resolution failed'; exit 2; }
    uuids+=("$uuid"); printf 'physical GPU %s -> %s\n' "$id" "$uuid"
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
python - <<'PY'
import os, subprocess, torch
selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
if os.environ.get('REQUIRE_IDLE','1')!='0':
    apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
    for line in apps.splitlines():
        fields=[x.strip() for x in line.split(',')]
        if len(fields)==2 and fields[0] in selected:
            raise SystemExit(f'Selected GPU already has compute PID {fields[1]}; timing requires idle cards')
if torch.cuda.device_count()!=len(selected): raise SystemExit('Visible GPU count mismatch')
for i in range(len(selected)):
    free,total=torch.cuda.mem_get_info(i)
    print('logical',i,torch.cuda.get_device_name(i),'free GiB',free/2**30,flush=True)
PY
python -m pytest fastv_dllm/tests -q

case "$MODE" in
  audit) default_limit=1; default_tokens=64 ;;
  probe) default_limit=32; default_tokens=512 ;;
  smoke) default_limit=2; default_tokens=64 ;;
  evaluate) default_limit=256; default_tokens=512 ;;
esac
args=(--stage "$MODE" --dataset "$DATASET" --output "$RUN_DIR/output"
      --limit "${LIMIT:-$default_limit}" --max-new-tokens "${MAX_NEW_TOKENS:-$default_tokens}"
      --block-size "${BLOCK_SIZE:-32}" --small-block-size "${SMALL_BLOCK_SIZE:-8}"
      --threshold "${THRESHOLD:-0.90}" --prune-after-layer "${PRUNE_AFTER_LAYER:-4}"
      --support-keep "${SUPPORT_KEEP:-8}" --observe-layers "${OBSERVE_LAYERS:-2,4,8,12}"
      --observe-keeps "${OBSERVE_KEEPS:-4,8,12,16}")
if [[ ${#physical[@]} -gt 1 ]]; then
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#physical[@]}" \
        -m fastv_dllm.evaluate "${args[@]}"
else
    python -u -m fastv_dllm.evaluate "${args[@]}"
fi

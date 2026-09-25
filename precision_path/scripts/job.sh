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
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
if [[ "$MODE" == calibrate ]]; then
    python -u -m precision_path.calibrate --calibration "$CALIBRATION_REPORT" \
        --evaluation "$EVALUATION_REPORT" --alpha "${ALPHA:-0.01}" --output "$RUN_DIR/calibration.json"
    exit
fi
[[ "$MODE" == cost || "$MODE" == audit ]] || { echo 'Invalid MODE'; exit 2; }
IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate GPU: $id"; exit 2; }
    seen[$id]=1
    uuid="$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)"
    [[ "$uuid" == GPU-* && "$uuid" != *$'\n'* ]] || { echo 'GPU resolution failed'; exit 2; }
    uuids+=("$uuid")
    printf 'physical GPU %s -> %s\n' "$id" "$uuid"
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
python - <<'PY'
import os,subprocess
import torch
selected=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for line in apps.splitlines():
    fields=[x.strip() for x in line.split(',')]
    if len(fields)==2 and fields[0] in selected:
        raise SystemExit(f'Selected GPU already has compute PID {fields[1]}; refuse shared-load timing')
if torch.cuda.device_count()!=len(selected):
    raise SystemExit('Visible GPU count mismatch')
for i in range(len(selected)):
    free,total=torch.cuda.mem_get_info(i)
    print(i,torch.cuda.get_device_name(i),'free GiB',free/2**30,flush=True)
    if free<28*2**30:
        raise SystemExit('Need at least 28 GiB free for clean dual-model measurements')
PY
python -m pytest precision_path/tests -q
default_prompts=4
if [[ ${#physical[@]} -gt $default_prompts ]]; then default_prompts=${#physical[@]}; fi
args=(--stage "$MODE" --role "${ROLE:-development}" --data "$DATA_DIR" --output "$RUN_DIR/output"
      --prompts "${PROMPTS:-$default_prompts}" --max-new-tokens "${MAX_NEW_TOKENS:-512}"
      --threshold "${THRESHOLD:-0.90}" --repeats "${REPEATS:-5}" --max-states "${MAX_STATES:-4}"
      --seed "${SEED:-1234}")
if [[ "$MODE" == audit ]]; then args+=(--cost-report "$COST_REPORT"); fi
if [[ ${#physical[@]} -gt 1 ]]; then
    python -u -m torch.distributed.run --standalone --nnodes=1 \
        --nproc_per_node="${#physical[@]}" -m precision_path.probe "${args[@]}"
else
    python -u -m precision_path.probe "${args[@]}"
fi

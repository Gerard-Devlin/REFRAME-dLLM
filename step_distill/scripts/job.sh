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
IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate GPU: $id"; exit 2; }
    seen[$id]=1
    uuid="$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)"
    [[ "$uuid" == GPU-* && "$uuid" != *$'\n'* ]] || exit 2
    uuids+=("$uuid")
    printf 'physical GPU %s -> %s\n' "$id" "$uuid"
done
[[ "$MODE" != export || ${#physical[@]} == 1 ]] || { echo 'Export requires one GPU'; exit 2; }
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
python - <<'PY'
import os, subprocess, torch
chosen=os.environ['CUDA_VISIBLE_DEVICES'].split(',')
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
for line in apps.splitlines():
    fields=[v.strip() for v in line.split(',')]
    if len(fields)==2 and fields[0] in chosen:
        raise SystemExit(f'Selected GPU occupied by PID {fields[1]}; refusing shared-load run')
if torch.cuda.device_count()!=len(chosen):
    raise SystemExit('Visible GPU count differs from requested physical GPU list')
for i in range(torch.cuda.device_count()):
    free,_=torch.cuda.mem_get_info(i)
    print('logical GPU',i,torch.cuda.get_device_name(i),'free GiB',free/2**30,flush=True)
    if free<28*2**30: raise SystemExit('Need 28 GiB free; no automatic capacity reduction')
PY
python -m pytest step_distill/tests -q
args=("$MODE" --data "$DATA_DIR" --dataset "$DATASET" --output "$RUN_DIR/output" --branch "${BRANCH:-basic}")
for pair in 'AUDIT_REPORT:audit' 'TRAJECTORIES:trajectories' 'SMOKE_REPORT:smoke' 'RESUME:resume' 'CHECKPOINT:checkpoint' 'BASIC_MODEL:basic-model' 'RELEASE_MODEL:release-model' 'DEV_REPORT:dev-report' 'SPLIT:split' 'TIME_LIMIT:time-limit'; do
    name="${pair%%:*}"; flag="${pair#*:}"
    if [[ -v "$name" && -n "${!name}" ]]; then args+=("--$flag" "${!name}"); fi
done
python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#physical[@]}" \
    -m step_distill "${args[@]}"

#!/usr/bin/env bash
set -euo pipefail
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
cd "$REPO"
if [[ "$PHASE" != prepare && ! -s "$DATA_DIR/manifest.json" ]]; then
    printf 'Data preparation is not complete: %s/manifest.json is missing or empty. Check the prepare job before retrying.\n' "$DATA_DIR" >&2
    exit 2
fi
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/home/xuyouwen/hf_home_local
export HF_HUB_CACHE=/home/xuyouwen/hf_hub_local
export HF_DATASETS_CACHE=/home/xuyouwen/hf_home_local/datasets
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
STEPS="${STEPS:-200}"
MAX_SECONDS="${MAX_SECONDS:-600}"
GLOBAL_BATCH="${GLOBAL_BATCH:-48}"
MICRO_BATCH="${MICRO_BATCH:-8}"
WIDTH="${WIDTH:-384}"
LAYERS="${LAYERS:-6}"
HEADS="${HEADS:-6}"
SEED="${SEED:-1234}"
if [[ "$PHASE" == prepare ]]; then
    unset HF_HUB_OFFLINE HF_DATASETS_OFFLINE TRANSFORMERS_OFFLINE HF_EVALUATE_OFFLINE
    python -m relation_diffusion.prepare --output "$DATA_DIR"
    exit 0
fi
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
# Bind requested nvidia-smi physical IDs by UUID, avoiding CUDA enumeration drift.
IFS=',' read -r -a IDS <<< "$GPU_IDS"
declare -A SEEN=()
UUIDS=()
for ID in "${IDS[@]}"; do
    if [[ -v "SEEN[$ID]" ]]; then echo "Duplicate GPU: $ID" >&2; exit 2; fi
    SEEN[$ID]=1
    UUIDS+=("$(nvidia-smi -i "$ID" --query-gpu=uuid --format=csv,noheader)")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${UUIDS[*]}")"
NPROC="${#UUIDS[@]}"
python - <<'PY'
import os, torch
minimum = float(os.environ.get('MIN_FREE_GIB', '24'))
for i in range(torch.cuda.device_count()):
    free,total = torch.cuda.mem_get_info(i)
    print('logical GPU',i,torch.cuda.get_device_name(i),'free GiB',free/2**30,flush=True)
    if free < minimum*2**30:
        raise SystemExit('Requested GPU has insufficient free memory; refusing to launch. Check other users first.')
PY
SHAPE=(--width "$WIDTH" --layers "$LAYERS" --heads "$HEADS")
if [[ "$PHASE" == preflight ]]; then
    if [[ "$NPROC" -ne 1 ]]; then echo 'Preflight measures single-GPU inference. Specify one GPU.' >&2; exit 2; fi
    python -m relation_diffusion.preflight --data "$DATA_DIR" --output "$RUN_DIR/preflight.json" "${SHAPE[@]}"
    exit 0
fi
if (( GLOBAL_BATCH % (NPROC * MICRO_BATCH) != 0 )); then
    echo 'GLOBAL_BATCH must be divisible by GPU count * MICRO_BATCH' >&2; exit 2
fi
train_one() {
    local CODE="$1" OBJ="$2" DEST="$RUN_DIR/$1-$2-seed$SEED"
    python -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
        -m relation_diffusion.train --data "$DATA_DIR" --output "$DEST" \
        --codec "$CODE" --objective "$OBJ" --steps "$STEPS" --max-seconds "$MAX_SECONDS" \
        --global-batch "$GLOBAL_BATCH" --micro-batch "$MICRO_BATCH" --seed "$SEED" "${SHAPE[@]}"
    python - "$DEST/status.json" <<'PY'
import json,sys
s=json.load(open(sys.argv[1]))
if s['status']!='complete':
    raise SystemExit('Training hit the time budget. Checkpoint saved. Campaign stopped; do not compare unequal update counts.')
PY
    # Evaluate one model on one GPU; inference timings must never include DDP.
    CUDA_VISIBLE_DEVICES="${UUIDS[0]}" python -m relation_diffusion.evaluate \
        --checkpoint "$DEST/checkpoint.pt" --data "$DATA_DIR" \
        --output "$DEST/evaluation.json" --limit "${EVAL_LIMIT:-256}"
}
if [[ "$PHASE" == train ]]; then
    train_one "${CODEC:-identity}" "${OBJECTIVE:-diffusion}"
else
    # Two-model bounded pilot only. Additional controls/seeds are separate decisions.
    train_one identity diffusion
    train_one relation2 diffusion
    python -m relation_diffusion.compare --root "$RUN_DIR" --output "$RUN_DIR/comparison.json"
fi

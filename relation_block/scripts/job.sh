#!/usr/bin/env bash
set -euo pipefail
cd "$REPO"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME=/home/xuyouwen/hf_home_local
export HF_HUB_CACHE=/home/xuyouwen/hf_hub_local
export HF_DATASETS_CACHE=/home/xuyouwen/hf_home_local/datasets
export HF_ENDPOINT=https://hf-mirror.com
export HF_HUB_DISABLE_XET=1 TOKENIZERS_PARALLELISM=false
# Some existing conda activation hooks append to LD_LIBRARY_PATH without
# guarding an initially unset variable. Define it before `set -u` reaches the
# hook; this preserves any caller-provided value and avoids changing the env.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
env_name="${CONDA_ENV:-fastdllm311}"
if [[ "$PHASE" == setup ]]; then
    # Reuse the user's existing environment. Never clone/install torch or CUDA.
    conda activate "$env_name"
    python -m pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
        'transformers==4.57.3' 'huggingface_hub>=0.34,<1' 'safetensors>=0.4.5' 'einops>=0.8' \
        'pytest>=8,<10' 'tensorboard>=2.18,<3'
    python -c "import torch, transformers, datasets; print(torch.__version__, transformers.__version__, datasets.__version__)"
    exit 0
fi
conda activate "$env_name"
if [[ "$PHASE" == weights || "$PHASE" == subset ]]; then
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_EVALUATE_OFFLINE
    if [[ "$PHASE" == weights ]]; then
        python -u -c "from relation_block.common import snapshot; print(snapshot(offline=False))"
    else
        python -u -m relation_block.download_subset \
            --output "${SUBSET_DIR:-$HF_HOME/nemotron/math_code_100m_bpe2048_v1}" \
            --tokens "${SUBSET_TOKENS:-100000000}" --length "${SEQ_LENGTH:-2048}" \
            --seed "${SEED:-1234}" --reasoning "${REASONING:-any}"
    fi
    exit 0
fi
if [[ "$PHASE" == prepare ]]; then
    unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE HF_EVALUATE_OFFLINE
    if [[ "${TRAIN_SOURCE:-nemotron}" == nemotron ]]; then
        python -u -m relation_block.prepare --data "$DATA_DIR" \
            --source-dir "${SUBSET_DIR:-$HF_HOME/nemotron/math_code_100m_bpe2048_v1}" --length "${SEQ_LENGTH:-2048}"
    elif [[ "$TRAIN_SOURCE" == alpaca ]]; then
        python -u -m relation_block.prepare --data "$DATA_DIR" --download --length "${SEQ_LENGTH:-512}"
    else
        echo 'TRAIN_SOURCE must be nemotron or alpaca'; exit 2
    fi
    exit 0
fi
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_EVALUATE_OFFLINE=1
if [[ "$PHASE" == compare ]]; then
    IFS=',' read -ra runs <<< "${COMPARE_RUNS:?Set comma-separated evaluation directories}"
    python -m relation_block.compare "${runs[@]}"
    exit 0
fi
IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
uuids=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate GPU $id"; exit 2; }
    seen[$id]=1
    uuid="$(nvidia-smi -i "$id" --query-gpu=uuid --format=csv,noheader)"
    uuids+=("$uuid")
done
export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${uuids[*]}")"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
python - <<'PY'
import os, torch
for i in range(torch.cuda.device_count()):
    free, total = torch.cuda.mem_get_info(i)
    print('logical GPU', i, torch.cuda.get_device_name(i), 'free GiB', free/2**30, flush=True)
    if free/2**30 < float(os.getenv('MIN_FREE_GIB', '24')):
        raise SystemExit('Insufficient free memory; choose idle GPUs')
PY
train_model() {
    if [[ ${#physical[@]} -gt 1 ]]; then
        python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#physical[@]}" \
            -m relation_block.train "$@"
    else
        python -u -m relation_block.train "$@"
    fi
}
evaluate_model() {
    # Explicitly use the first SELECTED physical GPU for every timing comparison.
    CUDA_VISIBLE_DEVICES="${uuids[0]}" python -u -m relation_block.evaluate "$@"
}
case "$PHASE" in
resident)
    args=(--data "$DATA_DIR" --output "$RUN_DIR/train" --arm "${ARM:-token}"
          --global-batch "${GLOBAL_BATCH:-$((2 * ${#physical[@]}))}" --micro-batch 1
          --lr "${LR:-4e-6}" --seed "${SEED:-1234}" --eval-limit "${EVAL_LIMIT:-256}"
          --rounds "${ROUNDS:-8,16}" --max-new-tokens 512
          --reconstruction-limit "${RECONSTRUCTION_LIMIT:-32}")
    if [[ -n "${RESUME:-}" ]]; then
        args+=(--resume "$RESUME")
    else
        python -u -m relation_block.full_smoke --data "$DATA_DIR" --output "$RUN_DIR/full_smoke" \
            --world-size "${#physical[@]}" --global-batch "${GLOBAL_BATCH:-$((2 * ${#physical[@]}))}" --micro-batch 1
    fi
    python -u -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${#physical[@]}" \
        -m relation_block.resident_train "${args[@]}" ;;
continuation)
    python -u -m relation_block.continuation campaign --data "$DATA_DIR" --output "$RUN_DIR/campaign" \
        --world-size "${#physical[@]}" --global-batch "${GLOBAL_BATCH:-12}" --seed "${SEED:-1234}" \
        --limit "${EVAL_LIMIT:-256}" --rounds "${ROUNDS:-8,16}" \
        --reconstruction-limit "${RECONSTRUCTION_LIMIT:-32}" ;;
diagnose-length)
    python -u -m relation_block.diagnose length --data "$DATA_DIR" --output "$RUN_DIR/diagnostic" \
        --source "${SOURCE_RUN:?Set SOURCE_RUN to the completed full training run}" \
        --limit "${EVAL_LIMIT:-256}" --reconstruction-limit "${RECONSTRUCTION_LIMIT:-32}" ;;
diagnose-zero)
    python -u -m relation_block.diagnose zero --data "$DATA_DIR" --output "$RUN_DIR/diagnostic" \
        --world-size "${#physical[@]}" --global-batch "${GLOBAL_BATCH:-12}" ;;
smoke)
    python -u -m relation_block.full_smoke --data "$DATA_DIR" --output "$RUN_DIR" \
        --world-size "${#physical[@]}" --global-batch "${GLOBAL_BATCH:-12}" --micro-batch "${MICRO_BATCH:-1}" ;;
pilot|full)
    if [[ "$PHASE" == full ]]; then
        # A full campaign explicitly authorizes both gates followed by the epoch.
        # Standalone smoke remains bounded and never starts formal training.
        CUDA_VISIBLE_DEVICES="${uuids[0]}" python -m pytest relation_block/tests -q
        CUDA_VISIBLE_DEVICES="${uuids[0]}" python -u -m relation_block.preflight --data "$DATA_DIR"
        python -u -m relation_block.full_smoke --data "$DATA_DIR" --output "$RUN_DIR/full_smoke" \
            --world-size "${#physical[@]}" --global-batch "${GLOBAL_BATCH:-12}" --micro-batch "${MICRO_BATCH:-1}"
    fi
    # Full epoch by default. STEPS is an explicit bounded override, never an implicit 200-step cap.
    for arm in token relation; do
        train_model --data "$DATA_DIR" --output "$RUN_DIR/$arm/train" --arm "$arm" \
            --steps "${STEPS:-0}" --global-batch "${GLOBAL_BATCH:-12}" --micro-batch "${MICRO_BATCH:-1}" \
            --seed "${SEED:-1234}" --max-seconds "${MAX_SECONDS:-0}" --lr "${LR:-2e-5}" \
            --save-every "${SAVE_EVERY:-500}" --eval-every "${EVAL_EVERY:-1000}" \
            --eval-limit "${EVAL_LIMIT:-32}" --final-eval-limit "${FINAL_EVAL_LIMIT:-256}" \
            --rounds "${ROUNDS:-2,4,8,16}" --max-new-tokens "${MAX_NEW_TOKENS:-512}"
    done
    python -m relation_block.compare "$RUN_DIR/token/eval/final" "$RUN_DIR/relation/eval/final" ;;
preflight)
    python -m pytest relation_block/tests -q
    python -u -m relation_block.preflight --data "$DATA_DIR" ;;
train)
    args=(--data "$DATA_DIR" --output "$RUN_DIR/train" --arm "$ARM" --steps "${STEPS:-0}"
          --global-batch "${GLOBAL_BATCH:-12}" --micro-batch "${MICRO_BATCH:-1}"
          --seed "${SEED:-1234}" --max-seconds "${MAX_SECONDS:-0}" --lr "${LR:-2e-5}"
          --save-every "${SAVE_EVERY:-500}" --eval-every "${EVAL_EVERY:-1000}"
          --eval-limit "${EVAL_LIMIT:-32}" --final-eval-limit "${FINAL_EVAL_LIMIT:-256}"
          --rounds "${ROUNDS:-2,4,8,16}" --max-new-tokens "${MAX_NEW_TOKENS:-512}")
    if [[ -n "${RESUME:-}" ]]; then args+=(--resume "$RESUME"); fi
    train_model "${args[@]}" ;;
native|baseline|evaluate)
    args=(--data "$DATA_DIR" --output "$RUN_DIR/eval" --split "${EVAL_SPLIT:-dev}" --limit "${EVAL_LIMIT:-32}"
          --rounds "${ROUNDS:-2,4,8,16}" --max-new-tokens "${MAX_NEW_TOKENS:-256}")
    if [[ "$PHASE" == native ]]; then args+=(--official); fi
    if [[ "$PHASE" == evaluate ]]; then args+=(--checkpoint "$CHECKPOINT"); fi
    evaluate_model "${args[@]}" ;;
esac

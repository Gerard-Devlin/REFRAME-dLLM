#!/usr/bin/env bash
set -euo pipefail
mode="${1:?mode required}"
run_id="${2:?run id required}"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
mkdir -p reframe/logs reframe/results
exec > >(tee "reframe/logs/${mode}_${run_id}.log") 2>&1
trap 'code=$?; printf "Job finished: exit=%s; log=reframe/logs/%s_%s.log\n" "$code" "$mode" "$run_id"' EXIT

source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HOME="${HF_HOME:-/home/xuyouwen/hf_home_local}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/home/xuyouwen/hf_home_local/datasets}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export TOKENIZERS_PARALLELISM=false HF_ALLOW_CODE_EVAL=1
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
model="${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"

if [[ "$mode" == "gsm8k" ]]; then
    # Set LIMIT=1319 and GEN_LENGTH=256 only after the smoke succeeds.
    python reframe/eval_reframe.py --model reframe_llada \
        --model_args "model_path=${model},gen_length=${GEN_LENGTH:-64},block_length=32,threshold=0.9,show_speed=True,reframe_kind=${REFRAME_KIND:-pair},reframe_pilots=${PILOTS:-16},reframe_refresh_blocks=${REFRESH_BLOCKS:-2},reframe_log=reframe/results/metrics_${run_id}" \
        --tasks gsm8k --num_fewshot 5 --batch_size 1 --limit "${LIMIT:-2}" \
        --confirm_run_unsafe_code --log_samples \
        --output_path "reframe/results/gsm8k_${run_id}"
else
    extra=()
    if [[ "$mode" == "oracle" ]]; then extra+=(--oracle); fi
    python reframe/run.py --model-path "$model" --device cuda --dtype bfloat16 --backend flash \
        --gen-length "${GEN_LENGTH:-128}" --block-length 32 --pilots "${PILOTS:-16}" \
        --refresh-blocks "${REFRESH_BLOCKS:-2}" --warmup 1 --repeats 3 \
        --output "reframe/results/${mode}_${run_id}.jsonl" "${extra[@]}"
    python reframe/summarize.py "reframe/results/${mode}_${run_id}.jsonl"
fi

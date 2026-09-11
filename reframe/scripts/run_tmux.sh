#!/usr/bin/env bash
set -euo pipefail

mode="${1:-compare}"
case "$mode" in compare|oracle|gsm8k) ;; *) echo "Usage: bash reframe/scripts/run_tmux.sh [compare|oracle|gsm8k]"; exit 2;; esac
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
run_id="$(date +%Y%m%d_%H%M%S)_$$"
session="reframe_${mode}_${run_id}"
# An existing tmux server keeps its own environment. Pass job settings
# explicitly so a later GPU/length override does not silently use old values.
settings=(
    "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}"
    "CONDA_ROOT=${CONDA_ROOT:-/opt/miniconda3}"
    "CONDA_ENV=${CONDA_ENV:-fastdllm311}"
    "HF_HOME=${HF_HOME:-/home/xuyouwen/hf_home_local}"
    "HF_HUB_CACHE=${HF_HUB_CACHE:-/home/xuyouwen/hf_hub_local}"
    "HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-/home/xuyouwen/hf_home_local/datasets}"
    "MODEL_PATH=${MODEL_PATH:-GSAI-ML/LLaDA-8B-Instruct}"
    "GEN_LENGTH=${GEN_LENGTH:-}"
    "LIMIT=${LIMIT:-2}"
    "PILOTS=${PILOTS:-16}"
    "REFRESH_BLOCKS=${REFRESH_BLOCKS:-2}"
    "REFRAME_KIND=${REFRAME_KIND:-pair}"
)
env_args=()
for setting in "${settings[@]}"; do env_args+=(-e "$setting"); done
tmux new-session -d -s "$session" "${env_args[@]}" bash "$script_dir/job.sh" "$mode" "$run_id"
printf 'Started %s\nAttach: tmux attach -t %s\n' "$session" "$session"

#!/usr/bin/env bash
set -eo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CONDA_ROOT="${CONDA_ROOT:-$(conda info --base)}"
CONDA_ENV="${CONDA_ENV:-fastdllm311}"
RUN_ROOT="${RUN_ROOT:?Set a new RUN_ROOT, or an existing main-matrix run to resume}"
GPU_IDS="${GPU_IDS:?Specify eligible physical GPU IDs}"
IFS=',' read -ra requested_gpus <<< "$GPU_IDS"
default_cap=${#requested_gpus[@]}
(( default_cap <= 6 )) || default_cap=6
MAX_GPUS="${MAX_GPUS:-$default_cap}"
for name in GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET; do
  [[ -v "$name" ]] || { echo "$name is required"; exit 2; }
  printf -v "$name" '%s' "$(realpath -e "${!name}")"
done
mkdir -p "$RUN_ROOT"
RUN_ROOT="$(realpath -e "$RUN_ROOT")"
mkdir -p "$RUN_ROOT/launcher"
session="focus-main-$(date +%Y%m%d-%H%M%S)-$$"
env_file="$RUN_ROOT/launcher/$session.env"
for name in REPO CONDA_ROOT CONDA_ENV RUN_ROOT GPU_IDS MAX_GPUS HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET; do
  if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >>"$env_file"; fi
done
entry="$RUN_ROOT/launcher/$session.sh"
cat >"$entry" <<'ENTRY'
#!/usr/bin/env bash
set -eo pipefail
source "$1"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$CONDA_ENV"
set -u
cd "$REPO"
export PYTHONPATH="$REPO/focus_dllm/dllm-eval:$REPO${PYTHONPATH:+:$PYTHONPATH}"
unset HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_XET=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=1
if [[ ! -f "$RUN_ROOT/elastic/manifest.json" ]]; then
  python -u -m dllm_eval.worker initialize --matrix main --run-root "$RUN_ROOT" \
    --gsm8k-dataset "$GSM8K_DATASET" --math-dataset "$MATH_DATASET" \
    --humaneval-dataset "$HUMANEVAL_DATASET" --mbpp-dataset "$MBPP_DATASET"
fi
python -c 'import json,os; from pathlib import Path; m=json.loads((Path(os.environ["RUN_ROOT"])/"elastic/manifest.json").read_text()); assert m.get("matrix")=="main", "Not a main-matrix run"'
python -u -m dllm_eval.scheduler --repo "$REPO" --run-root "$RUN_ROOT" \
  --candidates "$GPU_IDS" --max-total-gpus "$MAX_GPUS" --progress-seconds 30
ENTRY
wrapper="$RUN_ROOT/launcher/$session.wrapper.sh"
printf '#!/usr/bin/env bash\nbash %q %q >> %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nexit "$rc"\n' \
  "$entry" "$env_file" "$RUN_ROOT/job.log" "$RUN_ROOT/exit_code" >"$wrapper"
tmux new-session -d -s "$session" "bash $(printf '%q' "$wrapper")"
printf '%s\n' "$session" >"$RUN_ROOT/tmux_session"
printf 'Session: %s\nLog: %s/job.log\nProgress: %s/elastic/progress.log\n' "$session" "$RUN_ROOT" "$RUN_ROOT"

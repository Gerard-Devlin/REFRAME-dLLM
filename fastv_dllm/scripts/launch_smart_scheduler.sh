#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_ROOT="$(realpath -e "${RUN_ROOT:?Existing RUN_ROOT is required}")"
CURRENT_RUN="$(realpath -e "${CURRENT_RUN:?CURRENT_RUN is required}")"
MAIN_TMUX="${MAIN_TMUX:?MAIN_TMUX is required}"
for name in GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET; do
  [[ -v "$name" ]] || { echo "$name is required"; exit 2; }
  printf -v "$name" '%s' "$(realpath -e "${!name}")"
done
STATE_DIR="$RUN_ROOT/smart_scheduler"
[[ ! -e "$STATE_DIR" ]] || { echo 'Smart scheduler already exists'; exit 2; }
mkdir -p "$STATE_DIR"
for name in REPO RUN_ROOT STATE_DIR CURRENT_RUN MAIN_TMUX GSM8K_DATASET MATH_DATASET HUMANEVAL_DATASET MBPP_DATASET CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
  if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >>"$STATE_DIR/scheduler.env"; fi
done
cat >"$STATE_DIR/entry.sh" <<'ENTRY'
#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/scheduler.env"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "${CONDA_ROOT:-/opt/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-fastdllm311}"
cd "$REPO"
exec 9>"$REPO/fastv_dllm/runs/paper_smart_scheduler.lock"
flock -n 9 || { echo 'Another smart scheduler owns the lock'; exit 73; }
python -m pytest fastv_dllm/tests -q
exec python -u -m fastv_dllm.smart_paper_scheduler \
  --repo "$REPO" --run-root "$RUN_ROOT" --state-dir "$STATE_DIR/runtime" \
  --current-run "$CURRENT_RUN" --main-tmux "$MAIN_TMUX" --current-gpus 1,7 \
  --candidates 0,1,2,3,4,5,6,7 --max-total-gpus 6 \
  --gsm8k-dataset "$GSM8K_DATASET" --math-dataset "$MATH_DATASET" \
  --humaneval-dataset "$HUMANEVAL_DATASET" --mbpp-dataset "$MBPP_DATASET"
ENTRY
cat >"$STATE_DIR/wrapper.sh" <<EOF
#!/usr/bin/env bash
bash $(printf '%q' "$STATE_DIR/entry.sh") >$(printf '%q' "$STATE_DIR/scheduler.log") 2>&1
rc=\$?
printf '%s\n' "\$rc" >$(printf '%q' "$STATE_DIR/exit_code")
exit "\$rc"
EOF
session="fv-smart-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$STATE_DIR/wrapper.sh")"
printf '%s\n' "$session" >"$STATE_DIR/tmux_session"
printf 'Session: %s\nState: %s\nLog: %s/scheduler.log\n' "$session" "$STATE_DIR" "$STATE_DIR"

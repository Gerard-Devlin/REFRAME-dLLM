#!/usr/bin/env bash
set -euo pipefail
MODE="${1:?Usage: launch_tmux.sh audit|probe|smoke|evaluate}"
case "$MODE" in audit|probe|smoke|evaluate) ;; *) echo 'Invalid mode'; exit 2;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_IDS="${GPU_IDS:?Explicit physical GPU_IDS required, e.g. 3 or 0,2,3}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS'; exit 2; }
DATASET="$(realpath -e "${DATASET:?Prepared GSM8K JSON required}")"
RUN_DIR="$(realpath -m "${RUN_DIR:-$REPO/fastv_dllm/runs/${MODE}_$(date +%Y%m%d_%H%M%S)_$$}")"
[[ ! -e "$RUN_DIR" ]] || { echo 'Use a new RUN_DIR'; exit 2; }
mkdir -p "$RUN_DIR"
for name in REPO MODE RUN_DIR GPU_IDS DATASET TASK LIMIT GEN_LENGTH BLOCK_LENGTH THRESHOLD DECODING_MODE PRUNE_AFTER_LAYER SUPPORT_KEEP_RATIO CACHE_MODE METHODS REQUIRE_IDLE CONDA_ROOT CONDA_ENV HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; fi
done
printf '#!/usr/bin/env bash\nsource %q\ndate -Is > %q\nbash %q > %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\ndate -Is > %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$RUN_DIR/started_at" "$REPO/fastv_dllm/scripts/job.sh" \
    "$RUN_DIR/job.log" "$RUN_DIR/exit_code" "$RUN_DIR/finished_at" > "$RUN_DIR/entry.sh"
session="fv-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf '%s\n' "$session" > "$RUN_DIR/tmux_session"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"

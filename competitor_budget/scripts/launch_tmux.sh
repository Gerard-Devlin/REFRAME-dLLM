#!/usr/bin/env bash
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_IDS="${GPU_IDS:?Set comma-separated physical GPU IDs, for example 0,5,6,7}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS'; exit 2; }
MODE="${MODE:-observe}"
[[ "$MODE" == observe || "$MODE" == compare || "$MODE" == sweep ]] || { echo 'MODE must be observe, compare or sweep'; exit 2; }
DATASET="${DATASET:?Set prepared GSM8K JSON path}"
[[ -s "$DATASET" ]] || { echo "Missing dataset: $DATASET"; exit 2; }
RUN_DIR="${RUN_DIR:-$REPO/competitor_budget/runs/${MODE}_$(date +%Y%m%d_%H%M%S)}"
[[ ! -e "$RUN_DIR" ]] || { echo "RUN_DIR exists: $RUN_DIR"; exit 2; }
mkdir -p "$RUN_DIR"
for name in REPO GPU_IDS MODE DATASET RUN_DIR; do
    printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"
done
for name in PRESET LIMIT MAX_NEW_TOKENS BLOCK_SIZE SMALL_BLOCK_SIZE THRESHOLD MARGIN USE_BLOCK_CACHE CONDA_ENV CONDA_ROOT MIN_FREE_GIB HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE; do
    if [[ -v "$name" ]]; then
        printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"
    fi
done
printf '#!/usr/bin/env bash\nsource %q\nbash %q > %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$REPO/competitor_budget/scripts/job.sh" "$RUN_DIR/job.log" "$RUN_DIR/exit_code" > "$RUN_DIR/entry.sh"
session="cb-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"

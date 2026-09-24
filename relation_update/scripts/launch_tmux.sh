#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
[[ "$MODE" == collect || "$MODE" == train || "$MODE" == oracle ]] || { echo 'Usage: bash relation_update/scripts/launch_tmux.sh collect|train|oracle'; exit 2; }
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GPU_IDS="${GPU_IDS:?Set comma-separated physical GPU IDs explicitly}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Invalid GPU_IDS; expected physical IDs such as 3 or 0,1,2,3'; exit 2; }
IFS=',' read -ra physical <<< "$GPU_IDS"
declare -A seen=()
for id in "${physical[@]}"; do
    [[ ! -v "seen[$id]" ]] || { echo "Duplicate physical GPU: $id"; exit 2; }
    seen[$id]=1
done

DATA_DIR="${DATA_DIR:-/home/xuyouwen/hf_home_local/relation_block/nemotron_bpe2048_v1}"
TRACE_DIR="${TRACE_DIR:?Set TRACE_DIR to the trace collection directory}"
if [[ "$MODE" == collect || "$MODE" == oracle ]]; then
    [[ -s "$DATA_DIR/manifest.json" ]] || { echo "Missing prepared-data manifest: $DATA_DIR/manifest.json"; exit 2; }
    [[ ! -e "$TRACE_DIR" ]] || { echo "TRACE_DIR already exists; choose a new collection directory: $TRACE_DIR"; exit 2; }
else
    [[ -s "$TRACE_DIR/manifest.json" ]] || { echo "Trace collection has not finished: $TRACE_DIR/manifest.json is missing"; exit 2; }
fi

# Persist absolute paths so the detached shell does not depend on its cwd.
DATA_DIR="$(realpath -m "$DATA_DIR")"
TRACE_DIR="$(realpath -m "$TRACE_DIR")"
RUN_DIR="${RUN_DIR:-$REPO/relation_update/runs/${MODE}_$(date +%Y%m%d_%H%M%S)_$$}"
RUN_DIR="$(realpath -m "$RUN_DIR")"
if [[ "$MODE" != train && ( "$RUN_DIR" == "$TRACE_DIR" || "$RUN_DIR" == "$TRACE_DIR/"* ) ]]; then
    echo 'RUN_DIR cannot equal or be inside the new TRACE_DIR'; exit 2
fi
[[ "${RESUME:-0}" == 0 || "${RESUME:-0}" == 1 ]] || { echo 'RESUME must be 0 or 1'; exit 2; }
if [[ "${RESUME:-0}" == 1 ]]; then
    [[ "$MODE" == train && -d "$RUN_DIR/train" ]] || { echo 'RESUME=1 requires an existing training RUN_DIR'; exit 2; }
    if [[ -s "$RUN_DIR/tmux_session" ]] && tmux has-session -t "=$(cat "$RUN_DIR/tmux_session")" 2>/dev/null; then
        echo "The existing job is still active: $RUN_DIR"; exit 2
    fi
else
    [[ ! -e "$RUN_DIR" ]] || { echo "RUN_DIR already exists; choose a new log directory: $RUN_DIR"; exit 2; }
fi
mkdir -p "$RUN_DIR"
exec 9> "$RUN_DIR/launch.lock"
flock -n 9 || { echo "Another launcher is using $RUN_DIR"; exit 2; }
if [[ "${RESUME:-0}" == 1 ]]; then
    # Repeat after acquiring the lock to reject concurrent resume launches.
    if [[ -s "$RUN_DIR/tmux_session" ]] && tmux has-session -t "=$(cat "$RUN_DIR/tmux_session")" 2>/dev/null; then
        echo "The existing job is still active: $RUN_DIR"; exit 2
    fi
    previous="$(date +%Y%m%d_%H%M%S)_$$"
    [[ ! -f "$RUN_DIR/job.env" ]] || cp "$RUN_DIR/job.env" "$RUN_DIR/job.env.$previous"
    [[ ! -f "$RUN_DIR/exit_code" ]] || mv "$RUN_DIR/exit_code" "$RUN_DIR/exit_code.$previous"
fi
: > "$RUN_DIR/job.env"
for name in REPO GPU_IDS MODE DATA_DIR TRACE_DIR RUN_DIR; do
    printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"
done
for name in ORACLE_PROMPTS TIMING_REPEATS TRAIN_PROMPTS HELDOUT_PROMPTS MAX_PAIRS_PER_PROMPT TOP_K FEATURE_SIZE MAX_NEW_TOKENS THRESHOLD SEED EPOCHS BATCH_PER_GPU WIDTH LR RESUME CONDA_ENV CONDA_ROOT MIN_FREE_GIB HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_ENDPOINT; do
    if [[ -v "$name" ]]; then
        printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"
    fi
done
printf '#!/usr/bin/env bash\nsource %q\ndate -Is >> %q\nbash %q >> %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\ndate -Is >> %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$RUN_DIR/started_at" "$REPO/relation_update/scripts/job.sh" \
    "$RUN_DIR/job.log" "$RUN_DIR/exit_code" "$RUN_DIR/finished_at" > "$RUN_DIR/entry.sh"
session="ru-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")" 9>&-
printf '%s\n' "$session" > "$RUN_DIR/tmux_session"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"

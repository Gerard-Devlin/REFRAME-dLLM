#!/usr/bin/env bash
set -euo pipefail
PHASE="${1:-}"
case "$PHASE" in setup|prepare|preflight|native|baseline|train|evaluate|compare|smoke|pilot) ;; *) echo 'phase: setup prepare preflight native baseline train evaluate compare smoke pilot'; exit 2;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$REPO/relation_block/runs/${PHASE}_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-/home/xuyouwen/hf_home_local/relation_block/alpaca_bpe512_v1}"
GPU_IDS="${GPU_IDS:-}"
if [[ "$PHASE" != setup && "$PHASE" != prepare && "$PHASE" != compare ]]; then
    [[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]] || { echo 'Set GPU_IDS explicitly, e.g. 3 or 0,1,2,3,4,5'; exit 2; }
    [[ -s "$DATA_DIR/manifest.json" ]] || { echo 'Prepare has not finished: missing manifest.json'; exit 2; }
    if [[ "$PHASE" != train && "$PHASE" != smoke && "$PHASE" != pilot && "$GPU_IDS" == *,* ]]; then
        echo 'Use one GPU for preflight or standalone evaluation; train/smoke/pilot accept multiple GPUs'; exit 2
    fi
fi
if [[ "$PHASE" == train ]]; then
    [[ "${ARM:-}" == token || "${ARM:-}" == relation ]] || { echo 'Set ARM=token or ARM=relation'; exit 2; }
fi
if [[ "$PHASE" == evaluate && ! -s "${CHECKPOINT:-}" ]]; then echo 'Set CHECKPOINT to checkpoint.pt'; exit 2; fi
[[ ! -e "$RUN_DIR" ]] || { echo "RUN_DIR exists: $RUN_DIR"; exit 2; }
mkdir -p "$RUN_DIR"
for name in REPO RUN_DIR DATA_DIR GPU_IDS PHASE; do printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; done
for name in ARM STEPS GLOBAL_BATCH MICRO_BATCH SEED MAX_SECONDS CHECKPOINT RESUME EVAL_LIMIT ROUNDS MAX_NEW_TOKENS EVAL_SPLIT COMPARE_RUNS CONDA_ROOT CONDA_ENV MIN_FREE_GIB; do
    if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; fi
done
printf '#!/usr/bin/env bash\nsource %q\nbash %q > %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$REPO/relation_block/scripts/job.sh" "$RUN_DIR/job.log" "$RUN_DIR/exit_code" > "$RUN_DIR/entry.sh"
session="rb-${PHASE}-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"

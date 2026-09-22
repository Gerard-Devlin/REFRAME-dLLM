#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUN_DIR="${RUN_DIR:-$REPO/relation_adapter/runs/frozen_$(date +%Y%m%d_%H%M%S)}"
DATA_DIR="${DATA_DIR:-/home/xuyouwen/hf_home_local/relation_block/nemotron_bpe2048_v1}"
GPU_IDS="${GPU_IDS:-}"
[[ "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)+$ ]] || { echo 'Set at least two GPU IDs, e.g. GPU_IDS=1,2,3,4'; exit 2; }
[[ -s "$DATA_DIR/manifest.json" ]] || { echo 'Missing prepared data manifest'; exit 2; }
if [[ "${RESUME:-0}" == 1 ]]; then
    [[ -d "$RUN_DIR" ]] || { echo 'Resume run directory does not exist'; exit 2; }
else
    [[ ! -e "$RUN_DIR" ]] || { echo "Run directory already exists: $RUN_DIR"; exit 2; }
    mkdir -p "$RUN_DIR"
fi
for name in REPO RUN_DIR DATA_DIR GPU_IDS; do printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; done
for name in GLOBAL_BATCH ADAPTER_RANK ADAPTER_LR SEED EVAL_LIMIT FINAL_EVAL_LIMIT HELDOUT_LIMIT RESUME SMOKE CONDA_ROOT CONDA_ENV MIN_FREE_GIB; do
    if [[ -v "$name" ]]; then printf 'export %s=%q\n' "$name" "${!name}" >> "$RUN_DIR/job.env"; fi
done
printf '#!/usr/bin/env bash\nsource %q\nbash %q >> %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nexit "$rc"\n' \
    "$RUN_DIR/job.env" "$REPO/relation_adapter/scripts/job.sh" "$RUN_DIR/job.log" "$RUN_DIR/exit_code" > "$RUN_DIR/entry.sh"
session="ra-frozen-$(date +%Y%m%d-%H%M%S)-$$"
tmux new-session -d -s "$session" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$session" "$RUN_DIR" "$session"

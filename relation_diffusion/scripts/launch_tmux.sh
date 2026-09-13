#!/usr/bin/env bash
# Launch exactly one explicitly selected phase. Never chain into long training.
set -euo pipefail
PHASE="${1:-}"
case "$PHASE" in prepare|preflight|train|pilot) ;; *) echo 'Usage: bash relation_diffusion/scripts/launch_tmux.sh prepare|preflight|train|pilot' >&2; exit 2;; esac
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_DIR="${DATA_DIR:-/home/xuyouwen/hf_home_local/relation_diffusion/wikitext2_byte128_v1}"
RUN_DIR="${RUN_DIR:-$REPO/relation_diffusion/runs/${PHASE}_$(date +%Y%m%d_%H%M%S)}"
SESSION="${SESSION:-relation-${PHASE}-$(date +%Y%m%d-%H%M%S)}"
GPU_IDS="${GPU_IDS:-}"
if [[ "$PHASE" != prepare && ! -s "$DATA_DIR/manifest.json" ]]; then
    printf 'Data preparation is not complete: %s/manifest.json is missing or empty.\n' "$DATA_DIR" >&2
    echo 'Check the prepare job.log and exit_code. Wait for successful preparation before launching; no GPU job was started.' >&2
    exit 2
fi
if [[ "$PHASE" != prepare && ! "$GPU_IDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo 'Set GPU_IDS explicitly, e.g. GPU_IDS=3 or GPU_IDS=0,1,2,3,4,5. Only use assigned idle GPUs.' >&2
    exit 2
fi
if [[ -e "$RUN_DIR" ]]; then echo "RUN_DIR already exists: $RUN_DIR" >&2; exit 2; fi
mkdir -p "$RUN_DIR"
ENVFILE="$RUN_DIR/job.env"
# Bash %q quoting preserves literal values; no eval or string-built commands.
for NAME in REPO DATA_DIR RUN_DIR PHASE GPU_IDS; do printf 'export %s=%q\n' "$NAME" "${!NAME}" >> "$ENVFILE"; done
for NAME in STEPS MAX_SECONDS GLOBAL_BATCH MICRO_BATCH WIDTH LAYERS HEADS SEED CODEC OBJECTIVE EVAL_LIMIT MIN_FREE_GIB CONDA_ROOT CONDA_ENV; do
    if [[ -v "$NAME" ]]; then printf 'export %s=%q\n' "$NAME" "${!NAME}" >> "$ENVFILE"; fi
done
printf '#!/usr/bin/env bash\nsource %q\nbash %q > %q 2>&1\nrc=$?\nprintf "%%s\\n" "$rc" > %q\nexit "$rc"\n' \
    "$ENVFILE" "$REPO/relation_diffusion/scripts/job.sh" "$RUN_DIR/job.log" "$RUN_DIR/exit_code" > "$RUN_DIR/entry.sh"
tmux new-session -d -s "$SESSION" "bash $(printf '%q' "$RUN_DIR/entry.sh")"
printf 'Session: %s\nLog: %s/job.log\nAttach: tmux attach -t %s\n' "$SESSION" "$RUN_DIR" "$SESSION"

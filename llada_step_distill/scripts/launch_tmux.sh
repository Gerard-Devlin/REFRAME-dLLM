#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: launch_tmux.sh MODE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
mkdir -p "$RUN_DIR"
SESSION="lsd-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
LOG="$RUN_DIR/job.log"
COMMAND="bash llada_step_distill/scripts/run.sh '$MODE' 2>&1 | tee '$LOG'; code=\${PIPESTATUS[0]}; echo \$code > '$RUN_DIR/exit_code'; exit \$code"
tmux new-session -d -s "$SESSION" "bash -lc \"$COMMAND\""
echo "Session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"

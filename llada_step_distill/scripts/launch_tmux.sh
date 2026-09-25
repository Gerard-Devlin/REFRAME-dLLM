#!/usr/bin/env bash
set -euo pipefail

MODE=${1:?usage: launch_tmux.sh MODE}
RUN_DIR=${RUN_DIR:?set RUN_DIR}
mkdir -p "$RUN_DIR"
SESSION="lsd-${MODE}-$(date +%Y%m%d-%H%M%S)-$$"
LOG="$RUN_DIR/job.log"
ENV_FILE="$RUN_DIR/job.env"
: > "$ENV_FILE"
for NAME in \
  PATH PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH \
  HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE TOKENIZERS_PARALLELISM \
  GPU_IDS DATASET NEMOTRON_ROOT PREPARED SUPERVISION SPLIT ATTEMPTS TRAIN_SIZE VALIDATION_SIZE SHARD_SIZE OVERSAMPLE \
  LIMIT UPDATES OVERFIT_RECORDS INIT_ADAPTER RESUME STEPS ADAPTER MERGED_MODEL CHECKPOINT; do
  if [[ -v "$NAME" ]]; then
    printf 'export %s=%q\n' "$NAME" "${!NAME}" >> "$ENV_FILE"
  fi
done
COMMAND="source '$ENV_FILE'; bash llada_step_distill/scripts/run.sh '$MODE' 2>&1 | tee '$LOG'; code=\${PIPESTATUS[0]}; echo \$code > '$RUN_DIR/exit_code'; exit \$code"
tmux new-session -d -s "$SESSION" "bash -lc \"$COMMAND\""
echo "Session: $SESSION"
echo "Log: $LOG"
echo "Attach: tmux attach -t $SESSION"

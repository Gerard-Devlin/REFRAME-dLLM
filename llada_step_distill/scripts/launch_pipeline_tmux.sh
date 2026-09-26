#!/usr/bin/env bash
set -euo pipefail

GPU_IDS=${GPU_IDS:?set GPU_IDS}
EXP_ROOT=${EXP_ROOT:?set EXP_ROOT}
NEMOTRON_ROOT=${NEMOTRON_ROOT:?set NEMOTRON_ROOT}
DATASET=${DATASET:?set DATASET}

PREPARED=${PIPELINE_PREPARED:-$EXP_ROOT/prepared}
SUPERVISION=${PIPELINE_SUPERVISION:-$EXP_ROOT/supervision}
RUN_ROOT=${PIPELINE_RUN_ROOT:-$EXP_ROOT/pipeline}
TENSORBOARD_PORT=${TENSORBOARD_PORT:-6006}

export GPU_IDS EXP_ROOT NEMOTRON_ROOT DATASET PREPARED SUPERVISION RUN_ROOT TENSORBOARD_PORT
mkdir -p "$RUN_ROOT"

SESSION="lsd-pipeline-$(date +%Y%m%d-%H%M%S)-$$"
ENV_FILE="$RUN_ROOT/pipeline.env"
JOB_FILE="$RUN_ROOT/pipeline.job.sh"
LOG_FILE="$RUN_ROOT/pipeline.log"
: > "$ENV_FILE"

for NAME in \
  PATH PYTHONPATH CONDA_PREFIX CONDA_DEFAULT_ENV LD_LIBRARY_PATH \
  HF_HOME HF_HUB_CACHE HF_DATASETS_CACHE HF_HUB_OFFLINE TRANSFORMERS_OFFLINE HF_DATASETS_OFFLINE TOKENIZERS_PARALLELISM \
  GPU_IDS EXP_ROOT NEMOTRON_ROOT DATASET PREPARED SUPERVISION RUN_ROOT TENSORBOARD_PORT \
  PIPELINE_PREPARED PIPELINE_SUPERVISION PIPELINE_RUN_ROOT \
  TRAIN_SIZE VALIDATION_SIZE SHARD_SIZE OVERSAMPLE SPLIT ATTEMPTS OVERFIT_UPDATES SMOKE_UPDATES PIPELINE_EVALUATE; do
  if [[ -v "$NAME" ]]; then
    printf 'export %s=%q\n' "$NAME" "${!NAME}" >> "$ENV_FILE"
  fi
done

WORKDIR=$(pwd)
cat > "$JOB_FILE" <<EOF
#!/usr/bin/env bash
set -o pipefail
source $(printf '%q' "$ENV_FILE")
cd $(printf '%q' "$WORKDIR")
bash llada_step_distill/scripts/run_pipeline.sh 2>&1 | tee $(printf '%q' "$LOG_FILE")
code=\${PIPESTATUS[0]}
echo \$code > $(printf '%q' "$RUN_ROOT/exit_code")
exit \$code
EOF
chmod +x "$JOB_FILE"

tmux new-session -d -s "$SESSION" -n pipeline "bash '$JOB_FILE'"
if command -v tensorboard >/dev/null 2>&1; then
  tmux new-window -d -t "$SESSION" -n tensorboard \
    "source '$ENV_FILE'; tensorboard --logdir '$RUN_ROOT' --host 127.0.0.1 --port '$TENSORBOARD_PORT'"
fi

echo "Session: $SESSION"
echo "Pipeline log: $LOG_FILE"
echo "TensorBoard: http://127.0.0.1:$TENSORBOARD_PORT"
echo "Attach: tmux attach -t $SESSION"

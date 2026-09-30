# dllm-eval

The `python -m dllm_eval.worker initialize --matrix main` campaign creates a fresh three-row
comparison: original LLaDA, Fast-dLLM v1 (PrefixCache + parallel), and FOCUS-dLLM on
that same cached parallel path, each at 256/512 on the four full test sets.
It rejects historical result directories. Each task writes scored `summary.json`,
individual sample records, and the scoring artifact; no presentation files are generated.
MATH's primary Minerva metric and secondary `math_verify` remain separate.

The scheduler borrows the document-parallel layout of
[lmms-eval's create_iterator](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/utils.py):
one persistent model per GPU, one active evaluation task, and remaining prompt
IDs sliced as `ids[rank::world_size]`. Atomic prompt claims preserve uniqueness
when a worker restarts or GPU availability changes. There is no inter-GPU model
communication. All ranks finish the current task before advancing; CPU scoring
can overlap the next task. The hard six-GPU cap and idle-GPU checks still apply.

As in [lmms-eval's generation loop](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/lmms_eval/models/simple/llava.py),
progress uses tqdm, but here the controller sums every GPU and prints newline
snapshots every 30 seconds so `tail -f` works. `elastic/progress.log` distinguishes
task progress, matrix progress, GPU count, rate, estimated remaining time, and
CPU scoring. An ETA is an estimate, not a deadline. Updates and queue I/O happen
outside the unchanged decoder's measured region. `scripts/launch_tmux.sh`
starts/resumes the campaign in tmux with explicit `GPU_IDS` and dataset paths.


Directory organization follows the separation of evaluation package, model
adapters, task configuration, and scripts in
[FlashVID/lmms-eval](https://github.com/Fanziyang-v/FlashVID/tree/main/lmms-eval).
This is a small LLaDA evaluator, not a vendored copy of the entire lmms-eval suite.

```text
dllm-eval/
  dllm_eval/
    models/llada.py        # adapter to the unchanged audited decoder
    tasks/main_table.json # fixed method and dataset settings
    worker.py             # initialization, generation, final scoring
    scheduler.py          # idle GPU selection and task-level sharding
    queue.py              # durable sample ownership and recovery
    progress.py           # global tqdm log snapshots
    score_*.py            # task scorers
  scripts/launch_tmux.sh
  tests/
  runs/<run-name>/         # ignored; manifest, logs, samples, metrics
```

Direct module commands require the package directory on `PYTHONPATH`:

```bash
export PYTHONPATH="$PWD/focus_dllm/dllm-eval:$PWD${PYTHONPATH:+:$PYTHONPATH}"
python -m dllm_eval.worker --help
python -m pytest focus_dllm/dllm-eval/tests focus_dllm/tests -q
```

The tmux launcher sets this automatically. No new environment is required.

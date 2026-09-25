# Fast-dLLM v2 two-to-one distillation

This directory adapts fixed-trajectory few-step distillation to the pinned
Fast-dLLM v2 1.5B checkpoint. It is an experiment, not an implementation or
reproduction claim for CDLM or OPTD.

The six stages are deliberately gated:

1. `audit` observes 32 native trajectories and computes an optimistic,
   non-overlapping same-sub-block two-step cost ceiling. A ceiling below 1.5x
   blocks data collection and training.
2. `collect` records 2,000 training and 128 validation prompts. Records contain
   only the initial native state plus the teacher's following two actions.
3. `smoke` fits 32 real states for 50 optimizer updates and checks DDP gradient
   agreement and exact save/resume behavior.
4. `train` runs one shuffled epoch for either `basic` or `release`, evaluating
   at 0/25/50/75/100 percent.
5. `export` merges rank-64 LoRA into a standalone BF16 checkpoint and verifies
   logits, native actions, short generation, and reload behavior.
6. `evaluate` compares the original and both merged branches at native
   thresholds 0.85/0.90/0.95. Held-out evaluation is locked until a development
   candidate reaches the configured quality/latency screen.

All stages consume an existing prepared prompt directory and evaluation JSON;
they perform no downloads. Every GPU stage requires explicit `GPU_IDS` and a
new output directory. Launch with:

```bash
bash step_distill/scripts/launch_tmux.sh audit
```

The launcher records its environment and runs in a detached tmux session. See
`python -m step_distill --help` for stage arguments. Artifacts and machine-specific
commands belong outside the repository under `step_distill/runs/`, which Git ignores.

The student stays in inference mode during optimization because this pinned v2
implementation changes RoPE behavior when `training=True`. Gradients still flow
through FP32 LoRA parameters and the student's own differentiable prefix cache.
The teacher is frozen and never appears in deployment.

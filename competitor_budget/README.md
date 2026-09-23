# Competitor-budget decoding (research prototype)

This directory changes only the commit set chosen after a normal Fast-dLLM v2
forward. It uses top-1 confidence `c_i` and full-vocabulary runner-up probability
`r_i`. A confidence-ordered prefix `S` is certified when
`sum(1-c_i) + max(r_i) < 1-margin`. The actual rule extends the native v2 set
only when the certified prefix contains every native commit. It never adds a
model forward, changes the chosen token, or changes the backbone.

The proof assumes current per-position marginals come from a single compatible
joint distribution. Neural conditionals need not satisfy this, so this is **not
lossless**. Probabilities are also rounded by the model's BF16 inference path.
The native/modified paths are compared on free-running accuracy,
forward count and measured time. This first implementation is pinned to the
official 1.5B checkpoint and its decoder source hash. The 7B checkpoint has a
different generation method and requires its own integration.

`MODE=observe` computes proposals without changing any native commit. It records
top-two probabilities and both commit sets per step, plus the fraction of extra
proposals that match the native trajectory's final token. This is only a gate,
not a speed measurement. `MODE=compare` runs native and active policies on the
same prompts, alternating execution order. Every run first checks that this
local decoder matches the pinned official method in output, model-forward count
and model-forward inputs on a short generation.

Input is a prepared GSM8K JSON list with `id`, `question`, and `answer`. The
prompts and numeric extraction match `relation_block.evaluate`, but this remains
an exploratory 0-shot evaluation rather than official lm-eval reproduction.

The launcher requires explicit physical `GPU_IDS` and an unused `RUN_DIR`:

```bash
GPU_IDS=0,5,6,7 DATASET=/path/to/gsm8k_dev_full.json MODE=observe \
  bash competitor_budget/scripts/launch_tmux.sh
```

Results appear in `RUN_DIR/eval/summary.json` and per-rank JSONL traces. A
high observed proposal rate is necessary before spending GPU time on
`MODE=compare`; the paired quality and wall-clock results decide usefulness.

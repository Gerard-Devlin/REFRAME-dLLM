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

## Conservative extensions

The original certificate can approve several individually uncertain tokens.
Compatible joint marginals are an assumption, and the resulting irreversible
commits can damage subsequent reasoning. The following fixed hypotheses restrict
**only additional commits**; native commits, including forced tokens, are kept.
They introduce no additional model calls or learned parameters.

| Preset | Margin | Extra confidence floor | Maximum extra tokens per decision | Consecutive confident argmax observations |
| --- | ---: | ---: | ---: | ---: |
| `original` | 0 | 0 | Unlimited | 1 |
| `guarded` | 0.20 | 0.85 | 2 | 1 |
| `strict` | 0.20 | 0.90 | 1 | 1 |
| `stable` | 0.20 | 0.85 | 2 | 2 |

All three new presets also leave EOS to the native rule. `stable` counts only
normal forwards on the current sub-block: changing the argmax, dropping below
the confidence floor, or filling the position resets its streak. A new
sub-block/request starts without history. Thus its first forward uses native
commits. Agreement is an empirical guard, not a proof that the token is correct.
`guarded` versus `stable` isolates the history requirement. `strict` trades more
of the potential speedup for a smaller set of extra commits.

`MODE=sweep` evaluates six methods per question: native at the configured
threshold, native at 0.90, and all four presets. The 0.90 baseline tests whether
the improvement exceeds simply lowering the native threshold. Each method is
warmed up, execution order rotates on every GPU, and the native reference is
measured once per question. Sweep inputs must have `train:` IDs; choose settings
on development questions and run the frozen choice on test with `MODE=compare`.
The preset values are unvalidated starting points, not accuracy recovery claims.

```bash
GPU_IDS=0,1 DATASET=/path/to/gsm8k_dev_full.json MODE=sweep LIMIT=256 MARGIN=0 \
  bash competitor_budget/scripts/launch_tmux.sh

# After choosing a preset on development data:
GPU_IDS=0,1 DATASET=/path/to/gsm8k_test.json MODE=compare PRESET=stable LIMIT=1319 MARGIN=0 \
  bash competitor_budget/scripts/launch_tmux.sh

python -m competitor_budget.report /path/to/run/eval/summary.json
```

The summary includes paired correct-to-wrong / wrong-to-correct counts, mean
generated length, truncation, NFE, and end-to-end per-sample latency (including
guard overhead). Speed is summed reference latency divided by summed method
latency, not total multi-GPU throughput. A smaller NFE alone is insufficient.
Compare the development quality/latency curve before spending another full
test run; a few development answers changing is not evidence of equivalence.

# Relation-space block diffusion: BPE feasibility pilot

This is the new main experimental implementation. `relation_diffusion/` is the
historical byte pilot; `v1/` and `v2/` are not modified. There are **no trained
relation weights or demonstrated speedups** in this directory.

## What is implemented

- Frozen starting checkpoint: `Efficient-Large-Model/Fast_dLLM_v2_1.5B`, revision
  `da5608172d2b74380e4e780baa19c71645e4f981`. Downloads run on the server only.
- Train-only sparse one-layer BPE coupling: fixed pairs (1,2), (3,4), ... in
  each 32-token block. For each supported anchor, swap its most frequent
  follower with one shared ordinary vocabulary ID; other IDs remain unchanged.
  This is a **frequent-follower candidate**, not a semantic relation extractor.
  The shared code ID is an ordinary token whose meaning becomes conditional
  on the anchor. This collision-free permutation is its own inverse.
- First token of each block, prompt tokens and all special IDs are protected.
  No answer-dependent pair selection, no future-block side information.
  O(vocabulary) lookup storage, no vocabulary-squared allocations.
- Token and relation arms start from identical weights. Both use rank-16 LoRA
  on Q/K/V/O and gate/up/down projections, frozen embeddings and LM head,
  no adapter dropout. This is a restricted adaptation experiment; a negative
  result does not establish failure of full-parameter relation training.
- Training uses complementary masks on eligible response positions. The noisy
  branch sees its own block and earlier **clean original-text** blocks only.
  The clean branch sees previous/current clean blocks. Positions are duplicated
  across the branches. A shifted head predicts each within-block target; the
  first raw token of a block is supervised from the previous clean block head.
  This boundary loss and relation/clean split are explicit adaptation choices,
  **not an exact reproduction of the original v2 fine-tuning recipe**.
- Chunked/recomputed vocabulary losses and layer gradient checkpointing reduce
  training memory. BF16 frozen weights, FP32 LoRA parameters/AdamW state.
- Global samples and mask draws are deterministic by step, independent of GPU
  count. Gradient accumulation is normalized by global response-token count.
  Single GPU / torchrun DDP. Periodic resumable adapter checkpoints.
  `GPU_IDS` is an explicit list of physical nvidia-smi indices. `smoke` and
  `pilot` campaigns train on all selected devices and evaluate serially on the
  first selected device. Global batch must divide by world size * micro batch;
  six GPUs with global batch12 and micro batch1 use two accumulation steps.
- Free-running GSM8K numeric accuracy, full generation latency, p95, tokens/s,
  call counts and output texts. Dev = first 128 GSM8K TRAIN examples, used for
  evaluation only; test is separately opt-in. No test-based codec fitting.

## Baselines and what the numbers mean

| Command phase | Weights | Sampler |
|---|---|---|
| `native` | Original official v2 | Official threshold=0.9, sub-block=8, block cache only |
| `baseline` | Original v2, explicit backend | Shared fixed-round confidence quota |
| `train` ARM=token then `evaluate` | Token LoRA | Shared fixed-round confidence quota |
| `train` ARM=relation then `evaluate` | Relation LoRA | Same shared sampler |

The native row is a reference, **not a claim to reproduce every optimized v2
configuration** (sub-block/DualCache is disabled here). The token/relation pair
isolates the representation change under the same scheduler. Do not attribute
native-vs-shared-scheduler differences to the relation codec. Before a paper,
add the tuned official strong-cache configurations and more seeds/tasks.

For the shared sampler, complete prefix blocks are cached; a partial prompt
block is jointly processed with the new positions. Completed generated blocks
are decoded back to original tokens and **encoded again** before appending
their clean KV. Both arms pay this cost; its forward calls and time are included.
Noisy/relation KV is never promoted to clean prefix KV. The final block is not
re-encoded if generation has ended. NFE is the actual calls, not the requested
round limit (there are boundary, prefill, early-stop and clean-encode effects).

Fixed rounds mean a confidence-ranked quota at each step, not the official
confidence threshold schedule. This pilot is 0-shot with a fixed chat prompt
and documented numeric extractor, not lm-eval's 5-shot GSM8K metric. Never
compare these accuracy values directly against a paper's differently prompted
table. 32 examples are a screening run, not a reliable quality-preservation claim.

## Data and budgets

Training source is the already tracked Alpaca conversation file under `v2/data/`.
It is a convenient engineering/feasibility set, **not official v2's full data**.
Exact duplicate token sequences are removed before a 256-example heldout split.
The codec is fit only on the remainder. Exact GSM8K question matches are excluded;
this does not establish absence of paraphrases or pretraining contamination.
Long prompts (>256 tokens at default length512) are excluded; responses are
truncated to length512. Logs count original unpadded tokens as well as padded
slots. Complementary/noisy-clean computation is extra and is not misreported as
additional unique training text.

Default 200 updates * global batch12 * length512 = 1,228,800 padded slots;
actual original tokens are smaller and recorded in `status.json`. This is an
engineering/early-adaptation run, not a promised sufficient learning budget.
No automatic escalation to 10M/100M tokens. Compare both completed budgets first.
Same original-token budget does not guarantee identical wall-clock compute;
record `training_seconds`, peak memory and hardware separately.

## Gates and tests

`prepare` publishes manifest last; data hashes are rechecked by every GPU phase.
`preflight` first invalidates any old gate, then:

1. compares real-checkpoint logits to the pinned official remote implementation;
2. compares cached and uncached outputs at a complete block boundary;
3. checks token/relation forward + backward on real weights, without an optimizer step.

Only a passing gate for the exact implementation/data enables training/evaluation.
Remote code is executed only for official parity/native evaluation, from the
pinned local snapshot downloaded through the mirror. The explicit backend loads
safetensors with strict keys and rejects unsupported RoPE/sliding-window configs.

Local tests use random tiny weights only, no checkpoint downloads:

```bash
python -m pytest relation_block/tests -q
RUN_DDP_TESTS=1 python -m pytest relation_block/tests -q
```

Checks cover bijection/specials/prompt/block boundaries, sparse fitting, local
error propagation, transitive attention leakage, independent Qwen reference
parity, cache parity, LoRA gradients/checkpoints, free generation, microbatch
equivalence and optional real two-process CPU/Gloo gradient equivalence.

Local tiny tests do not establish real-checkpoint/NCCL compatibility; server
gate results and completed training/evaluation logs are required.

## Stop / continue criteria

Stop on parity, cache, inversion, NaN, or leakage failures. If adapted token
quality collapses relative to the same-sampler original model, fix the training
recipe before attributing anything to the codec. If relation adaptation only
recovers recoding damage or has no free-generation quality/latency advantage,
do not scale automatically. Native quality need not monotonically improve with
requested rounds. Judge the full observed frontier, including the token
baseline's cheapest settings, with paired samples and uncertainty.

Upstream references: [model](https://huggingface.co/Efficient-Large-Model/Fast_dLLM_v2_1.5B),
[Fast-dLLM v2 code](https://github.com/NVlabs/Fast-dLLM/tree/main/v2).

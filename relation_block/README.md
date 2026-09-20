# Relation-space block diffusion: BPE feasibility pilot

This is the new main experimental implementation. `relation_diffusion/` is the
historical byte pilot; `v1/` and `v2/` are not modified. There are **no trained
relation weights or demonstrated speedups** in this directory.

## What is implemented

- Pinned starting checkpoint: `Efficient-Large-Model/Fast_dLLM_v2_1.5B`, revision
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
- Token and relation arms start from identical weights and train **all parameters**,
  including tied embeddings/output head, attention, MLP and normalization layers.
  The production training/evaluation path does not use LoRA.
- Training uses complementary masks on eligible response positions. The noisy
  branch sees its own block and earlier **clean original-text** blocks only.
  The clean branch sees previous/current clean blocks. Positions are duplicated
  across the branches. A shifted head predicts each within-block target; the
  first raw token of a block is supervised from the previous clean block head.
  This boundary loss and relation/clean split are explicit adaptation choices,
  **not an exact reproduction of the original v2 fine-tuning recipe**.
- Chunked/recomputed vocabulary losses and layer gradient checkpointing reduce
  training memory. FP32 master parameters and AdamW state, BF16 autocast forward.
  The chunked trainable output head runs inside the DDP forward.
- One seeded, shuffled epoch without replacement; the final partial batch is
  token-normalized and empty ranks participate with zero-weight dummy examples.
  DDP + ZeroRedundancyOptimizer shard AdamW states across selected GPUs. Single
  GPU uses AdamW. Six GPUs, global batch12, microbatch1 accumulate twice.
- `GPU_IDS` explicitly selects physical nvidia-smi indices. `full` runs both arms
  sequentially on all selected GPUs; evaluation runs on the first selected GPU.
  `pilot` uses the same epoch budget but requires pre-existing gates. `STEPS=0` means
  one epoch; a positive value explicitly limits the run within that epoch.
- Full checkpoints every 500 updates, at evaluation points and at completion:
  FP32 model, per-rank optimizer/RNG shards, sampler/scheduler position, BF16
  inference export. Retain the newest two checkpoints per arm. Resume requires
  the same GPU count, data, implementation and planned schedule. Old adapter
  checkpoints cannot resume full training. Checkpoint publication is atomic.
- TensorBoard and JSONL report globally token-weighted loss, gradient norm,
  learning rate, original/supervised tokens, throughput, ETA and per-rank memory.
- Free-running GSM8K numeric accuracy, full generation latency, p95, tokens/s,
  call counts and output texts. Dev = first 256 GSM8K TRAIN examples, used for
  evaluation only; test is separately opt-in. No test-based codec fitting.

## Baselines and what the numbers mean

| Command phase | Weights | Sampler |
|---|---|---|
| `native` | Original official v2 | Official threshold=0.9, sub-block=8, block cache only |
| `baseline` | Original v2, explicit backend | Shared fixed-round confidence quota |
| `train` ARM=token then `evaluate` | Token full training | Shared fixed-round confidence quota |
| `train` ARM=relation then `evaluate` | Relation full training | Same shared sampler |

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

The default data path now uses a bounded export from
`nvidia/Llama-Nemotron-Post-Training-Dataset`, configuration `SFT`, splits `math`
and `code`. `download_subset` streams through the mirror, pins the resolved
dataset revision, applies a seeded bounded shuffle and retains at most half the
requested token budget per category. Token counts use the pinned v2 tokenizer
and chat template, including prompt and response, not bytes or sample counts.
Default export budget is 100M original tokens (50M each category), length2048.
Only whole examples fitting the window and prompt<=half the window are kept;
long reasoning traces are skipped, **never truncated into unfinished answers**.
This is a length-selected, bounded stream sample, not an unbiased sample of the
whole corpus. Export reports scanned/rejected counts and preserves provenance.
Reasoning on/off are both accepted unless explicitly filtered. The auxiliary
`system_prompt` metadata label is retained as metadata; actual conversation
messages are taken from `input` and the complete assistant text from `output`.

Completed category files can be reused after interruption; an incomplete
category restarts deterministically. Streaming network bytes can exceed the
retained subset size. Download budget and training-consumption budget are
separate: downloading 100M tokens does not train for 100M tokens.

Preparation verifies the subset hashes, removes exact duplicates and holds out
256 examples, excluding their exact tokenized prompts from training too. The
codec is fit only on training examples. Exact GSM8K question matches are excluded;
this does not establish absence of paraphrases or pretraining contamination.
The old bundled Alpaca path remains explicitly opt-in (`TRAIN_SOURCE=alpaca`),
for engineering checks only. These are not official v2's complete training data
or recipe. Logs count original unpadded tokens as well as padded slots.
Complementary/noisy-clean computation is extra, not additional unique text.

Default training consumes one complete prepared training epoch. Learning rate
2e-5, 3% warmup, cosine decay, AdamW weight decay0.01, clipping1.0. Intermediate
evaluation every 1000 updates uses 32 fixed dev examples; final evaluation uses
256, at rounds2/4/8/16 and max generation512. Length-cap rates are reported.
There is no default one-hour stop. Explicit MAX_SECONDS saves and stops;
resume is explicit. Training compute, evaluation and checkpoint I/O are not
reported as the same timing metric. Short-smoke measurements give provisional
full-epoch compute ETA; they do not predict learning quality.

## Gates and tests

`prepare` publishes manifest last; data hashes are rechecked by every GPU phase.
`preflight` first invalidates any old gate, then:

1. compares real-checkpoint logits to the pinned official remote implementation;
2. compares full/prefill/cached paths against the corresponding official paths;
3. checks token/relation forward + backward on real weights, without an optimizer step.

Preflight also extends the original 128 dev items to 256 from the offline cache,
checking that the original prefix is unchanged. The existing data manifest is
not rewritten. Full checkpoint metadata fingerprints the extended dev file.

Formal training additionally requires `smoke`: on the actual selected GPUs,
each arm runs four full optimizer steps, saves/exits, restores the sharded
checkpoint, finishes four more steps, and evaluates the full BF16 export.
The gate matches implementation/data/GPU-count/batch settings. OOM or resume
failure stops the campaign; no fallback to LoRA or a smaller batch occurs.
`full` performs preflight and actual-topology smoke automatically before both
training arms. Standalone `smoke` never starts the epoch. Successful smoke
discards its temporary large checkpoints after export evaluation; logs, metrics,
status and gate results remain. Failed smoke checkpoints remain for diagnosis.
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
parity, cache parity, full-parameter gradients, BF16 autocast/FP32 optimizer,
checkpoint retention/export, exact next-update resume, epoch tail coverage,
free generation and real two-process CPU/Gloo sharded-optimizer equivalence.
Historical LoRA utility tests remain separate from the training path.

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

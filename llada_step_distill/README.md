# LLaDA-8B-Instruct two-step distillation

This directory contains an isolated experiment for teaching the pinned
`GSAI-ML/LLaDA-8B-Instruct@08b83a6feb34df1a6011b80c3c00c7563e963b07`
checkpoint to match two teacher denoising updates with one student update. It does
not modify `v1/`, `v2/`, or `fastv_dllm/`.

The experiment intentionally has a hard pre-training gate. `audit` measures the
unmodified teacher at 16 and 32 fixed-quota steps per 32-token block. If their
development accuracy is already equal, the report recommends using 16 steps
directly instead of distilling.

The complete pipeline is:

1. `prepare` deterministically selects and decontaminates 10,000,000 unique
   Nemotron SFT rows and 20,000 validation rows. The exact domain quotas and data
   hashes are written to `manifest.json`.
2. `collect` creates aligned supervision shards. Teacher computation is sparse:
   1,000,000 rows use one sampled partial state with two teacher-correct release
   targets, 9,000,000 rows use response-only LLaDA SFT masking without a teacher
   forward, and 200,000 of the acceleration rows use an actual second teacher
   forward. No record stores a complete 512-token denoising trajectory.
3. `smoke` runs real optimizer updates, checkpoint save, and resume with the
   requested DDP topology. It does not silently reduce the rank or sequence
   length after OOM.
4. `train --stage a` trains one no-replacement epoch. `train --stage b` starts
   from the Stage-A adapter with a fresh optimizer and consumes only the 200,000
   real-transition records.
5. `evaluate` uses the original fixed-quota LLaDA decoder at 8/16/32 steps per
   block. `export` merges the adapter and verifies logits, actions, generation,
   and reload parity.

All attention paths use the checkpoint's FlashAttention integration. Because
teacher and student share that backend, measured changes between them are due to
the learned denoising behavior rather than an attention-kernel switch.

## Entry points

```text
python -m llada_step_distill prepare  ...
python -m llada_step_distill collect  ...
python -m llada_step_distill audit    ...
python -m llada_step_distill smoke    ...
python -m llada_step_distill train    ...
python -m llada_step_distill evaluate ...
python -m llada_step_distill export   ...
python -m llada_step_distill report   ...
```

`scripts/launch_tmux.sh` is a generic launcher. It takes all paths and GPU ids
from environment variables; it contains no machine-specific paths.

The rank-256 adapter covers exactly 224 per-layer projections and contains
671,088,640 trainable FP32 parameters. The original embeddings, output head,
normalization layers, and backbone parameters remain frozen. Multi-GPU training
uses DDP plus `ZeroRedundancyOptimizer`; checkpoints retain one optimizer shard
and RNG state per rank and therefore require the same world size for exact
resume.

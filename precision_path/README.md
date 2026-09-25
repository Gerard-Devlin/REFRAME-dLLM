# Precision-path diagnostics

Frozen Fast-dLLM v2, fresh current-state inference, and numerical precision as
the compute choice. No relation codes, adapters, training or stale-probability
replay. The old Relation source directories are removed; historical results and
Git history are not deleted.

## Current scope

This is **cost measurement and offline paired-trajectory analysis**, not an online
adaptive-precision executor. It does not claim speedup, lossless generation, or
novelty for quantization/conformal calibration.

The initial backend explicitly quantizes transformer q/k/v/o/up/gate/down linear
weights and current activations to FP8 E4M3 and invokes `torch._scaled_mm`.
Embeddings, tied output head and native attention/norm/RoPE paths stay unchanged.
Weight scaling is per-tensor RTN; activation scaling is dynamic per tensor;
accumulation uses `use_fast_accum=False` and BF16 output. No optimizer or training
data are used to quantize. Conversion is performed once and both copies remain
resident. The runtime reports dual-model peak memory. Unsupported kernels fail;
there is no fake-quant BF16 matmul fallback. `_scaled_mm` is a private API, so a
real device probe is mandatory. This basic backend may lose to BF16 at M=32.

## Staged execution

All launches run under tmux, use the existing environment and offline HF cache,
and accept explicit physical `GPU_IDS`. Selected GPUs must have no active compute
process and at least 28 GiB free. No process is stopped and no environment is
installed. Each GPU handles independent prompts with batch=1; no training/DDP
gradient synchronization is involved. The launcher prints its log and session.

1. `bash precision_path/scripts/launch_tmux.sh cost`: set `GPU_IDS`, `DATA_DIR`,
   optionally `RUN_DIR`. Defaults: 4 development prompts (or GPU count if larger), 512 generated tokens,
   4 measured ordinary states per prompt, 5 interleaved BF16/FP8 repetitions.
   `output/summary.json` reports rho and a zero-fallback optimistic cost model.
   A backend must clear the 1.5x cost target before audit. Early sampled states
   are a cheap rejection screen, not a global hardware bound.
2. `bash precision_path/scripts/launch_tmux.sh audit`: additionally set
   `COST_REPORT` to a passing cost summary and `ROLE=development|calibration|evaluation`.
   Backend/runtime/source/decode settings, including generation length, must
   match the cost report. Set `PROMPTS` explicitly. Every ordinary reference
   state is paired; no reservoir or truncation of the trajectory is allowed.
   A separate static-low generation is timed per prompt; it quantizes cache
   writes too, and is only a baseline. Output equality is not task accuracy.
   Development audits also export `development_risk_coverage.json` and `.csv`:
   accepted-call coverage, whole-request error, time-weighted fallback and cost
   across fixed score quantiles. Do not tune thresholds on evaluation-role data.
3. `bash precision_path/scripts/launch_tmux.sh calibrate`: set
   `CALIBRATION_REPORT`, `EVALUATION_REPORT`, `ALPHA` (default .01). This is a
   CPU postprocess, using request-level maxima and independent evaluation.

Common settings: `THRESHOLD=.90`, `MAX_NEW_TOKENS=512`, `SEED=1234`,
`CONDA_ENV=fastdllm311`. Existing prepared data must contain `manifest.json`,
`train.json`, `heldout.json` with token IDs and prompt prefix lengths. Reference
answers are not used. No new model or dataset download is required. All paths
and knobs are captured in each run's `job.env`; existing output directories are
rejected. Environment/server-specific walkthroughs belong in chat, not this repo.

## State and decision checks

The original BF16 generator drives all observations. Low calls see the same
current tokens and BF16 historical KV, with cache updates forbidden. Persistent
KV values are compared before/after the paired calls; any mutation aborts.
All prefill, completed-block writes and next-block-first-token duties remain
official BF16. Actual native sampler probabilities determine the action labels;
the threshold is strict `>`, includes forced argmax, and includes EOS token values.
Recorded actions are checked against the next native input when one exists.
The paired pass must reproduce the reference BF16 output exactly.
Both generation timing baselines include the same bounded-call guard. A quantized
model that repeatedly predicts MASK without progress aborts; it is never silently
truncated and reported as fast.

Cache checks and reference labels are OFFLINE diagnostic costs. They are not
claimed to be free online certification. A future executor still needs an
independent cache/output/first-divergence audit before claiming a trajectory bound.

## Score and calibration limits

The fixed nonnegative score combines token margin, confidence-threshold margin,
and forced-position competition. It uses **only current low logits**. The ideal
float32 logit perturbation radius is a ranking feature; BF16 softmax rounding and
the actual network quantization error are NOT certified by that feature.

Each complete request contributes `R=max(score on mismatching actions)`, or zero
when no action differs. The conformal order statistic is
`ceil((n+1)*(1-alpha))`. If it exceeds n, the threshold is encoded as null with
`accept_none=true`; no low action is accepted. Acceptance is strictly `score >
threshold`. In particular, n=32 cannot give nontrivial alpha=.01 acceptance.

Development and calibration prompts are disjoint hash partitions of the prepared
training prompts; heldout is used for evaluation. Prompt duplicates are removed
across source splits. Calibration validation checks config identity, role,
complete trajectories, request maxima and prompt disjointness. Score/backend
selection belongs to development only. Statistical exchangeability with future
requests is an assumption, not established by these checks or by disjointness.

The rank statement is marginal over calibration and new requests. It is not a
per-request/subgroup guarantee and is not portable across templates, backends,
batch shapes or changed generation limits. Low forward cost is always paid;
rejection adds BF16 cost. The postprocess reports reference-path acceptance risk,
time-weighted fallback and a modeled ordinary-call cost ratio, not actual online
speed or accuracy. A zero-acceptance policy currently still pays low+BF16 in that
model; it should not be described as fast.

## Closest references

- [PyTorch quantized inference](https://docs.pytorch.org/ao/stable/workflows/inference.html)
  distinguishes real low-precision GEMMs from FP8 weight-only BF16 compute and
  documents shape/kernel overhead limitations.
- [DLLMQuant](https://arxiv.org/abs/2508.14090) and
  [FAIR-Calib](https://arxiv.org/abs/2606.06547) are direct dLLM PTQ neighbors.
- [Conformal prediction introduction](https://arxiv.org/abs/2107.07511) supplies
  the existing finite-sample rank principle; this project does not claim it as new.

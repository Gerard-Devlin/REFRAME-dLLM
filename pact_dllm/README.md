# PACT-dLLM (experimental)

Dependency-driven cache planning and packed verification for a frozen LLaDA.
This changes the sampler and conditional context; it is not an exact or
lossless replacement for native bidirectional decoding. No speed/accuracy or
novelty claim is supported merely by this implementation.

## Implemented path

1. A full-canvas prefill initializes every KV. Normal proposals refresh active
   MASKs and every identity-dirty token. Threshold `.90` plus argmax fallback
   guarantees progress without special early EOS shortcuts.
2. Layer4 Q and existing K supply inexpensive coarse tile and candidate
   interactions. Age and **last observed** refreshed-K drift adjust this proxy;
   no current oracle KV or reference answer is used.
3. Confidence orders candidates. An arbitrary ordered DAG caps parents and
   ancestors. Maximum-weight closure jointly selects candidates, prerequisites,
   and shared refresh tiles. Each selected tile costs once. The proxy optimizer
   is exact; its row-cost/acceptance estimates are not a latency certificate.
4. A packed forward has draft rows, masked verification rows and a draft-free
   background bank. Each original position has exactly one eligible KV version.
   Data rows see their own label and ancestors; verification sees ancestors;
   shared background never sees unconfirmed labels. Transitive closure bounds
   all-layer information paths, rather than masking only a diagonal.
5. Commit only individually passing candidates with all parents accepted.
   Promote only clean background KV. Newly committed identities become dirty
   and must receive a paid refresh in the next normal proposal. Draft KV is
   never promoted. Fixed prefixes do not move; unselected context is retained.

The initial benefit is squared draft confidence; verify cost is two query rows,
tile cost is its number of rows, price `.12`. This is a preregistered first
configuration, not a calibrated performance model. Optional tiles are bounded
to eight four-position tiles; the optimizer is exact within that pool. Mandatory
identity repair and current-window MASK refresh are outside the optional plan
and always paid. The first implementation prioritizes correct semantics and
measurable overhead over a claim of optimized kernels.

## Run

CPU checks: `python -m unittest pact_dllm.test_pact -v`.

GPU evaluation requires the existing pinned Flash source, existing frozen model
cache and physical GPU1 contract. Use `python -m pact_dllm.evaluate --help`.
`reference`, `cache_only`, `decode_only`, `joint` use identical projection,
attention and readout infrastructure. Reference is our chain ablation, **not a
new run of original LLaDA/v1 or a reproduced Flash-Verify benchmark**.

`mechanism_smoke` is two already-used prompts/task and64tokens, with no task
accuracy claim. Development quality selection requires at least128/task;
holdout excludes the first128 shuffled development IDs. Source/config/data
hashes gate resume. Final-expression-v3 and official code execution are applied
only after generation; legitimate prompt fields are explicitly whitelisted.

The first accuracy screen runs `joint` on the same128development prompts/task
as the completed Flash/v1/FOCUS results, at256positions. `--baseline-root` checks
ordered prompt IDs, data/model/revision/scorer hashes before loading the model.
Completed baselines are read, never regenerated. Logs include correct counts,
unresolved scores, accuracy, and a paired bootstrap interval versus Flash.
Unresolved outcomes keep the full accuracy unknown and expose explicit bounds.
This is development evaluation, not an independent non-inferiority certificate.

The user explicitly authorized sharing GPU1 on2026-10-03. A deployment must opt
in with `allow_shared_gpu1`; the default remains exclusive. Shared launch checks
the exact UUID, at least24GiB free memory, prior completion, protected sources,
and the research lock. No other process is terminated or reset. Shared timing
is marked separately and does not establish dedicated-card speedup.

Recorded latency includes prefill, initialization, signal extraction, CPU
planning/synchronization, every forward and cache repair. Private perturbation
and dense controls run in a separate generation; their timing is not a speed
baseline. Record full tokens/actions/NFE, per-task summary, truncation and EOS.
In a smoke run each configuration is replayed after its kernel shapes compile;
first-execution time is retained separately. Tokens, actions and NFE must match
the replay. Tile pooling uses batched gather/reduction, with a bitwise check
against the original per-tile statistic. Four-node private fork/join controls
test the actual32-layer path even when the planner selects too few candidates
to expose branching. They do not demonstrate task quality.

No training, new checkpoint, early EOS shortcut, or automatic parameter sweep.
`focus_dllm/README.md`, old main results and original LLaDA/v1 remain unchanged.

## Limitations to test

Sparse dependencies can remove useful bidirectional context. Observed context
drift is approximate even with exact token identities. Per-node verification
probability is not a joint correctness certificate. Planning may cost more than
it saves, or reject all optional work. GPU-fault atomic recovery is not claimed:
an arena is discarded after execution failure. No failed experiment is silently
retried or expanded to a quality sweep.

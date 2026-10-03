# FOCUS-v4 conditional-residual solver: research specification

This is a proposed algorithm, not an implemented decoder, a speed result, or an
established novelty claim. The running development128 pipeline remains frozen.
No Python module, model, configuration, scorer, or baseline is changed by this
document. The original goal remains unachieved.

## Question motivated by the completed measurements

On development128 HumanEval256, budget20 uses 1.548 times the model calls of
current Flash-Verify and takes 1.715 times as long. On MBPP256 it already keeps
only 5.54% of observed future positions, yet uses 2.099 times the calls and takes
2.719 times as long. Block preparation is approximately 0.8-1.2% of request
latency. These are request-level accounting identities, not kernel profiles or
causal attribution. More aggressive physical pruning is not the main proposed
lever: MATH128 budget20 scores 27.34%, versus 37.50% for both io_borrow and Flash.

The new question is whether a block can reach a useful, revisable conditional
prediction in fewer *paid model evaluations*, by solving its update residual
instead of repeatedly making irreversible threshold decisions.

## Proposed state and operator

Keep the model weights, vocabulary, original positions, formal prefix and
future MASK positions. In the first feasibility experiment, do not physically
prune or reuse stale future hidden/KV. The current 32-slot block contains latent
candidate embeddings E, not finally committed tokens. A candidate state can be
changed until the block is sealed. This changes decoding semantics and is not
lossless equivalence to LLaDA, FOCUS-v1, or Flash.

Split block slots into two deterministic interleaved groups A and B. Define
one fixed conditional cycle T = T_B composed with T_A:

1. In the actual input to T_A, all A slots are hard MASK embeddings; B slots
   contain their current latent candidates. Run the model. Only distributions
   at A are used to update A. Do not treat outputs at visible B slots as valid
   checks of their own labels.
2. In the actual input to T_B, all B slots are hard MASK embeddings; A slots
   contain the newly updated candidates. Run the model and update B.

Every cycle therefore costs TWO complete Transformer evaluations. Erasing a
whole group changes the conditioning distribution; this is not exact
leave-one-out validation or a sample from the original joint distribution.
There is no duplicate label-bearing view, no diagonal-only self-leakage claim,
and no cache from a label-bearing input retained at its erased query slots.

Candidate construction is a shared component for all solver controls, using
the current paid top-k probabilities and frozen token embeddings. Its cost,
normalization and behavior on uncertain mixtures must be measured. Soft inputs
are outside the model's discrete training inputs; self-consistency can reinforce
an incorrect answer. Neither a small residual nor a repeated argmax proves task
correctness. Known soft-token constructions must be credited as components.

## The proposed accelerator

For the unchanged conditional operator of one block, save paid cycle pairs
(E_j, T(E_j)) and residuals R_j = T(E_j) - E_j. Fit a small, regularized
multisecant combination, subject to coefficients summing to one:

    minimize ||sum_j alpha_j R_j||_F^2 + epsilon ||alpha||_2^2
    subject to sum_j alpha_j = 1.

Use sum_j alpha_j T(E_j) as a proposed accelerated state, with a bounded
per-slot displacement and the same candidate-domain projection as the control.
The proposal is NOT directly accepted as an output. The next scheduled real
conditional cycle evaluates it and pays both model calls. An increased observed
residual rejects the extrapolation and restores a saved paid anchor. Charge the
failed cycle, fitting, device synchronization, projection and rollback.

Mix histories only for the identical operator: same block, fixed formal prefix,
future canvas, group ordering and candidate construction. Restart at every
block boundary or context change. Half-cycle residuals are from different
operators and cannot be silently fitted as if they were full-cycle residuals.
No teacher states, oracle KV, gold answers, tests or solutions enter construction.

Anderson/multisecant acceleration itself is classical. The candidate research
contribution would have to be the specific conditional operator, its constrained
residual treatment and demonstrated reduction in useful dLLM evaluations. Merely
adding Anderson to a soft-token decoder is insufficient evidence of novelty.
Top-k and projection can make this operator nonsmooth. No global contraction,
convergence, task-accuracy or exact-trajectory theorem is claimed.

## Costs and falsifiable gates before deployment

Initialization uses a real paid warm forward and creates formal prefix KV.
With c conditional cycles, a block costs at least 1 + 2c model calls, before any
extra final check. Acceleration needs history and a subsequent paid evaluation;
it cannot retrospectively claim that its extrapolated state saved those calls.

Measure full-request time, not only the small residual fit. For a target speedup
s against Flash, observed new work must satisfy T_new < T_Flash / s. On MBPP128,
the current 1.25x expansion requirement is below 0.5325 seconds/request. If the
initialization and earliest possible validated accelerated cycle already exceed
this budget at realistic output lengths, do not build a larger executor.

EOS handling is a separate execution policy shared by all NEW controls. Seal a
complete hard block before finishing after its EOS; no unresolved prefix holes,
provisional latent slots or unverified guessed EOS. This is not the novel solver
mechanism. Existing LLaDA/v1 records, all main-table scores, and pinned official
Flash remain unchanged; compare historical LLaDA/v1 timings with that label.

Required controls share candidate construction, operator, termination and costs:

- Ordinary conditional cycles without acceleration.
- Fixed damping/overrelaxation without a residual fit.
- The constrained, safeguarded multisecant solver.

Tiny interface/CPU checks are not accuracy selection. Quality screening is at
least128 development examples per task, with one common configuration, frozen
scoring and all actual calls/setup/rollback charged. No per-task winner selection.
Independent validation follows only a genuine all-task quality-speed signal;
HumanEval has only36 unused IDs after the current128-example selection.

Reject this direction if it converges to incorrect self-consistent outputs,
fails to reduce real model evaluations versus the matched cycle control, has
insufficient history before normal termination, or loses the gain after
initialization/correction costs. Do not call a residual reduction an accuracy
improvement or treat a new name as a contribution.

## Closest published mechanisms checked before implementation

- [COVER](https://arxiv.org/html/2602.06161v1): one-pass cache-override verification
  and drafting. A combined correction/drafting pass is already prior work.
- [Tolerator/FiRe](https://arxiv.org/abs/2510.05090): fill-then-refine with token
  cross-validation. Alternating erasure and correction is already prior work.
- [I-DLM](https://arxiv.org/abs/2604.11035): strided introspection with a causal
  model/training design. Its AR equivalence cannot be transferred to this fixed
  bidirectional LLaDA.
- [Rethinking Soft Tokens](https://arxiv.org/html/2609.37391v1): frozen-model
  geometric candidate feedback. Soft states and spherical interpolation are
  already prior work, not our proposed contribution.
- [Mean-Field Parallel Decoding](https://arxiv.org/html/2606.15805v1): fixed-point
  updates for a within-forward commit-score relaxation. The proposed residual
  here is of a paid, multi-evaluation conditional *block-state* operator; that
  distinction needs verification against the full method before novelty claims.
- [Anderson's original method](https://doi.org/10.1145/321296.321305) and
  [geometry-aware diffusion attention guidance](https://arxiv.org/abs/2603.02531):
  multisecant iteration and connections between attention extrapolation and
  Anderson acceleration also predate this proposal. Moving a familiar solver
  into dLLM code alone is not an independent contribution.

The search does not establish that no identical method exists. Further nearest
neighbor review and real measurements are mandatory before a paper claim.

## Additional source and feasibility audit (2026-10-03)

[Fixed-Point Masked Generative Modeling / CoFRe](https://arxiv.org/html/2605.31215v1)
already places a numerical fixed-point solver inside a masked denoiser. It
replaces a stack with shared layers and trains/adapts that architecture, including
cross-step consistency and token-aware state reuse. Our proposed operator instead
acts on a candidate block using the unchanged, entire pretrained LLaDA network.
This is a difference in the object being solved; it is not by itself evidence of
an original contribution or a superior cost-quality tradeoff. Neither adaptive
depth nor the generic phrase "fixed-point masked generation" can be claimed as
new here.

The actual `load_model` implementation imports
`v1.llada.model.modeling_llada.LLaDAModelLM`. Its wrapper forwards `inputs_embeds`
to the core's `input_embeddings`, and the core supports prefix KV. Reading only
the Hugging Face snapshot's remote model class would have given a different
cache contract; that snapshot implementation is not the runtime under test.
This is source-level evidence only. Before any soft-state experiment, require
real BF16/Flash tests of input IDs versus their exact embedding lookup, with
and without the same immutable prefix cache. Check logits, selected positions,
tokens, absolute RoPE positions and cache contents/versions. Do not edit v1.

For the first proposed accelerated cycle, two paid residual observations and a
subsequent paid evaluation require at least seven full-model calls per block
(one warm plus three two-call cycles). This cost is a floor for that particular
protocol, not a universal floor for future solvers. A method that finishes before
it obtains this history gains nothing from the multisecant component. Changing
groups, candidate projection, EOS extent or the prefix changes the operator and
invalidates its fitted history.

No number of saved original-trajectory blocks can certify how many blocks a new
decoder will generate. Any replay estimate using those lengths is explicitly a
counterfactual feasibility calculation, not measured end-to-end speed. A small
actual operator-cost check must include complete warm calls, both suffix
evaluations, full future queries, group readout, embedding construction and
residual-fit overhead. If this budget already consumes the Flash target before
useful convergence, stop rather than build the speculative executor.

The current design remains a candidate whose novelty is unresolved. A generic
soft-token feedback loop with an off-the-shelf solver is not an acceptable final
paper claim. Require an ablation showing that the proposed conditional residual
mechanism reduces paid calls at matched quality beyond ordinary cycles and fixed
damping, plus a concrete distinction from the closest published operator. No
solver code has been implemented and no solver GPU job has been launched.

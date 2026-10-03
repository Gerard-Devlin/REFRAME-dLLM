# FOCUS-v7 cache-producing prefix verifier

Research question: can a 64-row verification packet produce legal reusable KV
and the next proposal, so repeated large regular cache passes are unnecessary?
This is a bounded mechanism probe, not a deployed generator or achieved speedup.

The packet contains clean tracked rows, clean candidate MASK rows, ordered draft
sources and ordered MASK audits. Each query has exactly one version of every
candidate position. A draft source i sees draft identities <=i; audit i sees only
identities <i. Future positions use clean MASK sources; clean context never reads
speculative identities. Therefore rejected later identities cannot contaminate
the accepted prefix's source KV at any layer. Actual native Triton private K/V
buffers are captured and partially promoted into a private shadow bank.

Keep the verified prefix and, if possible, one argmax correction at the first
rejection. This correction has no matching input-identity KV and MUST be queried
as a changed identity in the next packet. Clean logits are at most proposals for
the next iteration, never direct commitments under the changed state. Static
external and clean future KV remain approximate: this is not native equivalence.
EOS and output handling must receive explicit treatment before free generation.

The useful difference to test is removing a repeated cache/proposal pass using
prefix-conditional KV already produced by verification, with bounded identity
repair. Prefix verification, correction tokens, tree masks and pipelining are
existing tools, not independent novelty claims. [FreeDave](https://arxiv.org/abs/2510.00294)
already pipelines drafting and verification; its pinned implementation expands
full states/branches and does not establish this packet's quality or cost.
No publishability claim follows from this source design.

## Predeclared mechanism gate

Six reused dev prompts (two each HumanEval/MBPP/MATH), two earliest eligible
official Flash windows with >=16 nonspecial proposals, K=8/16, gamma=.8. Official
Flash owns all online commits; shadows never alter public cache or output.
Compare ONLY newly produced packet prefix/correction to baseline regular-safe
tokens plus official accepted tokens. Do not credit baseline's preceding safe
commits to the packet. Original final canvas is retrospective reference, not gold.

Require one uniform K with >=24 produced tokens, >=.95 reference-token agreement,
no task below .90, and progress/cost ratio >=1.3 with every task >=1.0. Include
packet construction, capture, decisions and actual selected-KV copy cost. This
fixed-state condition is only a necessary screen: warm-up, bootstrap, stale
proposal acceptance, dirty repair and drain still require actual free generation.
No full generator if this screen fails. Reject if any public cache/output changes
or any real all-layer own/later-label perturbation changes a prefix audit/source.

## Completed cumulative-budget packet: rejected (2026-10-03)

The guarded GPU1 run exited0, with11 server CPU tests, six unchanged official
response hashes, unchanged public KV, and SDPA0. Six actual32-layer label
perturbations preserved all legal audit logits and source K/V exactly. Native
private buffers were actually copied to a separate bank, with that cost included.

| K | Produced tokens | Reference-final matches | Median packet wall time | Fixed-state progress/cost vs Flash |
|---|---:|---:|---:|---:|
|8|23|22|26.94ms|0.524x|
|16|24|24|26.49ms|0.558x|

The reference is an official Flash final canvas, not ground truth; windows share
six dev prompts. No task accuracy or end-to-end speed result follows. Baseline
on these12states produced41regular-safe+54verified tokens; the packet produced
only12verified+12correction tokens at K16. Cost/progress, not source isolation,
failed. Do not expand THIS cumulative-.8 policy to free generation.

CPU follow-up separated token disagreement from cumulative-budget exhaustion:
K16 has8budget-only first failures and4argmax disagreements. Longest argmax-match
prefix plus correction would propose83tokens rather than24 on these same states.
Their reference agreement and selected-KV-copy cost were not measured. This is
evidence for a SEPARATE deterministic-match admission diagnostic, not permission
to relabel the failed predeclared gate as a success or call it lossless.

Additional close work: [SimSD](https://arxiv.org/html/2606.02544) explicitly uses
paired data/MASK copies, temporal masking, position alignment and prefix/rejection
correction. Those are prior mechanisms. Our clean future bank addresses all-layer
reachability at an acceptance cost; this fact alone is not a paper contribution.
[DFlow](https://arxiv.org/html/2609.06498) relays rejected verifier representations
into a trained diffusion drafter, so generic cross-round information reuse is
also prior art. Whether a no-training cache-producing packet is useful remains
an experimental question.

Original source is archived as `results/prefix_cache_20261003/tested_source.tar`,
alongside the unchanged manifest, diagnostic and server test log. CPU policy
analysis is separately named and does not rescore tasks.

## Completed deterministic-match packet (2026-10-03)

The separate K16 screen exited0. Actual selected-KV copy and decisions included:
83produced tokens,79matching Flash's eventual canvas (.9518), packet median
26.49ms. Fixed-state progress/cost was1.923x, with task ratios HE1.835/MBPP1.548/
MATH2.243. Reference agreement was HE1.0/MBPP.895/MATH.95. The predeclared .98
overall/.95per-task reference-agreement gate FAILED; original report is preserved.
Six actual all-layer controls still had zero audit/source-KV perturbation error,
six official outputs remained identical, and SDPA0. No achieved end-to-end speed
or task accuracy follows from these figures. Exact tested source and reports are
in `results/match_cache_20261003/`.

Because the user allows different/better answers, `GENERATION_PLAN.md` declares a
separate six-old-dev task-scored failure diagnosis. It does not relax or relabel
the failed screen. The prototype pays for its own bootstrap, dated proposal
refill, full clean identity repair, real KV transactions and final rendering.
Official execution/final-expression scoring and fixed-work controls decide
whether the local progress signal survives actual generation. No automatic
dataset expansion or success/novelty claim is permitted.

## Actual six-dev generation: not a successful candidate

The separate full-generation diagnosis exited0 with21server CPU tests. Both
paths paid initialization, current generation and rendering; they used identical
BF16 weights, compact head and original FP64 probability reduction. Original
Flash output hashes matched all six old reference outputs, with SDPA0 and all
protected main/scoring/third-party hashes unchanged. Dated cold proposals were
never committed without a current packet audit. Every changed identity entered
the next clean query; captured legal native K/V were actually installed.

| Task (2old dev prompts each) | Flash mean seconds | v7 mean seconds | Latency ratio | Flash correct | v7 correct |
|---|---:|---:|---:|---:|---:|
|HumanEval|0.9889|1.0332|0.957x|1/2|0/2|
|MBPP|0.7033|0.4485|1.568x|0/2|0/2|
|MATH|2.8936|1.4453|2.002x|0/2|0/2|

These are tiny reused development samples, not benchmark accuracy estimates.
Quality did not survive HE122: the prototype ignores k and uses a <=10 string
length test, while Flash's program passes the official tests. This is semantic
error, not an answer-extraction artifact. Other tasks having zero correct answers
for both methods supplies no evidence of quality preservation.

On the first prompt of each task, filling ALL256positions gave latency ratios
HE1.437x, MBPP1.273x, MATH2.059x. They support a local execution opportunity beyond
earlierEOS, but still use different trajectories and are not equivalent-work
mathematical computations or a uniform speed result. HE54 in natural generation
uses39model calls versus Flash16; cold proposals and altered conditioning/ordering
can consume the saved per-cycle work. Current evidence does not isolate one cause.

Do not expand this frozen prototype to a larger dataset or announce superiority.
Keep the cache+verify architecture question open, but separate cache freshness,
legal conditioning and admission before further algorithm claims. Common fused
statistics and the strongest saved optimized Flash runtime still need a unified
comparison for ANY future successful candidate; this diagnosis alone does not
establish superiority over that runtime or other strong baselines.

Immutable outputs, manifest,21test log, exact tested source and separate CPU
analysis live in `results/free_generation_20261003/`. Earlier failed probability
and token-agreement screens remain unchanged. All three GPU jobs have exited and
GPU1 is released. No original LLaDA/v1 reevaluation or main-table edit occurred.

## User-requested128HumanEval comparison: faster, unacceptable quality loss

After the six-dev failure diagnosis, the user explicitly requested128questions
against their existing FOCUS. That authorization permits this separate larger
screen; it does not change the earlier failed gates. The algorithm remained
fixed throughout all128questions (HumanEval, length256, seed51713, offset0).
These are the same reused development IDs as the saved FOCUS evaluation, not
an independent holdout. Only v7 was generated; old baselines were not rerun.

| Method | Official pass@1 | Mean seconds/question | Mean model calls |
|---|---:|---:|---:|
| Original FOCUS |43/128 (33.59375%)|2.775245|79.83594|
| FOCUS with existing IO/infra adapter |43/128 (33.59375%)|2.470826|79.83594|
| Frozen v7 |17/128 (13.28125%)|1.037062|31.55469|

The historical latency ratio is2.3825x against the stronger engineering FOCUS
and2.6761x against original FOCUS. Baseline timings are saved same-ID results,
not contemporaneous paired GPU measurements. Setup, generation, cache repair,
selection, synchronization and rendering are included; model loading and the
separately recorded1.9957second warmup are excluded.

Quality falls20.3125percentage points versus either FOCUS version. The paired
10000sample bootstrap (seed1234) gives[-28.90625,-12.5]pp. Thirty previously
correct questions become wrong, while four previously wrong questions become
correct. Both methods have zero reported truncations; this result cannot be
presented as preserving accuracy or attributed solely to answer formatting.
V7's shorter outputs and different trajectories are recorded, so its latency
ratio also does not establish equal-work execution speed.

The job exited0 with all128official code-execution scores,21passing CPU tests,
all frozen research source hashes intact, unchanged protected main/scoring
sources, and no SDPA fallback. GPU1 was released. Full per-question outputs,
immutable baseline metadata, launch/test logs, exact tested source and an
archive validation report are in `results/quality128_20261003/`.

This fixed prototype fails the user's quality requirement. Preserve the real
speed result and the larger quality failure separately; do not call it a
successful accelerator or automatically expand it further.

## Implementation/admission mechanism audit

Following the user's request to investigate implementation and parameters, a
read-only observer reproduced the exact six saved v7 token/action/text/NFE
trajectories. Six repeated32layer packet controls had zero clean/audit logit and
private source KV error; cache objects, versions and contents were preserved.
The job exited0 with24CPU tests,52separately counted paid shadow calls and SDPA0.

Across23predeclared early/middle windows (368candidate positions), current clean
top1 matches the paid full current-canvas prediction276/368(75%). Holding the
packet order/mask fixed while replacing ONLY external cached KV with fresh paid
values raises that to365/368(99.18%). The audit's top1 changes64/368times, and its
prefix/correction decision changes10/23times. This identifies a cache freshness
effect on the packet's predictions; a full MASK prediction is not gold or the
same conditional distribution as sequential verification. Full recomputation
has already been paid and supplies no executable free acceleration.

The same six real v7 trajectories committed694accepted drafts and163forced audit
corrections. Respectively261(37.61%) and123(75.46%) had probability below.9;
73corrections(44.79%) were even below.5. That is an explicit aggressive admission
policy, rather than ordinary .9confidence decoding. It is not alone a proof of
which task failures it caused. Same-state controlled evidence and all outputs
are archived in `results/mechanism_20261003/`.

A separate fixed16question factorial screen in `REVISION_PLAN.md` tests own-paid
frontier cache/proposal refresh, clean-root+probabilistic-prefix admission, and
both. All original algorithm files and128records remain unchanged. This tests
engineering/algorithm mechanisms; confidence rules and periodic refresh alone
are not claimed as novel contributions or a publishable result.

## Completed16question implementation/admission screen

All three predeclared variants completed with29passing CPU checks, unchanged
frozen research sources, SDPA0 and GPU1 released. The disabled revision controls
also reproduced the original v7's first-question raw tokens, packet actions,
text and model-call count exactly. All full refreshes, cache/proposal maintenance,
selection and rendering are included in the new generation timings.

| Method | Official pass@1 on these16IDs | Mean seconds/question | Mean model calls |
|---|---:|---:|---:|
| Saved original v1 |7/16 (43.75%)|3.028949|historical|
| Saved original FOCUS |2/16 (12.5%)|2.687440|historical|
| Saved FOCUS IO/infra adapter |2/16 (12.5%)|2.396020|historical|
| Saved original v7 |2/16 (12.5%)|1.002777|historical|
| Own-paid frontier refresh only |2/16 (12.5%)|1.328011|30.8750|
| Conservative admission only |2/16 (12.5%)|2.633215|85.9375|
| Both revisions |2/16 (12.5%)|3.077940|86.8125|

The original v1 score was read from the frozen same-ID official code-execution
results; both original record/score file hashes were checked. No old baseline
was regenerated. The saved optimized Flash method got5/16on these same IDs;
its128question mean time must not be substituted for a same16question time.

Conservative admission repairs HumanEval/122 but loses HumanEval/144, leaving
the aggregate at2/16. Cache refresh alone preserves the old v7 correct set
(HumanEval/144 and HumanEval/34) without increasing the total. None improves
aggregate quality, and conservative admission increases model calls to about86.
These changes do not meet the speed-quality goal and are not expanded by default.

This tiny reused development subset is not representative: saved FOCUS got
43/128on the larger screen, while original v1 got53/128and v7 got17/128. Equal
2/16totals do not establish quality preservation; original FOCUS and v7 even
solve different questions. The user now permits small quality losses for large
real speed gains, but the observed v7 deficit is not a small one. Future
candidates need a broader paired quality comparison before acceptance.

The positive mechanism evidence remains separate from these failed remedies:
fresh external KV strongly changes same-state predictions, and the original
admission rule accepts many low-probability tokens. That does not prove that
periodic full refresh or stricter thresholds solve the problem, nor rule out
all implementation defects. Full per-question outputs, scored outcomes, source
snapshot, test/launch logs and CPU analysis are in
`results/revision16_20261003/`, including `cpu_v1_comparison.json`.

## Query-budget redistribution: local improvement, still behind Flash

The next fixed experiment reallocates the64query packet from T16/C16/D16/A16 to
T16/C32/D8/A8. Clean predictions cover twice as many MASK positions, while eight
candidates get private draft/audit replicas. Admission stays.9for clean roots
and.8for cumulative prefix probability; at most16positions commit so mandatory
identity repair fits. One-version geometry and32layer label isolation remain.

Two predeclared variants compare recent-background fill with oldest-cache-version
rotation, after mandatory repairs. No paid full refresh, teacher future state,
training or extra online forward was introduced. The new implementation with
C16/D16 reproduced the saved conservative first two prompts' exact text, raw
tokens, actions and NFE. Real GPU repeat and own/later-label controls had zero
eligible-logit/private-KV error and preserved public cache. Six shadow calls were
separate from online NFE.

| Method on the same16HumanEval IDs | Correct | Mean seconds | Mean model calls |
|---|---:|---:|---:|
| Saved original v1 |7/16|3.028949|historical|
| Saved original FOCUS |2/16|2.687440|historical|
| Saved FOCUS IO/infra adapter |2/16|2.396020|historical|
| Saved optimized Flash |5/16|1.370485|46.8750|
| Previous conservative C16/D16 |2/16|2.633215|85.9375|
| C32/D8, recent background |3/16|1.897730|59.5625|
| C32/D8, oldest-version rotation |4/16|1.696205|52.7500|

Age rotation solves HumanEval/122,34,30,158, but not144,124,83which v1 solves.
All four age-correct questions are also v1-correct. It is1.79x faster than saved
v1 and1.41x faster than engineering FOCUS, but loses3/16versus v1. Against saved
optimized Flash it is23.8% slower and gets one fewer right. Historical same-ID
timings are not contemporaneous paired GPU controls. These reused development
questions do not replace FOCUS's43/128larger result.

This improves the previous conservative implementation, but does not satisfy
the strong-baseline goal or establish publishable novelty. Do not expand solely
because it beats FOCUS on an unusually weak small subset. Accepted draft prefixes
contribute1554/2862packet commits (54.30%) for age rotation;370/828packets accept
no draft. Accounting is not a causal audit-benefit estimate or proof that removing
verification would be free. Neither variant has a past-horizon/count-render MASK
anomaly here; formatting does not explain away the remaining quality gap.

The first attempt stopped before scoring at a compact-readout observer API error.
Its log, successful legacy controls and tested source remain in
`results/budget16_readout_failure_20261003/`. After actual process termination and
a before-repair snapshot, only the new observer/test and explicit retry launcher
were repaired; the algorithm did not change. The repaired run passed37CPU checks,
completed exit0 and retained all frozen/protected source hashes. The own GPU
process exited; no baseline was regenerated. Complete records, controls, logs,
tested source and CPU validation are in `results/budget16_20261003/`. A later small
foreign GPU allocation is user-authorized and was not stopped/reset.

## FreeDave source review

`FREEDAVE_IMPLEMENTATION_REVIEW.md` records the reviewed official commit and
concrete differences in selected-state prediction handoff, branch coverage,
private cache persistence, numerical reference and physical model-call counting.
Eight CPU primitive checks are in `results/freedave_review_20261003.json`.
They do not constitute a new decoder, FreeDave GPU result or quality guarantee.

## Whole-state handoff control: negative

The complete-state paired control is archived in `results/handoff_20261003/`.
Packing two prepared calls is1.67–1.75x faster than their serial pair, but only
3/20proposed updates match the fresh complete update and only2/20also preserve
both packed/sequential actions. The resulting optimistic gross opportunity is
.92–1.03x before online overhead. Five numerical action disagreements also
prevent equivalence claims. The six original trajectories remain unchanged;
there is no new quality or online speed result. This control is stopped.

## User-requested fixed GSM8K screen

The user accepts speed near Flash with an independently developed substantive
mechanism and limited quality loss, and explicitly asks to try the just-tested
configuration on more GSM8K questions. `gsm_screen.py` therefore evaluates only
the existing coverage-age T16/C32/D8/A8 configuration, at256tokens, on the frozen
first128GSM8K development IDs (seed51713). It does not revive the handoff control
or introduce a new algorithm/sweep. Saved LLaDA/v1/FOCUS outputs, scores and
timings are reused and their hashes checked. Each output is persisted before
the reference is accessed; final-expression-v3 is primary and the unchanged
lm-eval filters are additionally retained. No holdout or novelty claim follows
from this screen. Historical timings and authorized foreign GPU allocation
limit the timing comparison; no foreign process is stopped. The launch uses
the exclusive research lock, terminal check, before snapshot,45CPU checks and
frozen research Python sources for the duration of the run.

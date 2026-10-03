# FIREBREAK: isolated verification experiment

Status: experimental proposal from the user's supplied design. It changes
verification contexts. There is no native-decoder equivalence, task-accuracy,
speedup or established novelty claim.

All new implementation, tests and launch manifests belong in this directory.
Server records go in a dedicated `firebreak_preflight_*` run directory. Preserve
the previous FOCUS-v4 records, including the failed Flash-GSM8K cell at94/128.
Do not run LLaDA/v1 generation again. Research uses physical GPU1 only.

## First experiment

Run a private, read-only same-state diagnostic before implementing a generator:
two previously used development prompts each from HumanEval, MBPP and MATH,
at most two eligible Flash verification windows per prompt, length256. These
interface/mechanism checks are not an accuracy screen; subsequent selection
requires128 examples per task and a common configuration.

Candidates and the base cache come from the same paid ordinary Flash call.
Capture the base before any pending draft is injected. Never use gold, tests,
solutions, teacher future tokens or paid full-teacher KV as online information.
Snapshot copying and metadata construction are charged diagnostic costs.

The verifier has three query families: draft D, isolated MASK I, observer MASK O.
For every query/original position, choose exactly one base-or-draft KV version:

* D_i: own-group draft j <= i; base version everywhere else.
* I_i: own-group draft j < i; base version everywhere else.
* O_i: own-group draft j < i and other-group drafts; base otherwise.

Only D rows become private keys. I/O rows are never keys for any other row.
All32 layers follow the same provenance rule and original absolute RoPE.
The public cache is immutable. Private KV must never be published, even when
its token identity is accepted. In this first diagnostic nothing is committed.

## 2026-10-03 implementation audit correction

The initial12-window diagnostic omitted the legal tracked query refresh: Flash
has just accepted some normal-proposal tokens, while their cached KV still
represents the earlier MASK input. Therefore its12/12/8/1 shadow acceptance
counts do NOT alone establish the proposed isolation mechanism's feasibility.
All original records remain preserved. The audited operator additionally packs
the EXACT known legal tracked tokens/positions supplied by the ordinary verifier.
These shared rows read base or refreshed legal shared KV, never speculative
draft KV. All D/I/O queries read the refreshed shared version exactly once;
only D and shared rows are private keys. Its32-layer compute and readout costs
are explicitly measured. No paid teacher KV or future output is substituted.
Shared-refresh isolation still changes Flash's graph, so not native equivalence.

Use contiguous chunks in candidate confidence order, R=1,2,4, without regrouping
or searching after seeing failures. Compare matched single-chain and grouped
isolation, grouped isolation plus read-only cross veto, and the actual official
Flash result. The matched single-chain uses the same per-token gate as grouping;
also report cumulative-probability prefixes to expose budget-policy differences.

Initial fixed gates: draft/isolated argmax match, isolated candidate p>=.80;
cross additionally requires candidate argmax match and full-distribution
Jensen-Shannon divergence <=.05 nats. Thresholds are diagnostic settings, not
selected optima or correctness certificates. For each group take the prefix of
the combined pass flags, then union those prefixes. Apply the cross veto BEFORE
the local prefix scan, so rejecting an early dependency cannot leave its tail
accepted. Observers can veto only, never rescue an isolated failure.

## Mandatory checks

CPU: per-position version exclusivity, multilayer label reachability, no own or
later-label path into I, no own-label path into O, no observer-key path, prefix
dependency closure including an early cross veto, ragged groups and special
tokens, stable masked softmax, Jensen-Shannon symmetry and exact query mapping.

GPU: compare the independently written streaming attention with a small dense
FP32 reference; require finite outputs. Perturb a single legal draft label and
confirm zero logit change on forbidden dependencies across the real32-layer
model. Check original public KV bit patterns before/after private calls. Clean
and observed official output, actual commits and model calls must match.

Measure cold/setup and stable private pass costs separately. Include both readout
distributions, JSD, snapshot copying and proposal call costs. Extra accepted
flags are not saved NFE. A per-window cost ratio is not end-to-end speed.
Teacher final commits can be a diagnostic agreement reference, never gold or
online candidate information. Do not call agreement task accuracy.

Stop before a full executor if isolation destroys acceptance, cross veto removes
the gain, provenance checks fail, or added work outweighs extra legal progress.
No random partition/threshold scan and no automatic full-task expansion.

## Baseline boundary failure

The old Flash-GSM8K cell failed because the pinned generator projects/slices32
normal rows while an EOS-shortened masked window has26 positions. The same
source uses that26-position mask to index the32 predictions. A process-local,
explicitly recorded baseline adapter restricts that consumer to the actual
masked-window extent. Third-party source and old records remain unchanged.
The diagnostic compares clean/observed runs of this identical adapter. This is
a boundary correction, not FIREBREAK's contribution or an acceleration claim.

## Attribution and novelty boundary

The attention implementation and provenance/commit rules are written in this
branch; existing pinned model weights and ordinary Flash generation remain
credited dependencies. Grouping, verification, KV override and early-rejection
motivation are prior art. Closest checks:

* [Flash-dLLM](https://arxiv.org/html/2609.26796v1)
* [DiCo](https://arxiv.org/html/2602.23792v1)
* [PUNT](https://arxiv.org/html/2510.21961v1)
* [CoRe](https://arxiv.org/abs/2602.04096)
* [COVER](https://arxiv.org/html/2602.06161v1)
* [BRISK-DLM](https://arxiv.org/html/2609.33390v1)

The possible contribution is the actual provenance-closed non-global-prefix
verification operator with a non-propagating cross veto, if it demonstrates a
useful quality-cost advantage. A renamed or split Flash chain is insufficient.

## Second implementation audit: background and projection

The shared-context correction leaves candidate MASK KV from the last normal
forward. In this audit, additionally recompute a draft-free MASK bank alongside
legal tracked rows, replacing each candidate position with exactly one draft
or clean MASK version. Neither legal nor clean background reads draft labels.
This is a paid background-update policy change, not a free equivalent fix.

Hold candidates, groups, gamma=.8 and eta=.05 fixed. Compare shared-only and
clean-background chain/cross pairs at the same teacher state. Separately use
the pinned original fused QKV projection in private buffers to test the known
BF16-rounding-before-RoPE discrepancy. Its full cost, including otherwise unused
private-row projections, is counted. The graph and acceptance rule stay fixed.
Dense attention and label-perturbation controls must still pass. No new tokens
are committed, no accuracy claim is made, and no 128-example expansion follows
automatically. A changed accept set alone does not establish a speed benefit.

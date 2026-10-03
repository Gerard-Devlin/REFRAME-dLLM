# FOCUS-v6 deferred atomic audit

Hypothesis: put one provisional epoch into the next cache/proposal pass while
auditing it, and commit the matching canvas/cache together only when the entire
epoch passes. On failure discard the tentative main lane, restore the accepted
canvas/cache, and pay a recovery call. All setup, drain and recovery costs count.
This is an approximate inference scheme, not native-decoder equivalence.

The first implementation is a state machine and a mechanism probe, not a full
decoder. Before implementing fusion, test whether short epochs can repay their
abort costs on real fixed Flash states. Use six already-used dev prompts,
two earliest eligible windows per prompt, K=2/4/8, gamma=.8, no threshold sweep.
Each candidate has a private draft representation conditioned only on its own
identity plus clean background. Its independent MASK audit reads peer drafts
but cannot acquire its own label through any layer. A future tentative main
lane may read all drafts, but never feeds audit/source keys before commit.

The naive alternative (own-label edge removed but mutually mixing draft rows)
already fails a two-layer reachability check. Shared contaminated main K/V
cannot repair that. Atomic commit handles rollback; it does not certify that a
mutually consistent candidate batch is objectively correct.

## Predeclared gate

Report all K configurations, by task and by prompt. Retrospective agreement is
against the saved official teacher final canvas, never gold and never input.
It is not task accuracy. To advance, at least one *uniform* K must have:

- >=12 accepted epoch tokens on this small diagnostic;
- accepted-token teacher-final agreement >=.98 and no task below .95;
- projected effective progress/cost ratio >=1.2 under the optimistic assumption
  that a fused call costs one regular call, plus one regular recovery per abort.
  No task may have an optimistic ratio below 1.0.

That timing assumption omits extra audit queries/cache transactions and is only
a necessary screening condition. It cannot be reported as achieved speedup.
Failure of the optimistic condition stops fusion implementation. A pass only
authorizes a bounded fused-kernel experiment followed by actual free generation
and official scoring. Baseline output hashes and public K/V must stay unchanged.

Nearby work includes [FreeDave](https://github.com/cychomatica/FreeDave), whose
pipeline already verifies previous drafts while generating new ones. Therefore
pipeline + rollback alone is not a novelty claim. The prospective difference
to evaluate is all-layer independent source/audit flow combined with an atomic,
versioned *approximate bidirectional cache* update. Whether that difference is
useful or publishable remains unproved; renaming the pipeline is not a result.

## Completed mechanism gate: stopped (2026-10-03)

The guarded GPU1 run completed with exit0 on all six prompts and 12 windows.
All official Flash response hashes matched the previous reference, public cache
contents were unchanged, and SDPA calls were0. Twelve real 32-layer private
label perturbations produced exactly zero own-audit logit changes, while peer
audit logits changed. The intended source isolation is implemented correctly
on these states; the algorithmic cost/progress premise failed:

| Epoch size | All-pass windows | Accepted epoch tokens | Optimistic effective rate vs Flash |
|---|---:|---:|---:|
|2|3/12|6|0.474x|
|4|1/12|4|0.414x|
|8|1/12|8|0.451x|

Accepted tokens match the reference final canvas in this diagnostic. Across K
configurations these contain only16distinct position/value tuples, and reference
output is not ground-truth accuracy. No free-generation quality claim follows.

CPU analysis also replaced the actual audit policy with a retrospective rule
that accepts an epoch only if every token matches the teacher final canvas.
Even that rule gives only0.835x/0.968x/0.656x in the same optimistic cost model.
This is not a universal upper bound: it is evidence that whole-epoch aborts are
too costly here even apart from imperfect audits. On the separate138-cycle
Flash trace, guessing zero accepted tokens is correct43times; guessing the
previous count is correct21/132adjacent pairs. Blind branch guessing therefore
does not supply a replacement for the failed pipeline on these traces.

Do not implement a fused runtime or expand generation for this atomic design.
Keep the failed source, manifest and diagnostics. The ledger is a CPU reference
over immutable cache-version handles; it does not implement tensor cache
promotion. The research goal remains unachieved. Results:
`results/atomic_audit_20261003/diagnostic.json`, SHA256
`e88c158e3a284c208ff25a9424c1e80931394ae2b37f6c35064fd72e7ce23e63`.
The exact GPU-tested source is preserved in `tested_source.tar` beside the
manifest (11 server CPU checks). After the run terminated, the reference ledger
was hardened to reject mutable storage pretending to be a version handle;
13 local CPU checks pass. The GPU audit/acceptance policy is unchanged, and no
second GPU run was performed or required for that reference-only correction.

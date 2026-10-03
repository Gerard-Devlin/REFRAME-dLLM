# Whole-state handoff mechanism control

Previous goal turn made progress: completed and preserved the query-budget
experiment and official scores, then reviewed the pinned FreeDave implementation.
Current guard confirms that the prior child PID is absent, terminal exit0, its
source set matches, GPU1 is empty and the protected main evaluation is terminal.
The goal is still unmet; budget-age4/16 at1.696s trails saved Flash5/16 at1.370s.

Falsifiable question: does replacing partial conditional audit replicas with an
explicit complete proposed successor state give useful, state-bound predictions
at a cost that could repay the second branch? This is a FreeDave-inspired control,
not a new decoding algorithm or novelty claim, and not an online deployment.

Use six already-used HumanEval development prompts from the completed budget16
run, and predetermined packet indices0,1,4,8. The original coverage_age decoder
still owns all generation and must reproduce its saved raw tokens/text/actions/
NFE. The observer reads its existing proposals and state, never gold or future
teacher outputs. There is no new accuracy or online speed claim in this probe.

Each branch recomputes the32-position MASK window plus32 committed background
positions. All physical positions outside these64 are read from the same frozen
public bank. Branch0 is the current state. Branch1 has the proposed first update
inserted. The update uses the existing dated predictions, threshold.9, all eligible
positions, or one highest-confidence fallback. The fresh branch0 prediction
defines the reference update using the same rule. Branch1 can supply a successor
only when the full position/token update matches. A failed branch is never
promoted or spliced into the original decoder.

Important kernel constraint: pinned `_flash_verify_attention_fwd` only reads
the64 private rows of its own block. An80/96-row dense private window would be
invalid without a kernel change. We instead pack two separate64-row blocks with
a128x64 stacked local mask and identical external-cache spans. This recomputes
the background in both branches and costs128 query rows, not64/80/96. The compact
contiguous head span projects96 rows versus32+32 for the two separate calls;
this extra work is retained in measured cost. No third-party/kernel changes.

Compare the packed two branches against two independent64-row calls on the same
inputs, frozen cache and positions. Record eligible-logit error, exact sampler
actions, confidence/top1, and private KV error. Mutating a private proposed label
must leave the first branch untouched at every layer; CPU reachability and a
real GPU control establish that limited invariant. Public-bank content/version
must remain unchanged. Sequential comparison is the same approximate frozen-bank
operator, not a full original LLaDA oracle or mathematical BF16 certificate.

Measure stable single, two-serial and packed forward wall time on only the first
eligible state of the first two prompts, after compilation/warmup. Count all
physical model calls independently. All measurements are private diagnostic
costs; online request setup, candidate maintenance and state commit costs remain
unknown. The gross opportunity is `(1+exact_update_fraction)*T_single/T_packed`,
also requiring both branch sampler actions to agree. It is an optimistic mechanism
screen, not end-to-end speedup. Sparse sample weighting is not population evidence.

Predeclared continuation gate: at least16 eligible windows; at least half have
full proposed-update equality; all compared packed/sequential actions agree;
gross opportunity exceeds1.15 at both timed states. Failing any gate retains the
negative result and does not launch an online decoder or expand evaluation.
Passing would only justify a state-bound reference implementation and fixed
broader screen, not a quality/losslessness/novelty claim.

Only physicalGPU1, fixed LLaDA8B revision/BF16/fused Flash, existing cache/environment.
Exclusive research lock, previous real PID terminal, hash checks, before snapshot
and CPU checks precede deployment/dispatch. Research Python files remain frozen
until the actual child terminates. No baseline regeneration, main/scorer edits,
training, foreign-process termination, automatic restart or server git pull.

## Executed outcome: stop this control

The six saved trajectories were reproduced exactly (tokens, text, actions and
NFE), and all five eligible label-mutation controls preserved branch0 logits
and private KV with zero error. The20observed windows gave only3whole proposed
update matches;15had packed/sequential action agreement, and only2satisfied
both requirements. Five action disagreements preclude a strict-equivalence
claim. Their precise numerical cause was not isolated.

Two prepared-forward comparisons show1.668/1.747x for serial-pair versus packed,
but the10% usable update fraction leaves only.920/1.031x gross opportunity.
These exclude online setup/maintenance/commit costs. The predeclared gate fails;
no online decoder or larger handoff experiment is launched. This diagnoses our
dated-proposal control, not official FreeDave performance. All237generation
calls and169private calls were counted separately. Results are retained in
`results/handoff_20261003/`; terminal exit0 and source hashes were verified.

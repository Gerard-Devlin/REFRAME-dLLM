# FreeDave implementation review, 2026-10-03

Reviewed the clean official checkout at commit
`806883355c5db3de22310fc8abad039a6e47be1f`, not a recreated description.
The source fingerprints and eight CPU primitive checks are recorded in
`results/freedave_review_20261003.json`. This review did not run a Transformer,
regenerate old baselines, or measure FreeDave latency/accuracy on the GPU.

## What its implementation actually does

1. **Predict the next transition while verifying the current proposal.**
   `token_transfer` builds multiple complete candidate states from an already
   computed prediction, and the model evaluates those states. After verification,
   both the selected state and its own computed predictions are selected using
   the same branch index. Those predictions form the next proposals; an unrelated
   rejected branch's predictions are not used. In the ordinary full-attention
   path, the committed state is the selected draft, not the extra target update.
   See [state construction](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L1734)
   and [selected-branch handoff](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L1997).

2. **Whole branch coverage, shared external context, isolated branch interiors.**
   Tree attention repeats the physical position IDs for complete private windows.
   Every branch reads the common cached prefix/suffix and only its own window.
   The dual-cache path recomputes the current window for each branch; the
   prefix-cache full-attention path includes the remaining suffix in each branch.
   It is not the same conditional program as our T/C/D/A packet. Sharing cached
   background does not make the private branch queries free. See
   [tree geometry](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/attn_utils.py#L102)
   and [dual-cache branch forward](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L1789).

3. **Private forward KV is composed, not silently written into the frozen base.**
   `DynamicDualCache.update` constructs prefix/current-private/suffix KV while
   preserving the base. On the LLaDA model path, `store_kv` is discarded and actual
   persistence is controlled inside `attention`; copying the caller's parameter
   alone would not establish write isolation. Different model adapters have
   different behavior. See
   [dual cache](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/cache_utils.py#L60),
   [LLaDA cache update](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/modeling/llada/modeling_llada.py#L726)
   and [discarded parameter](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/modeling/llada/modeling_llada.py#L1641).

4. **Optimized and reference verification are distinct.**
   The ordinary vectorized verifier builds a branch-agreement matrix and can
   advance to the largest agreeing branch. The debug reference requires
   consecutive agreement. Full-attention verification uses subset agreement,
   whereas the default primitive uses exact agreement. Subset agreement alone
   is not equality of a native confidence-threshold transition. The earlier
   `verification_rule_audit.json` already records that limitation; it is not a
   refutation of the paper's static-sampling theorem. See
   [vectorized rule](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L787)
   and [full-attention invocation](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L1977).

5. **Numerical backend and NFE reporting need separate controls.**
   The pathwise debug reference can force the SDPA math backend. Its sequential
   debug helper subtracts physical branch calls from the logical monitor count.
   Therefore neither its numerical reference nor its reported logical NFE can
   directly be substituted for our BF16/fused-Flash physical work. Count real
   forwards with a model hook, query rows, cache copies, setup and full request
   time. Our budget16 run already counted physical calls independently. See
   [reference backend](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L96)
   and [logical monitor adjustment](https://github.com/cychomatica/FreeDave/blob/806883355c5db3de22310fc8abad039a6e47be1f/generation/core.py#L970).

## Consequence for our current implementation

Our clean lane predicts the pre-commit canvas. A packet can then commit an audited
draft prefix plus independent clean roots. Its remaining clean predictions were
not computed on that combined post-commit state. Unseen windows additionally use
dated bootstrap predictions. This is a declared speculative approximation, not
evidence of an array-index bug. Mandatory identity repair only fixes changed
token identities on the next call; it does not make all deep context KV fresh.

FreeDave's selected-branch prediction handoff highlights a real missing invariant:
**a proposal needs a known source state, source cache version and transition rule.**
Validation on another conditional view does not magically rebind that source.
Likewise, all-layer label isolation proves rejected/private label exclusion,
not equality to the original bidirectional model. The eight CPU checks verify
tree isolation, cache composition, additive-mask semantics and a synthetic
selected-branch handoff, not task quality or BF16 Transformer equivalence.

The already completed fixed-budget experiment supports addressing useful fresh
coverage: T16/C32/D8/A8 with age rotation improves the prior conservative policy
from2/16 at2.633s to4/16 at1.696s. But the same-ID optimized Flash result remains
5/16 at1.370s, and v1 is7/16 at3.029s. Baseline times are historical. These reused
16 questions cannot establish small quality loss or broader superiority.

## What to carry into the next design, and what not to claim

Carry the explicit state/prediction/cache binding, real branch-isolation reference,
and physical-work accounting into future implementations. A useful comparison
would first bind two complete candidate windows to their own prediction outputs,
then check sequential-versus-packed actions and cache isolation. It must include
the current-window queries, identity repairs and any boundary refresh in cost.
For example, two32-position branches already use64 private queries; adding our
16 background-repair rows would use80, not64. Removing those repairs needs an
explicit cache policy, not free credit.

This is a design/control requirement, not an implemented new GPU method. Do not
copy a FreeDave pipeline and rename it as novelty. Do not expand the current
behind-Flash candidate to128 questions merely because a small FOCUS subset is
weak. A new candidate needs a falsifiable quality/cost mechanism and a fixed
screen before independent strong-baseline validation. Static-sampling theory,
subset agreement, CPU attention equality and benchmark accuracy are distinct
claims; none substitutes for the others.

# Bounded task-scored generation test

The deterministic-match screen finished with83new tokens,79matching Flash's
eventual canvas, and1.923x fixed-state progress/cost. Its predeclared reference
agreement gate FAILED (.952<.98, MBPP.895<.95). Preserve that failure unchanged.
Reference agreement is not objective task accuracy. The user explicitly permits
different/better answers and asks for actual ACC. Therefore this is a separate
six-old-dev-prompt task-scored failure diagnosis, not an expansion licensed by a
passed token gate, a relaxed gate, or independent validation.

Fix K16, tested64row legal-context packet, deterministic longest match prefix
plus one first-mismatch correction. One full native bootstrap owns the initial
KV and hidden state. It may commit at most16 .9-confidence tokens (argmax
fallback) from its own already paid current prediction. Newly arriving positions
use dated bootstrap hidden projections as proposals, NEVER direct commitments.
Those projections, selection, copies, mandatory identity repair and setup count
in end-to-end latency. Packet clean outputs refill current proposals. All changed
identities, including correction, enter the next clean tracked query. No rejected
draft or audit KV is promoted. MASK cannot be committed; EOS is allowed.

Use the original discovered-EOS horizon (first group of discovered EOS -> max
position+1), finish all holes before it. Use identical official rendering for
both paths and separately save raw full canvas, firstEOS, committed length and
truncation. This program still changes conditioning/cache/selection and is NOT
native-equivalent or distribution-preserving.

Compare optimized official Flash-Verify with common compact readout on the same
six prompts and process. Do not rerun native LLaDA or v1. One unreported warmup
per method uses an old dev prompt; all six reported runs include allocation,
bootstrap, proposal refill, commit and rendering. Alternate execution order.
Also compare fixed256position work on the first prompt of each task: both stop
only after the whole canvas is filled, with original token/EOS logic otherwise.
This is a cost control, not another accuracy metric or improved stopping method.

Frozen HE/MBPP execution tests and final-expression-v3/MathVerify0.1.0 assess
both complete outputs, outside generation timing. Gold/tests never reach model.
Record model calls/NFE, rows, packet acceptance, cold proposal count, EOS/length,
warmup separately, wall time and all outputs. Check every cache transaction,
source hashes and SDPA0. Six samples cannot certify quality preservation.
No automatic parameter search or dataset expansion. A follow-up screen requires
an explicit conclusion from actual scores and cost controls; failure is retained.

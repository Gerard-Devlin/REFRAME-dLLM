# User-reported RTX 5090 pilot observations — 2026-09-13

Source: the user's pasted complete training/evaluation logs, not a direct
server inspection in this turn. One seed (1234), one RTX 5090, 10,894,721
parameters, global batch 48. Both models completed their requested budgets.
Full evaluation files are needed for per-example comparisons and provenance.

| Budget | Model | 1-step path bits/symbol | 4-step | 16-step | Training/status seconds |
| --- | --- | ---: | ---: | ---: | ---: |
| 200 updates | identity | 4.685074 | 4.685052 | 4.685097 | 27.06 |
| 200 updates | relation2 | 4.277337 | 4.277519 | 4.277557 | 27.24 |
| 2000 updates | identity | 4.678051 | 4.679610 | 4.653378 | 265.30 |
| 2000 updates | relation2 | 4.243971 | 4.242199 | 4.234908 | 271.36 |

The 200-update paths were effectively flat. At 2000 updates the identity
1-to-16-step improvement is 0.024672 bits/symbol and relation2 improves by
0.009063. Both models' 2-step losses are slightly worse than their 1-step
losses, so monotonic improvement with steps has not been established.

Relation2's same-4-step loss is about 9.35% lower. Its 1-step loss already
beats this identity model's 16-step loss. This is evidence of a representation
advantage at this training budget and metric, but not a clean demonstration
that the method removes necessary denoising steps from a strong baseline.
The baseline barely benefits from extra steps; free-running language/task
quality, repeated seeds and well-learned dependencies are still unverified.

The earlier independent-frequency scores used 1024 validation blocks, whereas
the neural evaluator defaults to 256. Do not subtract those old scores as a
matched comparison. The new `audit` phase recomputes the frequency control
on exactly the same validation blocks as its neural checks.

Next action: pause further training while auditing the existing checkpoints.
Measure shuffled-prefix/history sensitivity under fixed reveal paths,
shuffled-visible-context sensitivity under training-like random masks, and
BF16 versus FP32 scores. No automatic promotion, larger model training,
or same-quality speedup claim is justified by these two logs alone.

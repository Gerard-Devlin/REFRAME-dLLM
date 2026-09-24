"""Fixed development ablations. Settings are hypotheses, not tuned results."""

from .budget import CommitGuard


# Every guarded setting uses the same margin/EOS rule. Stability is isolated
# by comparing `stable` against `guarded`; `strict` explores a smaller update.
PRESETS = {
    "original": (0.0, None),
    "guarded": (0.2, CommitGuard(min_confidence=0.85, max_extra=2, protect_stop=True)),
    "strict": (0.2, CommitGuard(min_confidence=0.90, max_extra=1, protect_stop=True)),
    "stable": (0.2, CommitGuard(min_confidence=0.85, max_extra=2,
                                min_observations=2, protect_stop=True)),
}

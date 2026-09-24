"""Sufficient joint-mode certificate from full-vocabulary top-two marginals.

This is a conditional probability statement, not a guarantee about a neural
decoder's next forward or downstream task accuracy.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class CommitGuard:
    """Empirical guards for extra commits only; not a losslessness guarantee."""

    min_confidence: float = 0.0
    max_extra: int | None = None
    min_observations: int = 1
    protect_stop: bool = False

    def __post_init__(self):
        if not 0 <= self.min_confidence <= 1:
            raise ValueError("min_confidence must be in [0,1]")
        if self.max_extra is not None and self.max_extra < 0:
            raise ValueError("max_extra must be nonnegative or None")
        if self.min_observations < 1:
            raise ValueError("min_observations must be positive")


class PredictionHistory:
    """Consecutive confident argmax observations within one active sub-block.

    Only predictions from ordinary forwards are recorded. Construct a new
    history on entering each sub-block; never carry it across requests/blocks.
    """

    def __init__(self):
        self.tokens = None
        self.counts = None

    def update(self, tokens, confidence, eligible, min_confidence):
        import torch

        if self.tokens is None:
            counts = torch.ones_like(tokens)
        else:
            counts = torch.where(tokens == self.tokens, self.counts + 1, 1)
        self.counts = torch.where(eligible & (confidence.float() >= min_confidence), counts, 0)
        self.tokens = tokens.clone()
        return self.counts


@dataclass(frozen=True)
class BudgetStep:
    baseline: object
    proposed: object
    plusplus: object
    top1: object
    top2: object


def _prefix_mask(confidence, runner_up, eligible, margin, plusplus=False):
    """Longest confidence-ordered prefix satisfying a monotone certificate."""
    import torch

    selected = torch.zeros_like(eligible)
    for row in range(eligible.shape[0]):
        positions = eligible[row].nonzero().flatten()
        if positions.numel() == 0:
            continue
        order = positions[torch.argsort(confidence[row, positions], descending=True, stable=True)]
        costs = torch.cumsum(1.0 - confidence[row, order], dim=0)
        competitor = (1.0 - confidence[row, order]) if plusplus else runner_up[row, order]
        bounds = torch.cummax(competitor, dim=0).values
        count = int((costs + bounds < 1.0 - margin).sum())
        selected[row, order[:count]] = True
    return selected


def decide(probabilities, sampled, eligible, threshold, margin=0.0, include_top1_bound=False,
           *, guard=None, observations=None, stop_token=None):
    """Return native selection and its certified extensions, never reducing it.

    `probabilities` must be normalized over the *entire* vocabulary, and the
    decoder must be greedy (sampled == argmax). All masks have shape [B, K].
    Filled positions may be present but are excluded by `eligible`.
    """
    import torch

    if probabilities.ndim != 3 or eligible.shape != sampled.shape or probabilities.shape[:2] != eligible.shape:
        raise ValueError("Expected probabilities [B,K,V] and sampled/eligible [B,K]")
    if probabilities.shape[-1] < 2 or not (0.0 <= margin < 1.0):
        raise ValueError("At least two vocabulary entries and margin in [0,1) required")
    if bool((sampled[eligible] != probabilities.argmax(-1)[eligible]).any()):
        raise ValueError("Certificate is only defined for greedy argmax commits")
    native_confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1)
    confidence = native_confidence.float()
    # Cast only the selected values; do not allocate an FP32 vocabulary tensor.
    second = torch.topk(probabilities, k=2, dim=-1).values[..., 1].float()
    # Match the official scalar comparison in the original probability dtype.
    # Converting to FP32 first can change a threshold decision under BF16.
    masked_confidence = torch.where(eligible, native_confidence, -torch.inf)
    baseline = (masked_confidence > threshold)
    forced = masked_confidence.argmax(-1)
    rows = eligible.any(-1).nonzero().flatten()
    baseline[rows, forced[rows]] = True
    baseline &= eligible
    proposed = _prefix_mask(confidence, second, eligible, margin)
    plusplus = (_prefix_mask(confidence, second, eligible, margin, plusplus=True)
                if include_top1_bound else None)
    # A threshold policy can approve positions outside the joint certificate.
    # In that case, retain the original set; no unsupported union is formed.
    for row in range(eligible.shape[0]):
        if not bool((proposed[row] | ~baseline[row]).all()) or int(proposed[row].sum()) <= int(baseline[row].sum()):
            proposed[row] = baseline[row]
    if guard is not None:
        extras = proposed & ~baseline & (confidence >= guard.min_confidence)
        if guard.min_observations > 1:
            if observations is None or observations.shape != sampled.shape:
                raise ValueError("Stability guard requires matching historical observation counts")
            extras &= observations >= guard.min_observations
        if guard.protect_stop:
            if stop_token is None:
                raise ValueError("protect_stop requires stop_token")
            extras &= sampled != stop_token
        if guard.max_extra is not None:
            order = torch.argsort(torch.where(extras, confidence, -torch.inf),
                                  dim=-1, descending=True, stable=True)
            keep = torch.zeros_like(extras)
            keep.scatter_(1, order[:, :guard.max_extra], True)
            extras &= keep
        # Removing extras produces a subset of the certified proposal. Native
        # commits remain intact, including forced low-confidence tokens/EOS.
        proposed = baseline | extras
    return BudgetStep(baseline, proposed, plusplus, confidence, second)

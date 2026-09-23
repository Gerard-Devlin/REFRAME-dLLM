"""Sufficient joint-mode certificate from full-vocabulary top-two marginals.

This is a conditional probability statement, not a guarantee about a neural
decoder's next forward or downstream task accuracy.
"""

from dataclasses import dataclass


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


def decide(probabilities, sampled, eligible, threshold, margin=0.0, include_top1_bound=False):
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
    confidence = probabilities.gather(-1, sampled.unsqueeze(-1)).squeeze(-1).float()
    second = torch.topk(probabilities.float(), k=2, dim=-1).values[..., 1]
    masked_confidence = torch.where(eligible, confidence, -torch.inf)
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
    return BudgetStep(baseline, proposed, plusplus, confidence, second)

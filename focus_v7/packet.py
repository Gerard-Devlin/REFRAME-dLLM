"""One-version-per-position verifier and cache-promotion reference.

Each audit sees earlier draft identities, its own MASK and clean future MASKs.
Draft source i sees only identities <=i. Clean rows never read any proposal.
This is an approximate conditional program, not the native bidirectional model.
"""
from dataclasses import dataclass
import math

import torch

from focus_v6.audit import Layout, build_call as atomic_call


def prefix_mask(layout):
    k, t, w = layout.candidates, layout.tracked, layout.width
    result = torch.zeros((w, w), dtype=torch.bool)
    result[:t+k, :t+k] = True
    i, j = torch.arange(k)[:, None], torch.arange(k)[None, :]
    result[layout.draft, :t] = True
    result[layout.draft, layout.draft] = j <= i
    result[layout.draft, layout.clean] = j > i
    result[layout.audit, :t] = True
    result[layout.audit, layout.draft] = j < i
    result[layout.audit, layout.clean] = j > i
    result[layout.audit, layout.audit] = torch.eye(k, dtype=torch.bool)
    return result


def build_call(state, k):
    query, positions, lengths, layout, candidates, drafts = atomic_call(state, k)
    positions[-1] = prefix_mask(layout).to(query.device)
    return query, positions, lengths, layout, candidates, drafts


@dataclass(frozen=True)
class Decision:
    accepted: int
    tokens: tuple[int, ...]
    correction: int | None

    @property
    def progress(self):
        return len(self.tokens)


def decide(probabilities, top1, drafts, *, gamma=.8, forbidden=()):
    """Verified prefix plus ONE first-rejection correction, never rejected tail.

    The correction is an ordinary argmax in the prefix-conditioned MASK view;
    it changes Flash's sampling policy. No distributional equivalence claim.
    Special-token policy must explicitly permit EOS in a future real decoder.
    """
    if not 0 < gamma <= 1 or not len(probabilities) == len(top1) == len(drafts) or not drafts:
        raise ValueError("invalid decision geometry")
    if any(not math.isfinite(p) or not 0 <= p <= 1 for p in probabilities):
        raise ValueError("invalid candidate probability")
    banned = set(forbidden)
    cumulative, accepted = 1., 0
    for p, predicted, proposed in zip(probabilities, top1, drafts):
        cumulative *= p
        if proposed in banned or predicted != proposed or cumulative < gamma:
            break
        accepted += 1
    correction = None
    tokens = tuple(map(int, drafts[:accepted]))
    if accepted < len(drafts) and top1[accepted] not in banned:
        correction = int(top1[accepted])
        tokens += (correction,)
    return Decision(accepted, tokens, correction)


def promotion_plan(layout, candidate_positions, tracked_positions, decision):
    """Choose conditional KV versions; a correction must remain dirty.

    A D_i version with i<p is legal even after later proposal failure: it never
    read later labels. Clean remaining MASK versions are approximate background.
    Corrected position has no matching input-identity KV, so cannot be promoted.
    """
    k, t, p = layout.candidates, layout.tracked, decision.accepted
    candidates, tracked = tuple(map(int, candidate_positions)), tuple(map(int, tracked_positions))
    if len(candidates) != k or len(tracked) != t or not 0 <= p <= k:
        raise ValueError("invalid promotion geometry")
    if len(set(candidates+tracked)) != k+t:
        raise ValueError("duplicate physical position")
    if len(decision.tokens) != p + int(decision.correction is not None):
        raise ValueError("invalid decision")
    dirty = (candidates[p],) if decision.correction is not None else ()
    rows = list(range(t))
    destinations = list(tracked)
    for i, position in enumerate(candidates):
        if position in dirty:
            continue
        rows.append(layout.draft.start+i if i < p else layout.clean.start+i)
        destinations.append(position)
    return tuple(rows), tuple(destinations), dirty


def require_identity_repair(dirty_positions, tracked_positions, canvas, query_ids):
    """Before the next packet, EVERY changed identity must enter a clean query.

    This checks the mandatory query identity, not equality to fresh full-model
    KV. The future cache remains approximate even after identity repair.
    """
    tracked = tuple(map(int, tracked_positions))
    if len(set(tracked)) != len(tracked) or len(tracked) != len(query_ids):
        raise ValueError("invalid repair rows")
    by_position = dict(zip(tracked, query_ids))
    for position in dirty_positions:
        if position not in by_position or by_position[position] != canvas[position]:
            raise ValueError("dirty token identity was not repaired")
    return True

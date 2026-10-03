"""Separate deterministic-match admission; NOT equivalent to native LLaDA."""
import math
from .packet import Decision


def decide(probabilities, top1, drafts, *, gamma=.8, forbidden=()):
    if not len(probabilities)==len(top1)==len(drafts) or not drafts:
        raise ValueError('invalid match geometry')
    if any(not math.isfinite(p) or not 0<=p<=1 for p in probabilities):
        raise ValueError('nonfinite diagnostic probability')
    banned = set(forbidden)
    accepted = 0
    for predicted, proposed in zip(top1,drafts):
        if proposed in banned or predicted!=proposed:
            break
        accepted += 1
    correction = None
    tokens = tuple(map(int,drafts[:accepted]))
    if accepted<len(drafts) and top1[accepted] not in banned:
        correction = int(top1[accepted]); tokens += (correction,)
    return Decision(accepted,tokens,correction)

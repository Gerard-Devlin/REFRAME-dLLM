"""Identity-aware cache metadata and prevalidated clean-only transactions."""
from dataclasses import dataclass


@dataclass(frozen=True)
class Ticket:
    epoch: int
    positions: tuple
    tokens: tuple
    provenance: str


class Ledger:
    def __init__(self, tokens):
        self.tokens = list(tokens)
        self.cached_tokens = list(tokens)
        self.observed_epoch = [0] * len(tokens)
        self.drift = [0.] * len(tokens)
        self.epoch = 0

    def dirty(self):
        return tuple(i for i, (a, b) in enumerate(zip(self.tokens, self.cached_tokens)) if a != b)

    def change(self, positions, tokens):
        pairs = tuple(zip(positions, tokens))
        if len(positions) != len(tokens) or len(set(positions)) != len(positions):
            raise ValueError('Unique paired commits required')
        if any(i < 0 or i >= len(self.tokens) or v < 0 for i, v in pairs):
            raise ValueError('Invalid commit')
        if any(self.tokens[i] != v for i, v in pairs):
            for i, v in pairs:
                self.tokens[i] = v
            self.epoch += 1

    def ticket(self, positions, provenance='clean'):
        p = tuple(positions)
        if len(set(p)) != len(p) or any(i < 0 or i >= len(self.tokens) for i in p):
            raise ValueError('Unique initialized positions required')
        return Ticket(self.epoch, p, tuple(self.tokens[i] for i in p), provenance)

    def validate(self, ticket):
        if ticket.provenance != 'clean' or ticket.epoch != self.epoch:
            raise ValueError('Draft/stale transactions cannot enter the background cache')
        if ticket.tokens != tuple(self.tokens[i] for i in ticket.positions):
            raise ValueError('Identity changed while forward was pending')

    def mark_refreshed(self, ticket, drifts=None):
        self.validate(ticket)
        if drifts is not None and len(drifts) != len(ticket.positions):
            raise ValueError('Paired observed drift required')
        for j, (i, v) in enumerate(zip(ticket.positions, ticket.tokens)):
            self.cached_tokens[i] = v
            self.observed_epoch[i] = self.epoch
            if drifts is not None:
                self.drift[i] = float(drifts[j])

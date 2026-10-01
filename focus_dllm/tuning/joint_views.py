"""Legal common rows plus private, ordered proposal views.

This is a conditional computation experiment, not an equivalence proof for the
native bidirectional decoder. Common MASK placeholders remain label independent.
Private verification rows never become keys for another row or public cache.
"""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class JointLayout:
    tokens: tuple
    positions: tuple
    common: int
    active: int
    search: int
    proposal_rows: tuple
    tile: int

    @property
    def data_begin(self):return self.common

    @property
    def verify_begin(self):return self.common + self.search

    def mask(self):
        result = torch.zeros((self.tile, self.tile), dtype=torch.bool)
        # All common predictions and K/V are blind to ALL speculative labels.
        result[:, :self.common] = True
        for i in range(self.search):
            data = self.data_begin + i
            verify = self.verify_begin + i
            for j, placeholder in enumerate(self.proposal_rows):
                if j <= i:result[data, placeholder] = False
                if j < i:result[verify, placeholder] = False
            result[data, self.data_begin:self.data_begin+i+1] = True
            result[verify, self.data_begin:self.data_begin+i] = True
        # Padding queries are harmless; padding keys are invisible everywhere.
        return result


def layout(active_positions, active_tokens, tracked_positions, tracked_tokens,
           proposed_positions, proposed_tokens, *, mask_id=126336,
           forbidden=(), tile=128, cache_length=None):
    p, x, t, y, d, z = (tuple(map(int, value)) for value in
        (active_positions, active_tokens, tracked_positions, tracked_tokens,
         proposed_positions, proposed_tokens))
    if len(p)!=len(x) or len(t)!=len(y) or len(d)!=len(z):
        raise ValueError('Position/token lengths differ')
    if not p or len(p)>32 or len(t)>64 or len(d)>16:
        raise ValueError('Outside fixed active/tracked/proposal budget')
    if len(set(p+t))!=len(p+t) or len(set(d))!=len(d):
        raise ValueError('Repeated public or proposal position')
    if any(v!=mask_id for v in x) or any(v==mask_id for v in y):
        raise ValueError('Active must be MASK; tracked must be legitimate decoded tokens')
    if any(v not in p for v in d):raise ValueError('Proposal is not currently active')
    if any(v==mask_id or v in forbidden for v in z):raise ValueError('Special proposal token')
    if any(v<0 or (cache_length is not None and v>=cache_length) for v in p+t):
        raise ValueError('Position outside initialized canvas')
    common=len(p+t);used=common+2*len(d)
    if tile!=128 or used>tile:raise ValueError('Joint kernel requires one 128-row tile')
    positions=p+t+d+d+(p[0],)*(tile-used)
    tokens=x+y+z+(mask_id,)*len(d)+(mask_id,)*(tile-used)
    return JointLayout(tokens,positions,common,len(p),len(d),tuple(p.index(v) for v in d),tile)


def label_reachability(plan, layers=32):
    """Structural (potential) dependencies, including each residual connection."""
    paths=torch.zeros((plan.tile,plan.search),dtype=torch.bool)
    if plan.search:paths[plan.data_begin:plan.verify_begin]=torch.eye(plan.search,dtype=torch.bool)
    edges=plan.mask()|torch.eye(plan.tile,dtype=torch.bool)
    for _ in range(layers):paths=(edges.float()@paths.float())>0
    return paths


def public_cache_write_positions(plan):
    return plan.positions[:plan.common]


def pending_candidates(previous, active_positions, forbidden=(), limit=16):
    """Old paid predictions are proposals only; they require a NEW verification."""
    active=set(map(int,active_positions));forbidden=set(forbidden)
    eligible=[v for v in previous if int(v['position']) in active and int(v['token']) not in forbidden]
    eligible.sort(key=lambda v:(-float(v['confidence']),int(v['position'])))
    return eligible[:limit]

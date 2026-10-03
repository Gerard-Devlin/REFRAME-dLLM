"""FIREBREAK private attention, written for query-dependent KV version choices.

Base and draft banks are streamed separately, with mutually exclusive masks
for each original position. Verification/observer states are never key banks.
Uses the conventional running-max/running-sum softmax recurrence.
"""
import triton
import triton.language as tl


@triton.jit
def versioned_attention(Q, BK, BV, DK, DV, Map, Choice, O,
                        M: tl.constexpr, N: tl.constexpr, B: tl.constexpr,
                        H: tl.constexpr, D: tl.constexpr, SCALE: tl.constexpr,
                        BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
    row = tl.program_id(0)*BM + tl.arange(0, BM)
    head = tl.program_id(1)
    dim = tl.arange(0, D)
    q = tl.load(Q+(row[:, None]*H+head)*D+dim[None],
                mask=row[:, None] < M, other=0.)
    maximum = tl.full((BM,), float('-inf'), tl.float32)
    mass = tl.zeros((BM,), tl.float32)
    output = tl.zeros((BM, D), tl.float32)
    for begin in range(0, N, BN):
        key = begin + tl.arange(0, BN)
        label = tl.load(Map+key, mask=key < N, other=-1)
        choose = tl.load(Choice+row[:, None]*B+tl.maximum(label[None], 0),
                         mask=(row[:, None] < M) & (label[None] >= 0), other=0)
        allowed = (key[None] < N) & ~choose & (label[None] != -2)
        k = tl.load(BK+(key[None]*H+head)*D+dim[:, None],
                    mask=key[None] < N, other=0.)
        v = tl.load(BV+(key[:, None]*H+head)*D+dim[None],
                    mask=key[:, None] < N, other=0.)
        score = tl.where(allowed, tl.dot(q, k)*SCALE, float('-inf'))
        updated = tl.maximum(maximum, tl.max(score, 1))
        safe = tl.where(updated == float('-inf'), 0., updated)
        decay = tl.exp(maximum-safe)
        p = tl.where(allowed, tl.exp(score-safe[:, None]), 0.)
        output = output*decay[:, None] + tl.dot(p.to(v.dtype), v)
        mass = mass*decay + tl.sum(p, 1)
        maximum = updated
    key = tl.arange(0, BD)
    allowed = tl.load(Choice+row[:, None]*B+key[None],
                      mask=(row[:, None] < M) & (key[None] < B), other=0)
    k = tl.load(DK+(key[None]*H+head)*D+dim[:, None],
                mask=key[None] < B, other=0.)
    v = tl.load(DV+(key[:, None]*H+head)*D+dim[None],
                mask=key[:, None] < B, other=0.)
    score = tl.where(allowed, tl.dot(q, k)*SCALE, float('-inf'))
    updated = tl.maximum(maximum, tl.max(score, 1))
    safe = tl.where(updated == float('-inf'), 0., updated)
    decay = tl.exp(maximum-safe)
    p = tl.where(allowed, tl.exp(score-safe[:, None]), 0.)
    output = output*decay[:, None] + tl.dot(p.to(v.dtype), v)
    mass = mass*decay + tl.sum(p, 1)
    tl.store(O+(row[:, None]*H+head)*D+dim[None], output/mass[:, None],
             mask=row[:, None] < M)

"""One-layer sparse, self-inverse BPE coupling. No vocabulary-squared tables.

Fixed pairs (1,2), (3,4), ... within each block. Position zero is unchanged
because it is seeded by the preceding clean block's shifted LM head.
Only pairs wholly inside the assistant response are transformed.
For each observed anchor, swap one frequent follower with a shared code ID.
Swaps protect every tokenizer special ID. This is a candidate, not a learned
semantic relation model or a guaranteed dependency reduction.
"""
from collections import Counter, defaultdict
import torch
from torch import nn


def fit(rows, block_size, vocab_size, protected, min_count=8):
    counts = defaultdict(Counter)
    protected = set(protected)
    for row in rows:
        x, p = row["ids"], row["prefix"]
        for start in range(0, len(x), block_size):
            for a in range(start + 1, min(start + block_size - 1, len(x) - 1), 2):
                if a < p or x[a] in protected or x[a + 1] in protected:
                    continue
                counts[x[a]][x[a + 1]] += 1
    # Choose a common follower ID as shared relation code, keeping most IDs
    # untouched and avoiding vocabulary expansion / random new embeddings.
    totals = Counter()
    for c in counts.values():
        totals.update(c)
    if not totals:
        raise ValueError("No eligible response pairs")
    code = min(totals, key=lambda t: (-totals[t], t))
    swaps = []
    for a, c in sorted(counts.items()):
        target = min(c, key=lambda t: (-c[t], t))
        if c[target] >= min_count and target != code and c[target] > c[code]:
            swaps.append([a, target, code])
    if not swaps:
        raise ValueError("No supported relation swaps; do not start training")
    return dict(version=1, block_size=block_size, vocab_size=vocab_size,
                protected=sorted(protected), swaps=swaps, shared_code=code,
                min_count=min_count, fitting="train-only frequent-follower swap")


class Codec(nn.Module):
    def __init__(self, spec, identity=False):
        super().__init__()
        self.spec, self.identity = spec, identity
        self.block_size = spec["block_size"]
        v = spec["vocab_size"]
        # O(V) maps, not O(V^2). Each anchor owns one transposition.
        left, right = torch.full((v,), -1), torch.full((v,), -1)
        protected = set(spec["protected"])
        for a, s, t in spec["swaps"]:
            if not all(0 <= i < v for i in (a, s, t)) or protected.intersection((a, s, t)):
                raise ValueError("Invalid or protected codec ID")
            if left[a] != -1 or s == t:
                raise ValueError("Duplicate anchor / invalid swap")
            left[a], right[a] = s, t
        self.register_buffer("left", left)
        self.register_buffer("right", right)

    def forward(self, ids, prefix, offset=0):
        """Both encode and decode; offset and prefix are absolute positions."""
        z = ids.clone()
        if self.identity:
            return z
        n = ids.shape[1]
        positions = torch.arange(n - 1, device=ids.device)
        absolute = positions + offset
        anchors = positions[(absolute % self.block_size % 2 == 1) &
                            (absolute % self.block_size < self.block_size - 1)]
        prefix = torch.as_tensor(prefix, device=ids.device).reshape(-1, 1)
        valid = (anchors[None, :] + offset >= prefix)
        a, b = ids[:, anchors], ids[:, anchors + 1]
        s, t = self.left[a], self.right[a]
        z[:, anchors + 1] = torch.where(valid & (b == s), t,
                                      torch.where(valid & (b == t), s, b))
        return z

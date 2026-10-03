"""Ordered DAGs describe permitted information, not inferred causal truth."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class DAG:
    parents: tuple

    def __post_init__(self):
        if not self.parents or len(self.parents) > 32:
            raise ValueError('One to32 candidates required')
        for i, row in enumerate(self.parents):
            if len(set(row)) != len(row) or any(not isinstance(p, int) or p < 0 or p >= i for p in row):
                raise ValueError('Unique parents must precede the child')

    def ancestors(self):
        result = []
        for row in self.parents:
            bits = 0
            for p in row:
                bits |= (1 << p) | result[p]
            result.append(bits)
        return tuple(result)

    def closed_accept(self, passes):
        if len(passes) != len(self.parents):
            raise ValueError('One pass flag per candidate required')
        accepted = set()
        for i, row in enumerate(self.parents):
            if passes[i] and all(p in accepted for p in row):
                accepted.add(i)
        return tuple(sorted(accepted))

    def subset(self, selected):
        selected = tuple(sorted(set(selected)))
        if any(i < 0 or i >= len(self.parents) for i in selected):
            raise ValueError('Unknown candidate')
        remap = {old: new for new, old in enumerate(selected)}
        if any(p not in remap for i in selected for p in self.parents[i]):
            raise ValueError('Execution set must contain all prerequisites')
        return DAG(tuple(tuple(remap[p] for p in self.parents[i]) for i in selected))

    def label_paths(self, layers=32):
        """Worst-case label reachability through all private key/residual paths."""
        if layers < 0:
            raise ValueError('Nonnegative layer count required')
        ancestors = self.ancestors()
        n = len(ancestors)
        paths = [1 << i for i in range(n)] + [0] * n
        for _ in range(layers):
            updated = []
            for r, old in enumerate(paths):
                i = r % n
                permitted = ancestors[i] | ((1 << i) if r < n else 0)
                for j in range(n):
                    if permitted & (1 << j):
                        old |= paths[j]
                updated.append(old)
            paths = updated
        return tuple(paths)


def build_dag(interaction, *, max_parents=2, minimum=.15, max_ancestors=8):
    """Fixed confidence order; cheap Q/pooled-K interaction is only a proxy."""
    n = len(interaction)
    if not 0 <= max_parents <= 4 or not 0 <= minimum <= 1 or max_ancestors < 0:
        raise ValueError('Invalid graph geometry')
    if any(len(row) != n or any(not math.isfinite(v) or v < 0 for v in row) for row in interaction):
        raise ValueError('Finite nonnegative square interaction required')
    rows, ancestors = [], []
    for i in range(n):
        total = sum(interaction[i][:i])
        eligible = sorted(range(i), key=lambda j: (-interaction[i][j], j))
        parents, bits = [], 0
        for p in eligible:
            if len(parents) >= max_parents:
                break
            if not total or interaction[i][p]/total < minimum:
                continue
            proposed = bits | (1 << p) | ancestors[p]
            if proposed.bit_count() <= max_ancestors:
                parents.append(p); bits = proposed
        rows.append(tuple(sorted(parents))); ancestors.append(bits)
    return DAG(tuple(rows))

"""Version choices and provenance, independent of a GPU or model implementation."""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Layout:
    positions: tuple
    tokens: tuple
    groups: tuple
    cross: bool
    cache_length: int

    @property
    def count(self):
        return len(self.positions)

    @property
    def families(self):
        return 3 if self.cross else 2

    def choices(self):
        """True selects the draft version; False selects the original base KV."""
        result = []
        for family in range(self.families):
            for i, group in enumerate(self.groups):
                result.append(tuple(
                    (other == group and (j <= i if family == 0 else j < i))
                    or (family == 2 and other != group)
                    for j, other in enumerate(self.groups)))
        return tuple(result)

    def dependencies(self, layers=32):
        """Potential injected-label dependencies, including the residual paths."""
        if layers < 0:
            raise ValueError('Nonnegative depth required')
        n = self.count
        paths = [1 << i for i in range(n)] + [0] * ((self.families-1)*n)
        choices = self.choices()
        for _ in range(layers):
            paths = [old | _union(paths[j] for j, enabled in enumerate(row) if enabled)
                     for old, row in zip(paths, choices)]
        return tuple(paths)

    def position_versions(self, query):
        """Exactly one source for every original key position."""
        row = self.choices()[query]
        selected = {p: j for j, p in enumerate(self.positions) if row[j]}
        return tuple(('draft', selected[p]) if p in selected else ('base', p)
                     for p in range(self.cache_length))


def _union(values):
    result = 0
    for value in values:
        result |= value
    return result


def layout(positions, tokens, *, cache_length, group_count, cross=True, forbidden=()):
    p, t = tuple(map(int, positions)), tuple(map(int, tokens))
    if not p or len(p) > 32 or len(p) != len(t) or len(set(p)) != len(p):
        raise ValueError('One to32 unique paired candidates required')
    if group_count not in (1, 2, 4) or any(x < 0 or x >= cache_length for x in p):
        raise ValueError('Fixed group count and initialized positions required')
    if any(x < 0 or x in forbidden for x in t):
        raise ValueError('Invalid or special draft token')
    width = math.ceil(len(p)/group_count)
    groups = tuple(i//width for i in range(len(p)))
    return Layout(p, t, groups, bool(cross), int(cache_length))


def prefix_union(groups, flags):
    if len(groups) != len(flags):
        raise ValueError('Paired groups/pass flags required')
    failed, accepted = set(), []
    for i, (group, passed) in enumerate(zip(groups, flags)):
        if not passed:
            failed.add(group)
        if group not in failed:
            accepted.append(i)
    return tuple(accepted)


def dependency_closed(plan, accepted):
    bits = _union(1 << i for i in accepted)
    deps = plan.dependencies()
    return all(deps[plan.count+i] & ~bits == 0 for i in accepted)


def cumulative_prefixes(groups, probabilities, matches, gamma):
    if len(groups) != len(probabilities) or len(groups) != len(matches):
        raise ValueError('Paired probability arrays required')
    products, flags = {}, []
    for group, probability, match in zip(groups, probabilities, matches):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError('Invalid candidate probability')
        products[group] = products.get(group, 1.) * probability
        flags.append(bool(match) and products[group] >= gamma)
    return prefix_union(groups, flags)

"""Exact maximum-weight closure of a fixed *proxy* cost/benefit graph."""
from collections import deque
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class Plan:
    selected: tuple
    objective: float


def maximum_closure(weights, prerequisites, mandatory=()):
    n = len(weights)
    if len(prerequisites) != n or any(not math.isfinite(w) for w in weights):
        raise ValueError('Finite paired weights/requirements required')
    if any(any(not isinstance(j, int) or j < 0 or j >= n for j in row) for row in prerequisites):
        raise ValueError('Unknown prerequisite')
    mandatory = tuple(mandatory)
    if any(not isinstance(i, int) or i < 0 or i >= n for i in mandatory):
        raise ValueError('Unknown mandatory task')
    source, sink = n, n+1
    network = [[] for _ in range(n+2)]

    def edge(a, b, c):
        network[a].append([b, len(network[b]), float(c)])
        network[b].append([a, len(network[a])-1, 0.])

    infinite = sum(abs(w) for w in weights)+1.
    for i, w in enumerate(weights):
        edge(source, i, max(w, 0.))
        edge(i, sink, max(-w, 0.))
        for j in prerequisites[i]:
            if i != j:
                edge(i, j, infinite)
    for i in mandatory:
        edge(source, i, infinite)
    eps = 1e-12
    while True:
        level = [-1]*(n+2); level[source] = 0
        queue = deque([source])
        while queue:
            a = queue.popleft()
            for b, _, c in network[a]:
                if c > eps and level[b] < 0:
                    level[b] = level[a]+1; queue.append(b)
        if level[sink] < 0:
            break
        cursor = [0]*(n+2)

        def push(a, flow):
            if a == sink:
                return flow
            while cursor[a] < len(network[a]):
                e = network[a][cursor[a]]
                b, rev, c = e
                if c > eps and level[b] == level[a]+1:
                    got = push(b, min(flow, c))
                    if got > eps:
                        e[2] -= got; network[b][rev][2] += got
                        return got
                cursor[a] += 1
            return 0.
        while push(source, infinite) > eps:
            pass
    reachable, queue = {source}, deque([source])
    while queue:
        for b, _, c in network[queue.popleft()]:
            if c > eps and b not in reachable:
                reachable.add(b); queue.append(b)
    selected = tuple(i for i in range(n) if i in reachable)
    assert all(i in reachable for i in mandatory)
    assert all(j in reachable for i in selected for j in prerequisites[i])
    return Plan(selected, sum(weights[i] for i in selected))


def joint_plan(dag, benefits, requirements, costs, *, price=.12, query_cost=2.):
    n, m = len(benefits), len(costs)
    if n != len(dag.parents) or len(requirements) != n or price < 0 or not math.isfinite(price):
        raise ValueError('Paired graph/benefit/requirements required')
    if any(c < 0 or not math.isfinite(c) for c in costs) or query_cost < 0 or not math.isfinite(query_cost):
        raise ValueError('Finite nonnegative task costs required')
    if any(any(j < 0 or j >= m for j in row) for row in requirements):
        raise ValueError('Unknown cache tile')
    weights = [b-price*query_cost for b in benefits]+[-price*c for c in costs]
    prerequisites = [tuple(dag.parents[i])+tuple(n+j for j in requirements[i]) for i in range(n)]
    prerequisites += [()] * m
    result = maximum_closure(weights, prerequisites)
    return dict(candidates=tuple(i for i in result.selected if i < n),
                tiles=tuple(i-n for i in result.selected if i >= n), objective=result.objective)

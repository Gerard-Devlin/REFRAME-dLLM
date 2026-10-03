"""Backward necessary dependencies of the pinned Flash T/D/M verification graph.

Edges mean possible mathematical dependence, not a measured error. Background
KV is fixed, positionwise norms/MLPs and residual edges are included. Verification
does not write the public cache in the pinned implementation.
"""
import math


def reference_mask(block=32, search=16):
    if block < 1 or search < 1 or search > block//2:
        raise ValueError('Positive pinned verification geometry required')
    tracked = 2*block-2*search
    rows = 2*block
    result = []
    for query in range(rows):
        row = []
        for key in range(rows):
            if query < tracked:
                allowed = not tracked <= key < tracked+search
            elif query < tracked+search:
                i = query-tracked
                allowed = key < tracked or (key < tracked+search and key-tracked <= i) or (key >= tracked+search and key-tracked-search > i)
            else:
                i = query-tracked-search
                allowed = key < tracked or (key < tracked+search and key-tracked < i) or (key >= tracked+search and key-tracked-search >= i)
            row.append(allowed)
        result.append(tuple(row))
    return tuple(result)


def required_rows(mask, targets, layers):
    if layers < 1 or not mask or any(len(row) != len(mask) for row in mask):
        raise ValueError('Square graph and positive depth required')
    needed = set(targets)
    if not needed or min(needed) < 0 or max(needed) >= len(mask):
        raise ValueError('Nonempty valid output set required')
    stages = [None]*(layers+1)
    stages[layers] = tuple(sorted(needed))
    for layer in range(layers, 0, -1):
        previous = set(needed)  # residual; tokenwise MLP/norm preserve rows
        previous.update(key for query in needed for key, enabled in enumerate(mask[query]) if enabled)
        needed = previous
        stages[layer-1] = tuple(sorted(needed))
    return tuple(stages)


def analyze(block=32, search=16, layers=32, prefix=1):
    if not 1 <= prefix <= search:
        raise ValueError('A verification prefix is required')
    mask = reference_mask(block, search)
    tracked = 2*block-2*search
    start = tracked+search
    stages = required_rows(mask, range(start,start+prefix),layers)
    full = required_rows(mask,range(start,start+search),layers)
    # Output-row counts are an optimistic work proxy. KV projections for ALL
    # necessary previous-layer keys remain mandatory, even at the final layer.
    proxy = sum(map(len,stages[1:])); full_proxy = sum(map(len,full[1:]))
    return dict(block=block,search=search,tracked=tracked,layers=layers,prefix=prefix,
        required_output_rows_by_layer=[len(s) for s in stages[1:]],
        required_input_rows=len(stages[0]),
        early_required_candidate_mask_rows=sum(start <= i < start+search for i in stages[0]),
        early_required_draft_rows=sum(tracked <= i < tracked+search for i in stages[0]),
        full_mask_target_output_row_proxy=full_proxy,prefix_output_row_proxy=proxy,
        prefix_vs_full_target_row_proxy_saving=1-proxy/full_proxy,
        scope='Structural necessary dependencies. Row proxy is not FLOPs or latency; key projection, launch, head and cache costs remain.')


def optimal_batches(cost, survive):
    """Bellman DP for GIVEN costs/conditional survival; not language prediction."""
    n = len(cost)-1
    if n < 1 or len(survive) != n+1:
        raise ValueError('Paired square tables required')
    value = [0.]*(n+1); next_end = [None]*(n+1)
    for i in range(n-1,-1,-1):
        choices = []
        for j in range(i+1,n+1):
            c,p = cost[i][j],survive[i][j]
            if c is None or p is None:
                continue
            if not math.isfinite(c) or c < 0 or not math.isfinite(p) or not 0 <= p <= 1:
                raise ValueError('Nonnegative cost and conditional probability required')
            choices.append((c+p*value[j],j))
        if not choices:
            raise ValueError('At least one legal batch required from every state')
        value[i],next_end[i] = min(choices)
    return value,next_end

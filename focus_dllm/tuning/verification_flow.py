"""Small information-flow and rejection diagnostics, not a decoder.

An absent direct attention edge is not a transitive noninterference guarantee.
The reachability calculation finds possible paths, not measured leakage or
generation errors. The alternative mask changes the target context; it is not
an exact implementation of the native bidirectional model.
"""
import ast
import functools
import inspect
import math
import textwrap

import torch


def view_mask(block, search, *, isolated=False):
    if not isinstance(block, int) or not isinstance(search, int) or block < 1 or not 0 <= search <= block // 2:
        raise ValueError('Invalid fixed query budget')
    tracked = 2 * block - 2 * search
    result = torch.ones(2 * block, 2 * block, dtype=torch.bool)
    ranks = torch.arange(search)
    earlier_equal = ranks[:, None] >= ranks[None, :]
    earlier = ranks[:, None] > ranks[None, :]
    if isolated:
        # Shared context cannot acquire speculative information. Both kinds of
        # speculative rows see only earlier temporal ranks (data sees itself).
        result[:tracked, tracked:] = False
        result[tracked:tracked + search, tracked:tracked + search] = earlier_equal
        result[tracked:tracked + search, tracked + search:] = earlier_equal
        result[tracked + search:, tracked:tracked + search] = earlier
        result[tracked + search:, tracked + search:] = earlier_equal
    else:
        # Exact boolean topology of the pinned Flash-dLLM verify query.
        result[:tracked, tracked:tracked + search] = False
        result[tracked:tracked + search, tracked:tracked + search] = earlier_equal
        result[tracked:tracked + search, tracked + search:] = ~earlier_equal
        result[tracked + search:, tracked:tracked + search] = earlier
        result[tracked + search:, tracked + search:] = ~earlier
    return result


def proposal_paths(mask, tracked, search, layers):
    """Query x proposal reachability, including residual connections.

This assumes positionwise norms/MLPs and a base cache containing no speculative
proposal. A proposed label enters only its data row. RoPE is positionwise.
"""
    if mask.dtype != torch.bool or mask.ndim != 2 or mask.shape[0] != mask.shape[1]:
        raise ValueError('Square boolean query mask required')
    if tracked < 0 or search < 0 or tracked + 2 * search != mask.shape[0] or layers < 0:
        raise ValueError('Invalid view geometry')
    value = torch.zeros(mask.shape[0], search, dtype=torch.bool)
    value[tracked:tracked + search] = torch.eye(search, dtype=torch.bool)
    for _ in range(layers):
        value |= (mask.to(torch.int64) @ value.to(torch.int64)) > 0
    return value


def rejection_record(probabilities, drafts, positions, top1, gamma=.8, forbidden=()):
    if not .5 <= gamma <= 1:
        raise ValueError('Diagnostic gamma must be in [.5,1]')
    values = list(map(float, probabilities))
    if not len(values) == len(drafts) == len(positions) == len(top1):
        raise ValueError('Aligned candidate arrays required')
    if len(set(positions)) != len(positions) or any(not math.isfinite(p) or not 0 <= p <= 1 for p in values):
        raise ValueError('Unique positions and valid probabilities required')
    cumulative = []
    mass = 1.
    for value in values:
        mass *= value
        cumulative.append(mass)
    accepted = sum(p >= gamma for p in cumulative)
    first = accepted if accepted < len(values) else None
    forbidden = set(forbidden)
    high = [i for i in range((first + 1) if first is not None else len(values), len(values))
            if values[i] >= .90 and int(drafts[i]) == int(top1[i]) and int(drafts[i]) not in forbidden]
    return dict(probabilities=values, drafts=list(map(int, drafts)), positions=list(map(int, positions)),
        top1=list(map(int, top1)), cumulative=cumulative, accepted=accepted, first_rejected=first,
        first_rejected_probability=values[first] if first is not None else None,
        budget_only_rejection=first is not None and values[first] >= gamma,
        rejected_argmax_mismatch=first is not None and int(drafts[first]) != int(top1[first]),
        discarded_high_confidence_indices=high,
        scope='Later probabilities condition on earlier drafts including rejected ones; not reusable accepted tokens or saved calls.')


def observe_cumulative(function, observer):
    """Add precisely one read-only callback before the pinned cumprod line.

    Original function/globals/source file are never changed. An unfamiliar
    source layout fails rather than applying an approximate monkey patch.
    """
    original = inspect.unwrap(function)
    source = textwrap.dedent(inspect.getsource(original))
    tree = ast.parse(source)
    definition = tree.body[0]
    if not isinstance(definition, ast.FunctionDef):
        raise ValueError('A standalone Python function is required')
    definition.decorator_list = []
    expected = 'x0_p_verify = x0_p_verify.cumprod(dim=0)'
    additions = [0]

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node) != expected:
                return node
            additions[0] += 1
            callback = ast.parse('_focus_observe(x0_p_verify, x_verify_j, '
                'query_pos_flat[acc_seqlen_verify + T:acc_seqlen_verify + T + S], '
                'p_verify.argmax(dim=-1), gamma)').body[0]
            return [ast.copy_location(callback, node), node]

    tree = Inject().visit(tree)
    if additions[0] != 1:
        raise ValueError('Expected exactly one pinned cumulative acceptance statement')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    if '_focus_observe' in namespace:
        raise ValueError('Observer namespace collision')
    namespace['_focus_observe'] = observer
    exec(compile(tree, original.__code__.co_filename + ':readonly_observer', 'exec'), namespace)
    copied = torch.no_grad()(namespace[original.__name__])
    return functools.update_wrapper(copied, original)

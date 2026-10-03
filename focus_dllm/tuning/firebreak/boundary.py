"""Recorded, process-local correction of a pinned Flash ragged-window consumer.

Third-party generation is a credited baseline, not our verification algorithm.
No third-party file is edited. Unknown source shapes fail closed.
"""
import ast
import functools
import inspect
import textwrap
import torch


def adapted(function, action_observer=None):
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef)]
    if len(defs) != 1:
        raise ValueError('Single pinned generation function required')
    defs[0].decorator_list = []
    fixes, observers = 0, 0
    expected = ast.parse('logits_masked_j = logits[acc_seqlen_masked : acc_seqlen_masked + block_m]').body[0]
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and ast.dump(node, include_attributes=False) == ast.dump(expected, include_attributes=False):
            node.value.slice.upper = ast.parse('acc_seqlen_masked + min(block_m, query_masked_pos[j].shape[0])', mode='eval').body
            fixes += 1
    if fixes != 1:
        raise ValueError('Pinned boundary expression changed')
    class Observe(ast.NodeTransformer):
        def visit_Assign(self, node):
            nonlocal observers
            if action_observer is None:
                return node
            target = node.targets[0] if len(node.targets) == 1 else None
            if (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name)
                    and target.value.id == 'x' and isinstance(target.slice, ast.Name)
                    and target.slice.id == 'pos_decoded_new_j'):
                if not isinstance(node.value, ast.Name) or node.value.id != 'x0_decoded_new_j':
                    raise ValueError('Unfamiliar commit expression')
                observers += 1
                return [node, ast.copy_location(ast.parse('_firebreak_action(pos_decoded_new_j, x0_decoded_new_j)').body[0], node)]
            return node
    tree = Observe().visit(tree)
    if action_observer is not None and observers != 2:
        raise ValueError('Expected two actual commit sites')
    ast.fix_missing_locations(tree)
    scope = dict(original.__globals__)
    if '_firebreak_action' in scope:
        raise ValueError('Observer namespace collision')
    scope['_firebreak_action'] = action_observer
    exec(compile(tree, original.__code__.co_filename+':firebreak_boundary', 'exec'), scope)
    return functools.update_wrapper(torch.no_grad()(scope[defs[0].name]), original)

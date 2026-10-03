"""Read-only pinned source instrumentation plus common compact head metadata."""
import ast
import functools
import inspect
import textwrap

import torch


def instrument(function, before_verify, official_accept, final_canvas):
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree.body[0].decorator_list = []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'model']
    calls.sort(key=lambda n: n.lineno)
    if len(calls) != 2:
        raise ValueError('pinned generator changed')
    for call, expression in zip(calls, ('(0, block_m)', '(seqlen_keep[0] + num_verify, num_verify)')):
        call.keywords.append(ast.keyword(arg='focus_head_rows', value=ast.parse(expression, mode='eval').body))
    counts = [0, 0, 0]

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            callback, index = None, None
            if node.value is calls[1]:
                callback, index = '_v7_before(model, locals())', 0
            elif ast.unparse(node) == 'x0_p_verify = x0_p_verify.cumprod(dim=0)':
                callback, index = '_v7_accept(x0_p_verify, x_verify_j, gamma)', 1
            elif ast.unparse(node).startswith('generated_answer_ids = x['):
                callback, index = '_v7_final(x, prompt_lengths[i], max_length, gen_length)', 2
            if callback is None:
                return self.generic_visit(node)
            counts[index] += 1
            return [ast.copy_location(ast.parse(callback).body[0], node), node]

    tree = Inject().visit(tree)
    if counts != [1, 1, 1]:
        raise ValueError(f'pinned observer layout changed: {counts}')
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace.update(_v7_before=before_verify, _v7_accept=official_accept, _v7_final=final_canvas)
    exec(compile(tree, original.__code__.co_filename+':v7_observer', 'exec'), namespace)
    return functools.update_wrapper(torch.no_grad()(namespace[original.__name__]), original)

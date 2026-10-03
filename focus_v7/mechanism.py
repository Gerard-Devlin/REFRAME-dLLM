"""Read-only v7 instrumentation and paid fresh-bank diagnostic helpers."""
import ast
from contextlib import contextmanager
import functools
import inspect
import textwrap

import torch


def instrument(function, observer):
    """Observe one known decision site before real promotion/commit."""
    source = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(source)))
    tree.body[0].decorator_list = []
    count = [0]

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            self.generic_visit(node)
            if (len(node.targets) == 1 and isinstance(node.targets[0], ast.Name)
                    and node.targets[0].id == 'decision' and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name) and node.value.func.id == 'decide'):
                count[0] += 1
                extra = ast.parse('_packet_observer(dict(locals()))').body[0]
                return [node, ast.copy_location(extra, node)]
            return node
    tree = Inject().visit(tree)
    if count != [1]:
        raise ValueError('expected one frozen v7 decision site')
    ast.fix_missing_locations(tree)
    scope = dict(source.__globals__, _packet_observer=observer)
    exec(compile(tree, source.__code__.co_filename+':mechanism_observer', 'exec'), scope)
    return functools.update_wrapper(torch.no_grad()(scope[source.__name__]), source)


@contextmanager
def private_cache(blocks):
    """Fresh private storage; restore original tensor objects even on failure."""
    saved = [(b.k_cache, b.v_cache) for b in blocks]
    try:
        for block, (k, v) in zip(blocks, saved):
            block.k_cache, block.v_cache = torch.empty_like(k), torch.empty_like(v)
        yield
    finally:
        for block, (k, v) in zip(blocks, saved):
            block.k_cache, block.v_cache = k, v


def full_call(current, candidates):
    """Full same-state normal forward with selected rows first for readout.

    Covers EXACTLY initialized physical key positions, including EOS padding.
    It never inserts a draft identity or reads a later trajectory state.
    """
    raw, state = current['raw'], current['state']
    key_length = int(state['seqlen_k'][0])
    chosen = list(map(int, candidates))
    if not chosen or len(set(chosen)) != len(chosen) or any(p < 0 or p >= key_length for p in chosen):
        raise ValueError('invalid selected physical positions')
    if not bool((current['canvas'][torch.tensor(chosen, device=raw.device)] == 126336).all()):
        raise ValueError('full reference must use the same current MASK canvas')
    others = [p for p in range(key_length) if p not in set(chosen)]
    qpos = torch.tensor(chosen+others, device=raw.device)
    def blocks(start, end):
        starts = torch.arange(start, end, 32, device=raw.device)
        ends = torch.minimum(starts+32, torch.tensor(end, device=raw.device))
        return torch.stack((torch.zeros_like(starts), torch.full_like(starts,key_length), starts, ends),1).int()
    masked, tracked = blocks(0,len(chosen)), blocks(len(chosen),key_length)
    empty = torch.tensor([], device=raw.device, dtype=torch.int32)
    pos = [qpos, empty, state['rotary_emb_pos'], [], torch.zeros_like(state['attn_scores']), []]
    lengths = [[-1], None, masked, tracked, torch.cat((masked,tracked)), [0],
               masked.shape[0], int(current['maximum']), 32, 128, None, False]
    return current['canvas'][qpos].unsqueeze(0), pos, lengths


def statistics(logits, drafts=None, mask_id=126336):
    value = logits.detach().clone()
    value[:,mask_id] = -torch.inf
    probability = value.double().softmax(-1)
    confidence, predicted = probability.max(-1)
    candidate = None if drafts is None else probability.gather(1,drafts[:,None]).squeeze(1).tolist()
    return dict(top1=predicted.tolist(), confidence=confidence.tolist(), candidate_probability=candidate)

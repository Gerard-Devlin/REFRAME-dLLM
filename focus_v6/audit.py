"""Independent candidate-source views for a deferred atomic audit.

Layout: [clean tracked | clean candidate MASK | private draft | private audit].
The clean views never read drafts. Draft i reads only its own proposed identity
and clean versions of peers. Audit i reads peer drafts and its own MASK row,
but no other audit. Own-label exclusion therefore holds across all layers.
The main tentative cache lane may read every draft, but it must not be a key
for any audit/draft/clean query. Its outputs remain provisional until commit.
"""
import ast
from dataclasses import dataclass
import functools
import inspect
import textwrap
from typing import Any

import torch


@dataclass(frozen=True)
class Layout:
    candidates: int
    width: int = 64

    def __post_init__(self):
        if not 1 <= self.candidates <= 16 or self.width < 3 * self.candidates:
            raise ValueError("invalid audit width")

    @property
    def tracked(self):
        return self.width - 3 * self.candidates

    @property
    def clean(self):
        return slice(self.tracked, self.tracked + self.candidates)

    @property
    def draft(self):
        return slice(self.tracked + self.candidates, self.tracked + 2 * self.candidates)

    @property
    def audit(self):
        return slice(self.tracked + 2 * self.candidates, self.width)


def isolated_mask(layout, *, main_rows=0):
    if main_rows < 0:
        raise ValueError("invalid tentative main lane")
    w, k, t = layout.width, layout.candidates, layout.tracked
    mask = torch.zeros((w + main_rows, w + main_rows), dtype=torch.bool)
    mask[:t + k, :t + k] = True
    eye = torch.eye(k, dtype=torch.bool)
    mask[layout.draft, :t] = True
    mask[layout.draft, layout.clean] = ~eye
    mask[layout.draft, layout.draft] = eye
    mask[layout.audit, :t] = True
    mask[layout.audit, layout.draft] = ~eye
    mask[layout.audit, layout.audit] = eye
    if main_rows:
        mask[w:, :t] = True
        mask[w:, layout.draft] = True
        mask[w:, w:] = True
    return mask


def naive_own_exclusion_mask(layout):
    """Counterexample: peer draft rows can carry an own label back later."""
    result = isolated_mask(layout)
    result[layout.draft, layout.draft] = True
    return result


def label_reachability(mask, layout, layers=32):
    reachable = torch.zeros((mask.shape[0], layout.candidates), dtype=torch.bool)
    reachable[layout.draft] = torch.eye(layout.candidates, dtype=torch.bool)
    edges = mask.to(torch.int64)
    for _ in range(layers):
        reachable |= (edges @ reachable.to(torch.int64)) > 0
    return reachable


def build_call(state: dict[str, Any], k: int):
    if int(state["block_m"]) != 32 or state["active_batch"] != [0] or int(state["num_verify"]) < k:
        raise ValueError("fixed batch-one/block32 geometry or candidate count changed")
    layout = Layout(k)
    decoded = int(state["num_decoded"][0])
    if decoded < layout.tracked:
        raise ValueError("insufficient committed background")
    full = state["full_pos"][0]
    tracked = full[decoded - layout.tracked:decoded]
    candidates = full[decoded:decoded + k]
    clean_ids = state["x"][candidates]
    if not bool((clean_ids == int(state["mask_id"])).all()):
        raise AssertionError("a provisional candidate is already committed")
    drafts = state["x_draft"][candidates]
    query = torch.cat((state["x"][tracked], clean_ids, drafts, clean_ids)).unsqueeze(0)
    query_positions = torch.cat((tracked, candidates, candidates, candidates))
    external = torch.cat((full[:decoded - layout.tracked], full[decoded + k:int(state["seqlen_k"][0])]))
    if bool(torch.isin(external, torch.cat((tracked, candidates))).any()):
        raise AssertionError("private position retained a duplicate external version")
    blocks = torch.tensor([[0, external.numel(), 0, 64]], device=query.device, dtype=torch.int32)
    positions = [query_positions, external, state["rotary_emb_pos"], state["info"],
                 state["attn_scores"], isolated_mask(layout).to(query.device)]
    lengths = [list(state["start_layer"]), blocks, None, state["query_tracked_blocks"], None,
               list(state["active_batch"]), int(state["num_active"]), int(state["max_length"]),
               32, int(state["block_n"]), state["elastic_cache"], True]
    return query, positions, lengths, layout, candidates, drafts


def instrument(function, before_verify, official_accept, final_canvas):
    """Copy a pinned generator with three private read-only observer points."""
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree.body[0].decorator_list = []
    model_calls = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)
                   and isinstance(n.value, ast.Call) and isinstance(n.value.func, ast.Name)
                   and n.value.func.id == "model"]
    model_calls.sort(key=lambda n: n.lineno)
    if len(model_calls) != 2:
        raise ValueError("expected pinned regular and verify calls")
    counts = [0, 0, 0]

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            callback, index = None, None
            if node is model_calls[1]:
                callback, index = "_v6_before(model, locals())", 0
            elif ast.unparse(node) == "x0_p_verify = x0_p_verify.cumprod(dim=0)":
                callback, index = "_v6_accept(x0_p_verify, x_verify_j, gamma)", 1
            elif ast.unparse(node).startswith("generated_answer_ids = x["):
                callback, index = "_v6_final(x, prompt_lengths[i], max_length, gen_length)", 2
            if callback is None:
                return self.generic_visit(node)
            counts[index] += 1
            return [ast.copy_location(ast.parse(callback).body[0], node), node]

    tree = Inject().visit(tree)
    if counts != [1, 1, 1]:
        raise ValueError(f"pinned observer layout changed: {counts}")
    ast.fix_missing_locations(tree)
    namespace = dict(original.__globals__)
    namespace.update(_v6_before=before_verify, _v6_accept=official_accept, _v6_final=final_canvas)
    exec(compile(tree, original.__code__.co_filename + ":v6_observer", "exec"), namespace)
    return functools.update_wrapper(torch.no_grad()(namespace[original.__name__]), original)

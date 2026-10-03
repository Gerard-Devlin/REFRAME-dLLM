"""Bounded approximate cache-producing verifier, no teacher/oracle input."""
import ast
from dataclasses import dataclass
import functools
import inspect
import math
import textwrap

import torch
import torch.nn.functional as F

from .cache import capture_private_kv, promote
from .greedy import decide
from .packet import build_call, promotion_plan, require_identity_repair


class BootstrapReady(Exception):
    def __init__(self, state):
        self.state = state


def bootstrap_function(function):
    """Reuse exact pinned initialization/normal call; stop BEFORE any commit."""
    original = inspect.unwrap(function)
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    tree.body[0].decorator_list = []
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == 'model']
    calls.sort(key=lambda n: n.lineno)
    if len(calls) != 2:
        raise ValueError('pinned two-call generator changed')
    calls[0].keywords.append(ast.keyword(arg='focus_head_rows',
        value=ast.parse('(0, block_m)', mode='eval').body))
    count = [0]

    class Inject(ast.NodeTransformer):
        def visit_Assign(self, node):
            if node.value is calls[0]:
                count[0] += 1
                extra = ast.parse('raise _bootstrap_ready(dict(locals()))').body[0]
                return [node, ast.copy_location(extra, node)]
            return self.generic_visit(node)
    tree = Inject().visit(tree)
    if count != [1]:
        raise ValueError('bootstrap call is not a unique assignment')
    ast.fix_missing_locations(tree)
    scope = dict(original.__globals__, _bootstrap_ready=BootstrapReady)
    exec(compile(tree, original.__code__.co_filename+':v7_bootstrap', 'exec'), scope)
    return functools.update_wrapper(torch.no_grad()(scope[original.__name__]), original)


def select_tracked(known, changed, count):
    """Every changed identity first; deterministic most-recent background fill."""
    if len(set(known)) != len(known) or not set(changed).issubset(known):
        raise ValueError('invalid clean background')
    result = list(dict.fromkeys(changed))
    if len(result) > count:
        raise ValueError('changed-identity repair exceeds query capacity')
    if len(result) == count:
        return result
    for position in reversed(known):
        if position not in result:
            result.append(position)
        if len(result) == count:
            break
    if len(result) != count:
        raise ValueError('insufficient clean context')
    return result


@dataclass
class Horizon:
    limit: int
    fixed: bool = False
    discovered: int | None = None

    def observe(self, positions, tokens, eos):
        if self.discovered is None:
            found = [p for p, v in zip(positions, tokens) if v == eos]
            if found:
                self.discovered = max(found)
                if not self.fixed:
                    self.limit = self.discovered+1


def render(tokenizer, raw_ids, stop_tokens=()):
    # Match the pinned official two-decode/one-retokenization path.
    text = tokenizer.decode(raw_ids, skip_special_tokens=False)
    for stop in stop_tokens:
        if stop in text:
            text = text.split(stop)[0]
    return tokenizer.decode(tokenizer(text)['input_ids'], skip_special_tokens=True)


def valid_predictions(logits, mask_id):
    # A committed MASK is not progress. Other special tokens, including EOS,
    # remain eligible. This is explicit research semantics, not a baseline edit.
    logits[:, mask_id] = -torch.inf
    probability = logits.double().softmax(-1)
    values, tokens = probability.max(-1)
    return values.tolist(), tokens.tolist()


@torch.no_grad()
def generate(model, tokenizer, external, ids, *, length=256, fixed_work=False):
    raw = model.model_ref
    mask_id, eos_id = 126336, 126081
    normalized = [None]
    def save_norm(_module, _inputs, value):
        normalized[0] = value.detach()
    handle = raw.model.transformer.ln_f.register_forward_hook(save_norm)
    try:
        try:
            bootstrap_function(external)(model, [torch.tensor(ids, device=raw.device)], [len(ids)],
                1, [None], [0], gen_length=length, block_length=32, threshold=.9, gamma=.8,
                track_num=4, mask_num=4, verify=True, tokenizer=tokenizer, stop_tokens=[])
        except BootstrapReady as ready:
            state = ready.state
        else:
            raise AssertionError('bootstrap did not stop before commit')
    finally:
        handle.remove()
    if normalized[0] is None or normalized[0].shape[1] != state['query_pos_flat'].numel():
        raise AssertionError('dated proposal source is not full normalized bootstrap')
    warm_hidden = normalized[0][0]
    row_for_position = {p: row for row, p in enumerate(state['query_pos_flat'].tolist())}
    canvas = state['x']
    canvas_cpu = canvas.tolist()
    maximum, key_length = int(state['max_length']), int(state['seqlen_k'][0])
    prompt_length = len(ids)
    known = list(range(prompt_length))
    horizon = Horizon(prompt_length+length, fixed=fixed_work)
    proposals = {}
    first_pos = state['query_masked_pos'][0].tolist()
    values, tokens = valid_predictions(state['output'].logits.squeeze(0)[:32], mask_id)
    proposals.update({p: (v, t) for p, v, t in zip(first_pos, values, tokens)})
    ordered = sorted(first_pos, key=lambda p: (-proposals[p][0], p))
    warm_keep = min(16, max(1, sum(proposals[p][0] >= .9 for p in ordered)))
    warm_positions = ordered[:warm_keep]
    warm_tokens = [proposals[p][1] for p in warm_positions]
    canvas[torch.tensor(warm_positions, device=raw.device)] = torch.tensor(warm_tokens, device=raw.device)
    for p, v in zip(warm_positions, warm_tokens):
        canvas_cpu[p] = v
    known.extend(warm_positions)
    changed = warm_positions
    dirty = tuple(warm_positions) # bootstrap cache still contains MASK identities
    horizon.observe(warm_positions, warm_tokens, eos_id)
    bank = [(b.k_cache, b.v_cache) for b in raw.model.transformer.blocks]
    packets, cold_positions, cold_projection_calls = [], 0, 0

    def refill(positions):
        nonlocal cold_positions, cold_projection_calls
        missing = [p for p in positions if p not in proposals]
        if not missing:
            return
        # Already-paid bootstrap hidden is dated, not fresh future teacher state.
        # Readout/copy cost is part of the caller's complete request timer.
        rows = torch.tensor([row_for_position[p] for p in missing], device=raw.device)
        hidden = warm_hidden.index_select(0, rows)
        if len(missing) < 32:
            hidden = torch.cat((hidden, hidden.new_zeros((32-len(missing), hidden.shape[1]))))
        if raw.config.weight_tying:
            logits = F.linear(hidden, raw.model.transformer.wte.weight)
        else:
            logits = raw.model.transformer.ff_out(hidden)
        if raw.config.scale_logits:
            logits.mul_(1/math.sqrt(raw.config.d_model))
        p, v = valid_predictions(logits[:len(missing)], mask_id)
        proposals.update({pos: (prob, token) for pos, prob, token in zip(missing, p, v)})
        cold_positions += len(missing); cold_projection_calls += 1

    while True:
        remaining = [p for p in range(prompt_length, horizon.limit) if canvas_cpu[p] == mask_id]
        if not remaining:
            break
        if len(packets) >= length:
            raise AssertionError('no-progress loop')
        window = remaining[:32]
        refill(window)
        candidates = sorted(window, key=lambda p: (-proposals[p][0], p))[:16]
        k = len(candidates)
        from focus_v6.audit import Layout
        layout = Layout(k)
        tracked = select_tracked(known, changed, layout.tracked)
        tracked_ids = [canvas_cpu[p] for p in tracked]
        require_identity_repair(dirty, tracked, canvas_cpu, tracked_ids)
        other_known = [p for p in known if p not in tracked]
        selected = set(known+candidates)
        others = [p for p in range(maximum) if p not in selected]
        full = torch.tensor([other_known+tracked+candidates+others], device=raw.device)
        if full.numel() != maximum:
            raise AssertionError('one physical position must have one external version')
        drafts_cpu = [proposals[p][1] for p in candidates]
        draft_ids = state['x_draft']
        draft_ids[torch.tensor(candidates, device=raw.device)] = torch.tensor(drafts_cpu, device=raw.device)
        packet_state = dict(state, full_pos=full, num_decoded=[len(known)], num_verify=k,
                            x=canvas, x_draft=draft_ids, start_layer=[raw.config.n_layers])
        query, positions, lengths, layout, chosen, drafts = build_call(packet_state, k)
        # Verify the real clean query identities, not only the planner's arrays.
        require_identity_repair(dirty, positions[0][:layout.tracked].tolist(),
                                canvas_cpu, query[0,:layout.tracked].tolist())
        with capture_private_kv() as captured:
            output = model(query, use_cache=True, positions=positions, lengths=lengths,
                           focus_head_rows=(layout.clean.start, 3*k))
        clean = output.logits.squeeze(0)[layout.clean]
        audit = output.logits.squeeze(0)[layout.audit]
        clean_p, clean_ids = valid_predictions(clean, mask_id)
        audit[:,mask_id] = -torch.inf
        probability = audit.double().softmax(-1).gather(1,drafts[:,None]).squeeze(1).tolist()
        decision = decide(probability, audit.argmax(-1).tolist(), drafts_cpu, forbidden=(mask_id,))
        if not decision.progress:
            raise AssertionError('ordinary correction failed to make progress')
        if len(captured) != len(bank):
            raise AssertionError('incomplete native private KV')
        rows, destinations, dirty = promotion_plan(layout, candidates, tracked, decision)
        promote(bank, captured, torch.tensor(rows,device=raw.device),
                torch.tensor(destinations,device=raw.device))
        committed = candidates[:decision.progress]
        canvas[torch.tensor(committed,device=raw.device)] = torch.tensor(decision.tokens,device=raw.device)
        for p,v in zip(committed,decision.tokens):
            if canvas_cpu[p] != mask_id:
                raise AssertionError('rewrote a committed token')
            canvas_cpu[p] = v
        proposals.update({p:(v,t) for p,v,t in zip(candidates,clean_p,clean_ids)})
        known.extend(committed); changed = committed
        horizon.observe(committed, decision.tokens, eos_id)
        packets.append(dict(k=k, accepted=decision.accepted, progress=decision.progress,
                            correction=decision.correction, positions=committed,
                            tokens=list(decision.tokens), dirty=list(dirty), queried_rows=64))
    ids_out = canvas_cpu[prompt_length:len(known)] # identical official count-based rendering slice
    first_eos = next((i for i,v in enumerate(canvas_cpu[prompt_length:prompt_length+length]) if v==eos_id), None)
    response = render(tokenizer, ids_out)
    return dict(text=response, raw_token_ids=canvas_cpu[prompt_length:prompt_length+length],
                nfe=1+len(packets), packets=packets, bootstrap_commits=warm_keep,
                cold_proposal_positions=cold_positions, cold_projection_calls=cold_projection_calls,
                committed_count=len(known)-prompt_length, first_eos=first_eos,
                discovered_eos=None if horizon.discovered is None else horizon.discovered-prompt_length,
                truncated=horizon.discovered is None, horizon=horizon.limit-prompt_length,
                fixed_work=fixed_work, sdpa_calls=0, approximation='conditional prefix KV plus dated cold proposals')

"""Private32-layer verifier on immutable, pre-proposal base KV snapshots."""
from dataclasses import dataclass
import math
import torch
from .attention import streaming, dense_reference
from .layout import prefix_union, dependency_closed, cumulative_prefixes


@dataclass
class Prepared:
    plan: object
    ids: torch.Tensor
    positions: torch.Tensor
    mapping: torch.Tensor
    choices: torch.Tensor
    rotary: object
    private_rows: torch.Tensor
    shared_count: int
    clean_mask_count: int


def prepare(plan, rotary, device, mask_id=126336, *, shared_positions=(), shared_tokens=(), clean_masks=False):
    b = plan.count
    shared_positions, shared_tokens = tuple(map(int,shared_positions)),tuple(map(int,shared_tokens))
    t = len(shared_positions)
    clean = b if clean_masks else 0
    if (t != len(shared_tokens) or b+t+clean > 128 or len(set(shared_positions)) != t
            or set(shared_positions) & set(plan.positions)
            or any(p < 0 or p >= plan.cache_length for p in shared_positions)
            or any(v < 0 or v == mask_id for v in shared_tokens)):
        raise ValueError('Unique legal accepted shared context, disjoint from drafts, required')
    ids = torch.tensor(plan.tokens+(mask_id,)*(b*(plan.families-1))+shared_tokens+(mask_id,)*clean,
                       device=device, dtype=torch.long)
    positions = torch.tensor(plan.positions*plan.families+shared_positions+(plan.positions if clean else ()), device=device, dtype=torch.long)
    mapping = torch.full((plan.cache_length,), -1, device=device, dtype=torch.int32)
    mapping[positions[:b]] = torch.arange(b, device=device, dtype=torch.int32)
    if t:
        mapping[positions[b*plan.families:b*plan.families+t]] = torch.arange(b,b+t,device=device,dtype=torch.int32)
    if clean:
        # This original position ALWAYS has exactly one private version: its
        # draft or its draft-free clean MASK, never the old cached base as well.
        mapping[positions[:b]] = -2
    # Shared legal context is refreshed at every layer, but NEVER reads any
    # speculative draft. I/O never become keys. One bank per original position.
    choices = tuple(row+(True,)*t+(tuple(not choice for choice in row) if clean else ()) for row in plan.choices())
    choices += ((False,)*b+(True,)*(t+clean),)*(t+clean)
    choices = torch.tensor(choices, device=device, dtype=torch.bool).contiguous()
    private_rows = torch.tensor(tuple(range(b))+tuple(range(b*plan.families,b*plan.families+t+clean)),
                                device=device,dtype=torch.long)
    return Prepared(plan, ids, positions, mapping, choices, rotary,private_rows,t,clean)


def rotate(x, positions, rotary):
    sine, cosine = (v.index_select(0, positions).float()[:, None] for v in rotary)
    half = x.shape[-1]//2
    value = x.float()
    turn = torch.cat((-value[..., half:], value[..., :half]), -1)
    return (value*cosine+turn*sine).to(x.dtype)


@torch.no_grad()
def forward(model, prepared, snapshot, *, changed_token=None, attention_reference=False, projection_reference=None):
    plan = prepared.plan
    core = model.model
    if len(snapshot) != len(core.transformer.blocks):
        raise ValueError('Complete pre-proposal layer snapshots required')
    ids = prepared.ids
    if changed_token is not None:
        ids = ids.clone(); ids[0] = changed_token
    x = core.transformer.wte(ids)
    if core.config.input_emb_norm:
        x = x*math.sqrt(core.config.d_model)
    x = core.transformer.emb_drop(x)
    b, h = plan.count, core.config.n_heads
    d = core.config.d_model//h
    if d != 128 or core.config.effective_n_kv_heads != h:
        raise ValueError('The fixed LLaDA8B attention geometry is required')
    for block, (base_k, base_v) in zip(core.transformer.blocks, snapshot):
        xn = block.attn_norm(x)
        if block.q_norm is not None or block.k_norm is not None:
            raise ValueError('Unexpected Q/K normalization in the pinned backbone')
        if projection_reference is not None:
            # Paid same-state operator control, never a cache write. Q/K/V use
            # the pinned native fused projection and its rounding order.
            q,k,v = projection_reference(block,xn,prepared,h,d)
        else:
            q = block.q_proj(xn).view(-1,h,d)
            keys = xn.index_select(0,prepared.private_rows)
            k = block.k_proj(keys).view(-1,h,d)
            v = block.v_proj(keys).view(-1,h,d).contiguous()
            q = rotate(q,prepared.positions,prepared.rotary).contiguous()
            k = rotate(k,prepared.positions.index_select(0,prepared.private_rows),prepared.rotary).contiguous()
        attention = dense_reference if attention_reference else streaming
        att = attention(q, base_k.view(-1, h, d), base_v.view(-1, h, d), k, v,
                        prepared.mapping, prepared.choices).to(x.dtype).flatten(1)
        x = x+block.dropout(block.attn_out(att))
        xn = block.ff_norm(x)
        x = x+block.dropout(block.ff_out(block.act(block.ff_proj(xn))*block.up_proj(xn)))
    normalized = core.transformer.ln_f(x)
    consume = normalized[b:b*plan.families]
    logits = (torch.nn.functional.linear(consume, core.transformer.wte.weight)
              if core.config.weight_tying else core.transformer.ff_out(consume))
    if core.config.scale_logits:
        logits = logits/math.sqrt(core.config.d_model)
    return dict(iso=logits[:b], cross=logits[b:] if plan.cross else None,
                draft_hidden=normalized[:b],shared_hidden=normalized[b*plan.families:])


@torch.no_grad()
def decide(output, plan, *, gamma=.80, eta=.05):
    if not 0 < gamma <= 1 or not 0 <= eta <= math.log(2):
        raise ValueError('Valid probability and JSD thresholds required')
    iso = output['iso']
    if not torch.isfinite(iso).all():
        raise ValueError('Nonfinite isolated logits')
    log_iso = iso.double().log_softmax(-1)
    p = log_iso.exp()
    candidates = torch.tensor(plan.tokens, device=iso.device, dtype=torch.long)
    probability = p.gather(1, candidates[:, None]).flatten()
    matches = iso.argmax(-1) == candidates
    flags = matches & (probability >= gamma)
    js = torch.zeros(plan.count, device=iso.device, dtype=torch.float64)
    cross_match = torch.ones_like(flags)
    if plan.cross:
        cross = output['cross']
        if cross is None or cross.shape != iso.shape or not torch.isfinite(cross).all():
            raise ValueError('Finite paired observer logits required')
        log_cross = cross.double().log_softmax(-1)
        midpoint = torch.logaddexp(log_iso, log_cross)-math.log(2)
        js = ((p*(log_iso-midpoint)).sum(-1)
              +(log_cross.exp()*(log_cross-midpoint)).sum(-1))/2
        cross_match = cross.argmax(-1) == candidates
        flags &= cross_match & (js <= eta)
    accepted = prefix_union(plan.groups, flags.cpu().tolist())
    if not dependency_closed(plan, accepted):
        raise RuntimeError('Accepted isolated result depends on a rejected draft')
    return dict(accepted=list(accepted), probabilities=probability.cpu().tolist(),
                iso_matches=matches.cpu().tolist(), cross_matches=cross_match.cpu().tolist(),
                js_nats=js.cpu().tolist(), combined_flags=flags.cpu().tolist(),
                cumulative_accepted=list(cumulative_prefixes(plan.groups,
                    probability.cpu().tolist(), matches.cpu().tolist(), gamma)),
                dependency_closed=True)

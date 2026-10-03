"""Private32-layer verifier on immutable, pre-proposal base KV snapshots."""
from dataclasses import dataclass
import math
import torch
from .attention import streaming
from .layout import prefix_union, dependency_closed, cumulative_prefixes


@dataclass
class Prepared:
    plan: object
    ids: torch.Tensor
    positions: torch.Tensor
    mapping: torch.Tensor
    choices: torch.Tensor
    rotary: object


def prepare(plan, rotary, device, mask_id=126336):
    b = plan.count
    ids = torch.tensor(plan.tokens+(mask_id,)*(b*(plan.families-1)),
                       device=device, dtype=torch.long)
    positions = torch.tensor(plan.positions*plan.families, device=device, dtype=torch.long)
    mapping = torch.full((plan.cache_length,), -1, device=device, dtype=torch.int32)
    mapping[positions[:b]] = torch.arange(b, device=device, dtype=torch.int32)
    choices = torch.tensor(plan.choices(), device=device, dtype=torch.bool).contiguous()
    return Prepared(plan, ids, positions, mapping, choices, rotary)


def rotate(x, positions, rotary):
    sine, cosine = (v.index_select(0, positions).float()[:, None] for v in rotary)
    half = x.shape[-1]//2
    value = x.float()
    turn = torch.cat((-value[..., half:], value[..., :half]), -1)
    return (value*cosine+turn*sine).to(x.dtype)


@torch.no_grad()
def forward(model, prepared, snapshot, *, changed_token=None):
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
        q = block.q_proj(xn).view(-1, h, d)
        # Only draft rows are private keys; no I/O KV is projected or consumed.
        k = block.k_proj(xn[:b]).view(b, h, d)
        v = block.v_proj(xn[:b]).view(b, h, d).contiguous()
        if block.q_norm is not None or block.k_norm is not None:
            raise ValueError('Unexpected Q/K normalization in the pinned backbone')
        q = rotate(q, prepared.positions, prepared.rotary).contiguous()
        k = rotate(k, prepared.positions[:b], prepared.rotary).contiguous()
        att = streaming(q, base_k.view(-1, h, d), base_v.view(-1, h, d), k, v,
                        prepared.mapping, prepared.choices).flatten(1)
        x = x+block.dropout(block.attn_out(att))
        xn = block.ff_norm(x)
        x = x+block.dropout(block.ff_out(block.act(block.ff_proj(xn))*block.up_proj(xn)))
    normalized = core.transformer.ln_f(x)
    consume = normalized[b:]
    logits = (torch.nn.functional.linear(consume, core.transformer.wte.weight)
              if core.config.weight_tying else core.transformer.ff_out(consume))
    if core.config.scale_logits:
        logits = logits/math.sqrt(core.config.d_model)
    return dict(iso=logits[:b], cross=logits[b:] if plan.cross else None,
                draft_hidden=normalized[:b])


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

"""Packed DAG verification; speculative keys never enter shared background state.

Uses the pinned Flash fused QKV/RoPE operator and the repository's audited
version-selection Triton attention. These conventional operators are credited
in REFERENCES.md; the planner and information-flow layout are separate.
"""
from dataclasses import dataclass
import math
import torch
from focus_dllm.tuning.firebreak.attention import streaming, dense_reference
from focus_dllm.tuning.firebreak.probe import native_projection
from .cache import Ledger


@dataclass
class Rows:
    ids: object
    positions: object
    mapping: object
    choices: object
    rotary: object
    private_rows: object
    projection_table: object
    clean_positions: tuple
    clean_rows: tuple
    consume: object


def prepare(canvas, clean_positions, rotary, device, *, candidate_positions=(), candidate_tokens=(), dag=None, mask_id=126336):
    clean = tuple(map(int, clean_positions)); cp = tuple(map(int, candidate_positions))
    ct = tuple(map(int, candidate_tokens)); b, s, n = len(cp), len(clean), len(canvas)
    if (not clean or len(set(clean)) != s or len(set(cp)) != b or len(ct) != b
            or any(i < 0 or i >= n for i in clean+cp) or not set(cp) <= set(clean)
            or b+s > 128 or 2*b+s > 128):
        raise ValueError('Unique initialized clean background and bounded private geometry required')
    if b and (dag is None or len(dag.parents) != b or any(canvas[i] != mask_id for i in cp)
              or any(v < 0 or v == mask_id for v in ct)):
        raise ValueError('One legal draft per masked candidate with an explicit DAG required')
    positions = cp+cp+clean
    ids = ct+(mask_id,)*b+tuple(canvas[i] for i in clean)
    ancestors = dag.ancestors() if b else ()
    ci = {p: i for i, p in enumerate(cp)}
    choices = []
    for r in range(2*b):
        i = r % b
        allowed = ancestors[i] | ((1 << i) if r < b else 0)
        draft = tuple(bool(allowed & (1 << j)) for j in range(b))
        background = tuple(not draft[ci[p]] if p in ci else True for p in clean)
        choices.append(draft+background)
    choices += [(False,)*b+(True,)*s] * s
    mapping = torch.full((n,), -1, device=device, dtype=torch.int32)
    mapping[torch.tensor(clean, device=device, dtype=torch.long)] = -2
    position_tensor = torch.tensor(positions, device=device, dtype=torch.long)
    m = len(ids)
    return Rows(torch.tensor(ids, device=device, dtype=torch.long), position_tensor,
        mapping, torch.tensor(choices, device=device, dtype=torch.bool).contiguous(), rotary,
        torch.tensor(tuple(range(b))+tuple(range(2*b, m)), device=device, dtype=torch.long),
        torch.tensor([(0,n,i,min(i+64,m)) for i in range(0,m,64)], device=device, dtype=torch.int32),
        clean, tuple(range(2*b,m)), torch.arange(b,2*b,device=device) if b else torch.arange(s,device=device))


def probabilities(logits):
    if not torch.isfinite(logits).all():
        raise ValueError('Nonfinite model logits')
    # Same high-precision probability computation as the pinned greedy sampler.
    return logits.double().softmax(-1)


class Runtime:
    def __init__(self, model, rotary, *, attention_reference=False):
        self.model, self.rotary = model, rotary
        self.reference = attention_reference
        self.attention = streaming
        self.cache = None
        self.ledger = None
        self.nfe = 0

    def _head(self, hidden):
        core = self.model.model
        value = core.transformer.ln_f(hidden)
        logits = (torch.nn.functional.linear(value,core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(value))
        return logits/math.sqrt(core.config.d_model) if core.config.scale_logits else logits

    def _layer(self, block, x, q, k, v, base, rows):
        att = dense_reference if self.reference else self.attention
        a = att(q,base[0],base[1],k,v,rows.mapping,rows.choices).to(x.dtype).flatten(1)
        x = x+block.dropout(block.attn_out(a))
        norm = block.ff_norm(x)
        return x+block.dropout(block.ff_out(block.act(block.ff_proj(norm))*block.up_proj(norm)))

    @torch.no_grad()
    def prefill(self, canvas, consume):
        """Paid full-canvas forward. A request gets a new private cache arena."""
        core = self.model.model; device = core.transformer.wte.weight.device
        n, h = len(canvas), core.config.n_heads; d = core.config.d_model//h
        self.cache, self.ledger, self.nfe = [], Ledger(canvas), 0
        rows = Rows(torch.tensor(canvas,device=device),torch.arange(n,device=device),
            torch.full((n,),-1,device=device,dtype=torch.int32),
            torch.zeros((n,1),device=device,dtype=torch.bool),self.rotary,
            torch.arange(n,device=device),
            torch.tensor([(0,n,i,min(i+64,n)) for i in range(0,n,64)],device=device,dtype=torch.int32),
            tuple(range(n)),tuple(range(n)),torch.tensor(consume,device=device))
        x = core.transformer.emb_drop(core.transformer.wte(rows.ids))
        query = None
        for depth, block in enumerate(core.transformer.blocks,1):
            q,k,v = native_projection(block,block.attn_norm(x),rows,h,d)
            # Base is the current full K/V; the dummy private bank is excluded.
            x = self._layer(block,x,q,k[:1],v[:1],(k,v),rows)
            self.cache.append((k.contiguous(),v.contiguous()))
            if depth == 4:
                query = q.index_select(0,rows.consume).detach()
        self.nfe += 1
        return dict(logits=self._head(x.index_select(0,rows.consume)),query=query,
                    queries=n,private_keys=0)

    @torch.no_grad()
    def run(self, rows, *, capture_layer=4):
        if self.cache is None:
            raise ValueError('Prefill first')
        if not set(self.ledger.dirty()) <= set(rows.clean_positions):
            raise ValueError('Every changed identity needs a fresh clean key version')
        core = self.model.model; h = core.config.n_heads; d = core.config.d_model//h
        ticket = self.ledger.ticket(rows.clean_positions)
        x = core.transformer.emb_drop(core.transformer.wte(rows.ids))
        updates, query = [], None
        b = len(rows.private_rows)-len(rows.clean_positions)
        for depth, (block, base) in enumerate(zip(core.transformer.blocks,self.cache),1):
            q,k,v = native_projection(block,block.attn_norm(x),rows,h,d)
            x = self._layer(block,x,q,k,v,base,rows)
            updates.append((k[b:].contiguous(),v[b:].contiguous()))
            if depth == capture_layer:
                query = q.index_select(0,rows.consume).detach()
        self.nfe += 1
        return dict(logits=self._head(x.index_select(0,rows.consume)),updates=updates,
                    ticket=ticket,query=query,clean_positions=rows.clean_positions,
                    queries=len(rows.ids),private_keys=len(rows.private_rows))

    @torch.no_grad()
    def promote_clean(self, result):
        """Prevalidate all layers before writes. Draft/verification rows absent.

        A CUDA failure invalidates the whole request, rather than reusing a
        partially written arena. This is not a GPU-fault atomicity guarantee.
        """
        self.ledger.validate(result['ticket'])
        updates = result['updates']; p = result['ticket'].positions
        if len(updates) != len(self.cache):
            raise ValueError('Incomplete clean transaction')
        for (k,v),(bk,bv) in zip(updates,self.cache):
            if k.shape != v.shape or k.shape != (len(p),)+tuple(bk.shape[1:]) or k.dtype != bk.dtype or k.device != bk.device:
                raise ValueError('Invalid cache transaction geometry')
        index = torch.tensor(p,device=self.cache[0][0].device)
        # Drift is observed only for refreshed positions, never an oracle label.
        old = self.cache[3][0].index_select(0,index).float()
        new = updates[3][0].float()
        drift = ((new-old).square().sum((1,2))/(old.square().sum((1,2))+1e-8)).sqrt().cpu().tolist()
        for (k,v),(bk,bv) in zip(updates,self.cache):
            bk.index_copy_(0,index,k); bv.index_copy_(0,index,v)
        self.ledger.mark_refreshed(result['ticket'],drift)

    def commit(self, positions, tokens):
        self.ledger.change(tuple(positions),tuple(tokens))

    def check_repaired(self):
        if self.ledger.dirty():
            raise ValueError('Changed identities require a paid clean refresh')

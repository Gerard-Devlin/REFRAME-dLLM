"""Isolate spatial coverage of the live future-query set on teacher states.

Mirrors StaticSupportForward without changing its cache, RoPE or query budget.
Only the choice of live future positions differs. No quality benefit is assumed.
"""
import math

import torch

from .static_support import StaticSupportForward
from .tensor_pruning import MappedRotary


def choose_support(scores, count, policy):
    """Choose exactly count future positions without host-dependent decisions."""
    if scores.ndim != 1 or count < 0 or count > len(scores):
        raise ValueError('Expected a vector and a valid support count')
    if policy not in {'attention', 'uniform', 'stratified'}:
        raise ValueError('Unknown support policy')
    if count == 0:
        return torch.empty(0, dtype=torch.long, device=scores.device)
    if policy == 'attention':
        return scores.topk(count, sorted=False).indices
    groups = torch.arange(count, device=scores.device)
    if policy == 'uniform':
        return (2 * groups + 1) * len(scores) // (2 * count)
    # One most-attended position per equal contiguous interval. Padding is
    # masked so that ragged intervals cannot select a neighbour's position.
    starts = groups * len(scores) // count
    ends = (groups + 1) * len(scores) // count
    positions = starts[:, None] + torch.arange((len(scores)+count-1)//count,
                                               device=scores.device)
    values = scores[positions.clamp_max(len(scores)-1)]
    values = values.masked_fill(positions >= ends[:, None], -torch.inf)
    return starts + values.argmax(-1)


class CoverageSupportForward(StaticSupportForward):
    def __init__(self, model, config, support_count=32, policy='attention'):
        super().__init__(model, config, support_count=support_count)
        if policy not in {'attention', 'uniform', 'stratified'}:
            raise ValueError('Unknown support policy')
        self.policy = policy

    @torch.no_grad()
    def body(self, ids, positions, past_key_values=None, **kwargs):
        core = self.model.model
        past_length = past_key_values[0][0].shape[-2]
        target = torch.as_tensor(positions, device=ids.device)
        changed = getattr(self, 'support_source', None) is not self.reference
        hidden = core.transformer.wte(ids)
        if core.config.input_emb_norm:
            hidden = hidden * core.config.d_model**.5
        hidden = core.transformer.emb_drop(hidden)
        for number, block in enumerate(core.transformer.blocks, 1):
            deep = number > self.config.prune_after_layer
            original = block.rotary_emb
            if deep:
                block.rotary_emb = self.support_rotaries[number-1]
            try:
                if number == self.config.prune_after_layer and changed:
                    hidden, capture, _ = self._captured_block(block, hidden,
                                                            past_key_values[number-1], False)
                else:
                    hidden, _ = block(hidden, attention_bias=None,
                        layer_past=(self.support_past[number-1] if deep else past_key_values[number-1]),
                        use_cache=False)
            finally:
                block.rotary_emb = original
            if number == self.config.prune_after_layer:
                if changed:
                    q, k = capture['q'].index_select(-2,target), capture['k']
                    relevance = (torch.matmul(q,k.transpose(-2,-1))/math.sqrt(q.shape[-1])).float().softmax(-1).mean((0,1,2))
                    count = min(self.support_count, ids.shape[1]-32)
                    self.support_scores = relevance[past_length+32:]
                    support = choose_support(self.support_scores, count, self.policy)+32
                    self.kept = torch.cat((torch.arange(32,device=ids.device),support)).sort().values
                    removed = torch.ones(ids.shape[1],dtype=torch.bool,device=ids.device)
                    removed[self.kept] = False
                    dropped = torch.arange(ids.shape[1],device=ids.device)[removed]+past_length
                    frozen = torch.cat((torch.arange(past_length,device=ids.device),dropped))
                    keys = torch.cat((frozen,self.kept+past_length))
                    self.support_past, self.support_rotaries = [], []
                    for index,(layer,pair) in enumerate(zip(core.transformer.blocks,self.reference)):
                        self.support_past.append(tuple(t.index_select(-2,frozen) for t in pair))
                        rotary = MappedRotary(layer.rotary_emb,self.kept,ids.shape[1],past_length,keys)
                        rotary.register_buffer('frozen_rotated_key',
                            self.rotated_reference[index].index_select(-2,frozen),persistent=False)
                        self.support_rotaries.append(rotary)
                    self.support_source = self.reference
                hidden = hidden.index_select(1,self.kept)
        hidden = core.transformer.ln_f(hidden.index_select(1,target))
        logits = (torch.nn.functional.linear(hidden,core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(hidden))
        if core.config.scale_logits:
            logits.mul_(1/math.sqrt(core.config.d_model))
        return logits

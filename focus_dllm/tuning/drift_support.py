"""Isolate support selection from already-computed shallow state changes.

This component ablation is not an online quality or acceleration claim.
It uses the native block warm's cut-layer hidden states and the current
cut-layer states; no future teacher result enters the selection rule.
"""
from contextlib import contextmanager
import math

import torch

from .static_support import StaticSupportForward
from .tensor_pruning import MappedRotary


@contextmanager
def capture_cut_hidden(model, layer):
    records = []
    def observe(_module, _args, output):
        records.append(output[0].detach().clone())
    handle = model.model.transformer.blocks[layer-1].register_forward_hook(observe)
    try:
        yield records
    finally:
        handle.remove()


def support_priority(attention, current, reference, policy):
    if attention.ndim != 1 or current.ndim != 2 or current.shape != reference.shape:
        raise ValueError('Expected aligned future-position attention and hidden states')
    if len(attention) != current.shape[0] or policy not in {'attention','drift','attention_drift'}:
        raise ValueError('Unsupported policy or unaligned position count')
    if policy == 'attention':
        return attention
    difference = (current.float()-reference.float()).square().mean(-1).sqrt()
    relative = difference/reference.float().square().mean(-1).sqrt().clamp_min(1e-6)
    return relative if policy == 'drift' else attention*relative


class DriftSupportForward(StaticSupportForward):
    def __init__(self, model, config, support_count=32, policy='attention_drift'):
        super().__init__(model, config, support_count=support_count)
        if policy not in {'attention','drift','attention_drift'}:
            raise ValueError('Unknown support policy')
        self.policy = policy
        self.cut_reference = None

    @torch.no_grad()
    def body(self, ids, positions, past_key_values=None, **kwargs):
        core = self.model.model
        past_length = past_key_values[0][0].shape[-2]
        target = torch.as_tensor(positions,device=ids.device)
        changed = getattr(self,'support_source',None) is not self.reference
        hidden = core.transformer.wte(ids)
        if core.config.input_emb_norm:
            hidden = hidden*core.config.d_model**.5
        hidden = core.transformer.emb_drop(hidden)
        for number, block in enumerate(core.transformer.blocks,1):
            deep = number > self.config.prune_after_layer
            original = block.rotary_emb
            if deep:
                block.rotary_emb = self.support_rotaries[number-1]
            try:
                if number == self.config.prune_after_layer and changed:
                    hidden,capture,_ = self._captured_block(block,hidden,past_key_values[number-1],False)
                else:
                    hidden,_ = block(hidden,attention_bias=None,
                        layer_past=(self.support_past[number-1] if deep else past_key_values[number-1]),use_cache=False)
            finally:
                block.rotary_emb = original
            if number == self.config.prune_after_layer:
                if changed:
                    q,k = capture['q'].index_select(-2,target),capture['k']
                    relevance = (torch.matmul(q,k.transpose(-2,-1))/math.sqrt(q.shape[-1])).float().softmax(-1).mean((0,1,2))
                    count = min(self.support_count,ids.shape[1]-32)
                    scores = relevance[past_length+32:]
                    if self.policy != 'attention':
                        if self.cut_reference is None or self.cut_reference.shape[1] != past_length+ids.shape[1]:
                            raise ValueError('Cut reference must come from the same native block warm')
                        scores = support_priority(scores,hidden[0,32:],self.cut_reference[0,past_length+32:],self.policy)
                    self.kept = torch.cat((torch.arange(32,device=ids.device),scores.topk(count,sorted=False).indices+32)).sort().values
                    removed = torch.ones(ids.shape[1],dtype=torch.bool,device=ids.device)
                    removed[self.kept] = False
                    dropped = torch.arange(ids.shape[1],device=ids.device)[removed]+past_length
                    frozen = torch.cat((torch.arange(past_length,device=ids.device),dropped))
                    keys = torch.cat((frozen,self.kept+past_length))
                    self.support_past,self.support_rotaries = [],[]
                    for index,(layer,pair) in enumerate(zip(core.transformer.blocks,self.reference)):
                        self.support_past.append(tuple(t.index_select(-2,frozen) for t in pair))
                        rotary = MappedRotary(layer.rotary_emb,self.kept,ids.shape[1],past_length,keys)
                        rotary.register_buffer('frozen_rotated_key',self.rotated_reference[index].index_select(-2,frozen),persistent=False)
                        self.support_rotaries.append(rotary)
                    self.support_source = self.reference
                hidden = hidden.index_select(1,self.kept)
        hidden = core.transformer.ln_f(hidden.index_select(1,target))
        logits = (torch.nn.functional.linear(hidden,core.transformer.wte.weight)
                  if core.config.weight_tying else core.transformer.ff_out(hidden))
        if core.config.scale_logits:
            logits.mul_(1/math.sqrt(core.config.d_model))
        return logits

"""Shared head-shape control. Padding cost is included, never an exact theorem."""
import math

import torch
import torch.nn.functional as F


def pad_rows(hidden,minimum):
    if minimum<0:
        raise ValueError('Negative head row budget')
    return F.pad(hidden,(0,0,0,max(0,minimum-hidden.shape[1])))


def project(model,hidden,minimum=32):
    count=hidden.shape[1]
    core,tr=model.model,model.model.transformer
    padded=pad_rows(hidden,minimum)
    logits=(F.linear(padded,tr.wte.weight) if core.config.weight_tying else tr.ff_out(padded))[:,:count]
    if core.config.scale_logits:
        logits=logits*(1/math.sqrt(core.config.d_model))
    return logits


@torch.no_grad()
def selected_forward(model,ids,targets,*,minimum=32,**kwargs):
    targets=torch.as_tensor(targets,device=ids.device,dtype=torch.long)
    if targets.ndim!=1 or targets.numel()==0:
        raise ValueError('Nonempty head positions required')
    # Normalize the original full hidden canvas; compact only its head input.
    norm=model.model.transformer.ln_f
    handle=norm.register_forward_hook(lambda _m,_i,value:
        pad_rows(value.index_select(1,targets),minimum))
    try:
        output=model(ids,**kwargs)
        output.logits=output.logits[:,:targets.numel()]
        return output
    finally:
        handle.remove()

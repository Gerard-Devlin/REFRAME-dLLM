"""Read-only event-frontier diagnostics; no asynchronous decoder or reuse rule."""
from contextlib import AbstractContextManager
import math
import numpy as np
import torch


def select_edges(trace):
    """Three distinct within-block consecutive real-state edges, by time only."""
    eligible=[i for i in range(len(trace)-1) if trace[i]['block']==trace[i+1]['block']]
    if len(eligible)<3:raise ValueError('Need three distinct within-block edges')
    return [eligible[0],eligible[len(eligible)//2],eligible[-1]]


def relative_change(old,new):
    if old.shape!=new.shape:raise ValueError('Shape changed')
    delta=new.float()-old.float()
    absolute=delta.norm(dim=-1)
    denominator=old.float().norm(dim=-1).clamp_min(1e-12)
    exact=(old==new).all(dim=-1)
    return absolute,absolute/denominator,exact


def rankcorr(x,y):
    def ranks(z):
        z=np.asarray(z,dtype=float);order=np.argsort(z,kind='stable');out=np.empty(len(z),float)
        i=0
        while i<len(z):
            j=i+1
            while j<len(z) and z[order[j]]==z[order[i]]:j+=1
            out[order[i:j]]=(i+j-1)/2;i=j
        return out
    if len(x)<2 or np.ptp(x)==0 or np.ptp(y)==0:return None
    return float(np.corrcoef(ranks(x),ranks(y))[0,1])


def frontier_metrics(actual,predictor,energy,mandatory,fraction=.25,threshold=.01):
    """Fixed budget: required changed input positions plus ranked extra positions."""
    actual=np.asarray(actual);predictor=np.asarray(predictor);energy=np.asarray(energy)
    mandatory=np.asarray(mandatory,bool)
    n=len(actual);budget=max(int(math.ceil(n*fraction)),int(mandatory.sum()))
    chosen=mandatory.copy();remaining=np.flatnonzero(~chosen)
    order=remaining[np.argsort(-predictor[remaining],kind='stable')]
    chosen[order[:budget-int(chosen.sum())]]=True
    changed=actual>threshold;outside=~chosen
    mass=float(energy.sum())
    count=int(changed.sum())
    return dict(coverage=float(chosen.mean()),significant_recall=(float((changed&chosen).sum()/count) if count else None),
        omitted_significant=int((changed&outside).sum()),
        delta_energy_captured=(float(energy[chosen].sum()/mass) if mass else 1.),
        omitted_max_relative=float(actual[outside].max()) if outside.any() else 0.,
        omitted_p95_relative=float(np.percentile(actual[outside],95)) if outside.any() else 0.,
        spearman_unchanged_inputs=rankcorr(predictor[~mandatory],actual[~mandatory]))


class Capture(AbstractContextManager):
    """Observe native block inputs/outputs and post-RoPE Q/K/V without replacing results.

    CPU copies are paid diagnostic overhead. All module hooks and attention methods
    are restored even on exception. Full activations remain in memory only.
    """
    def __init__(self,model):self.model=model;self.rows=[];self.hooks=[];self.saved=[]

    def __enter__(self):
        for block in self.model.model.transformer.blocks:
            row={};self.rows.append(row)
            def pre(module,inputs,_row=row):_row['h']=inputs[0].detach().cpu().clone()
            def post(module,inputs,output,_row=row):_row['out']=output[0].detach().cpu().clone()
            self.hooks.extend([block.register_forward_pre_hook(pre),block.register_forward_hook(post)])
            original=block._scaled_dot_product_attention
            existed='_scaled_dot_product_attention' in block.__dict__
            def observe(q,k,v,attn_mask=None,dropout_p=0.,is_causal=False,_original=original,_row=row):
                if attn_mask is not None or dropout_p or is_causal:raise ValueError('Only native unmasked evaluation')
                value=_original(q,k,v,attn_mask=attn_mask,dropout_p=dropout_p,is_causal=is_causal)
                _row.update(q=q.detach().cpu().clone(),k=k.detach().cpu().clone(),v=v.detach().cpu().clone())
                return value
            block._scaled_dot_product_attention=observe
            self.saved.append((block,original,existed))
        return self

    def __exit__(self,*exc):
        for hook in self.hooks:hook.remove()
        for block,original,existed in self.saved:
            if existed:block._scaled_dot_product_attention=original
            else:delattr(block,'_scaled_dot_product_attention')
        return False


@torch.no_grad()
def influence_columns(q,k,source,weights,device,tile=32):
    """Old attention times source magnitudes, with exact all-key FP32 denominator.

    One query tile is materialized, never the full NxN attention matrix. This is a
    paid optimistic diagnostic: computing these columns still scans ALL old keys
    for ALL queries. It is not a cheap deployed predictor or a Flash kernel time.
    weights has [heads, source-count, channels], outputs [positions, channels].
    """
    q=q[0].to(device=device,dtype=torch.float32);k=k[0].to(device=device,dtype=torch.float32)
    heads,n,dim=q.shape
    if k.shape!=q.shape:raise ValueError('This diagnostic requires equal Q/K heads and lengths')
    source=torch.as_tensor(source,device=device,dtype=torch.long)
    weights=weights.to(device=device,dtype=torch.float32)
    if weights.shape[:2]!=(heads,len(source)):raise ValueError('Source weights shape mismatch')
    result=[]
    for start in range(0,n,tile):
        scores=torch.matmul(q[:,start:start+tile],k.transpose(-1,-2))/math.sqrt(dim)
        probabilities=torch.softmax(scores,dim=-1).index_select(-1,source)
        result.append(torch.matmul(probabilities,weights).mean(0).cpu())
    return torch.cat(result)

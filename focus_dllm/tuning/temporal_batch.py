"""Offline temporal workload helpers; no drafter or speculative decoder."""
from contextlib import contextmanager
import torch

MASK_ID=126336
BLOCK=32


def reconstruct_states(prompt,trace,final):
    """Recover real teacher inputs from recorded irreversible native releases.

    Final output validates the trace only. No final token is used to create a
    state before its recorded release. These are paid offline oracle inputs.
    """
    canvas=[MASK_ID]*len(final);states=[];targets=[]
    if not trace or [r['call'] for r in trace]!=list(range(len(trace))):
        raise ValueError('Complete consecutive recorded trajectory required')
    for row in trace:
        active=[i for i,t in enumerate(canvas) if t==MASK_ID]
        if not active or active[0]//BLOCK!=row['block']:
            raise ValueError('Native block ordering changed')
        start=row['block']*BLOCK
        if canvas[start:start+BLOCK]!=row['canvas']:
            raise ValueError('Recorded local input does not match reconstructed state')
        states.append(list(prompt)+list(canvas))
        targets.append(list(range(len(prompt)+start,len(prompt)+start+BLOCK)))
        if len(row['commit_positions'])!=len(row['commit_values']) or not row['commit_positions']:
            raise ValueError('No complete recorded release action')
        for p,v in zip(row['commit_positions'],row['commit_values']):
            if p//BLOCK!=row['block'] or canvas[p]!=MASK_ID or v==MASK_ID:
                raise ValueError('Illegal release/rollback/cross-block action')
            canvas[p]=v
    if canvas!=final or MASK_ID in canvas:
        raise ValueError('Incomplete or incorrect reconstructed teacher result')
    return states,targets


def window_indices(count,width=16):
    if count<width or width<1:raise ValueError('Insufficient real states; never duplicate to fill batch')
    start=min(count//3,count-width)
    return list(range(start,start+width))


def choose_cached_window(windows):
    eligible=[w for w in windows if w['ordinary_total']>0]
    if not eligible:return None
    # Availability alone; never choose a block by timing, agreement or accuracy.
    return min(eligible,key=lambda w:(-w['ordinary_total'],w['block']))


@contextmanager
def compact_head(model,targets):
    """Full Transformer; only the final head rows change. Not bitwise theorem."""
    if targets.ndim!=2 or targets.shape[1]!=BLOCK:
        raise ValueError('One complete current-block head per batch row required')
    norm=model.model.transformer.ln_f
    def hook(_module,_inputs,hidden):
        if hidden.shape[0]!=targets.shape[0]:raise ValueError('Head batch mismatch')
        return hidden.gather(1,targets[:,:,None].expand(-1,-1,hidden.shape[-1]))
    handle=norm.register_forward_hook(hook)
    try:yield
    finally:handle.remove()


def extract_block(logits,targets):
    return logits.gather(1,targets[:,:,None].expand(-1,-1,logits.shape[-1]))


def summary_times(values):
    import numpy as np
    return dict(raw_seconds=values,median_seconds=float(np.median(values)),
                p95_seconds=float(np.percentile(values,95)))

import time
import torch
from .core import BLOCK, SMALL, MASK, EOS, OPTIONS, apply_action, digest

def synchronize():
    if torch.cuda.is_available(): torch.cuda.synchronize()

def choose(logits,canvas,start,threshold=.90):
    probs=logits.softmax(-1); tokens=probs.argmax(-1)
    conf=probs.gather(-1,tokens[:,None]).squeeze(-1)
    mask=torch.tensor([t==MASK for t in canvas[start:start+SMALL]],device=logits.device)
    conf=conf.masked_fill(~mask,-torch.inf)
    selected=(conf>threshold); selected[conf.argmax()]=True; selected &= mask
    return [[start+i,int(tokens[i])] for i in selected.nonzero().flatten().tolist()]

class Observer:
    """Teacher remains the only executor; observation cannot commit tokens."""
    def __init__(self,model,threshold=.90):
        self.model=model; self.threshold=threshold; self.calls=[]; self.history=[]
        self.pending=None; self.expected=None
    def __enter__(self):
        self.forward=self.model.forward; self.sample=self.model.sample_with_top_p
        self.model.forward=self.observe; self.model.sample_with_top_p=self.observe_sample
        return self
    def __exit__(self,*_):
        self.model.forward=self.forward; self.model.sample_with_top_p=self.sample
    def observe(self,*args,**kw):
        ids=kw.get('input_ids',args[0] if args else None)
        canvas=ids[0].tolist()
        if self.expected is not None and canvas!=self.expected:
            raise AssertionError('Native action replay differs from subsequent model input')
        self.expected=None
        update=kw.get('update_past_key_values',False)
        normal=not update and len(canvas)==BLOCK and MASK in canvas
        kind='denoise' if normal else ('prefill' if not self.calls else 'cache_write')
        if kw.get('use_block_cache',False): raise ValueError('Unsupported block cache')
        synchronize(); start_time=time.perf_counter(); out=self.forward(*args,**kw); synchronize()
        row=dict(index=len(self.calls),kind=kind,seconds=time.perf_counter()-start_time,
                 history=[list(c) for c in self.history],canvas=canvas,action=[],
                 eos_present=EOS in canvas, generates_block_first=bool(update))
        self.calls.append(row)
        if update:
            if MASK in canvas: raise ValueError('Dirty cache write')
            self.history.append(canvas)
        if normal:
            pos=canvas.index(MASK); row['start']=pos//SMALL*SMALL; self.pending=row
        return out
    def observe_sample(self,logits,top_p=.95,temperature=0):
        result=self.sample(logits,top_p=top_p,temperature=temperature)
        if self.pending is not None:
            if temperature or logits.shape[1]!=SMALL: raise ValueError('Unsupported sampler')
            row=self.pending
            row['action']=choose(logits[0],row['canvas'],row['start'],self.threshold)
            self.expected=apply_action(row['canvas'],row['action'])
            self.pending=None
        return result

@torch.no_grad()
def generate(model,ids,options=None,observe=False):
    options=options or OPTIONS
    device=next(model.parameters()).device; source=torch.tensor([ids],device=device)
    old=model.forward; counts={'calls':0,'denoise':0,'cache_write':0,'prefill':0}
    limit=2+(options['max_new_tokens']//BLOCK)*(BLOCK+1)
    def counted(*args,**kw):
        counts['calls']+=1
        if counts['calls']>limit: raise RuntimeError('Native generation made insufficient progress')
        if not kw.get('update_past_key_values',False): counts['denoise']+=1
        elif counts['calls']==1: counts['prefill']+=1
        else: counts['cache_write']+=1
        return old(*args,**kw)
    model.forward=counted
    try:
        synchronize(); started=time.perf_counter()
        if observe:
            with Observer(model,options['threshold']) as observer:
                out=model.generate(source,**options)
            calls=observer.calls
        else:
            out=model.generate(source,**options); calls=[]
        synchronize(); elapsed=time.perf_counter()-started
    finally: model.forward=old
    tokens=out[0,len(ids):].tolist()
    return dict(prompt_id=digest(ids),tokens=tokens,seconds=elapsed,counts=counts,calls=calls,
                length_capped=EOS not in tokens,generated_tokens=len(tokens))

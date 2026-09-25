"""Inference-semantic differentiable model; no PEFT or training-recipe switch."""
import copy
import hashlib
import math
import time
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from .core import MODEL_ID, REVISION, CODE_HASH, BLOCK, SMALL, MASK

TARGETS={'q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'}

def snapshot():
    from huggingface_hub import snapshot_download
    path=Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    if hashlib.sha256((path/'modeling.py').read_bytes()).hexdigest()!=CODE_HASH:
        raise ValueError('Official code mismatch')
    return path

def load(path=None):
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(str(path or snapshot()), trust_remote_code=True,
        torch_dtype=torch.bfloat16, local_files_only=True).cuda().eval()
    model.requires_grad_(False)
    return model

class LoRALinear(nn.Module):
    def __init__(self, base, rank=64, alpha=128):
        super().__init__(); self.base=base; self.scale=alpha/rank
        self.a=nn.Parameter(torch.empty(rank,base.in_features,device=base.weight.device,dtype=torch.float32))
        self.b=nn.Parameter(torch.zeros(base.out_features,rank,device=base.weight.device,dtype=torch.float32))
        nn.init.kaiming_uniform_(self.a,a=math.sqrt(5))
    def forward(self,x):
        delta=torch.nn.functional.linear(torch.nn.functional.linear(x,self.a),self.b)
        return self.base(x)+delta*self.scale

def add_lora(model):
    tied=model.get_input_embeddings().weight is model.get_output_embeddings().weight
    found=[]
    for name,layer in list(model.named_modules()):
        if name.split('.')[-1] in TARGETS and isinstance(layer,nn.Linear):
            parent,_,child=name.rpartition('.')
            setattr(model.get_submodule(parent),child,LoRALinear(layer)); found.append(name)
    if len(found)!=7*model.config.num_hidden_layers: raise AssertionError('Incomplete LoRA coverage')
    if tied != (model.get_input_embeddings().weight is model.get_output_embeddings().weight):
        raise AssertionError('Tying changed')
    model.eval()  # train() changes the pinned backbone's RoPE semantics!
    return found

def adapters(model):
    return {k:v.detach().cpu().clone() for k,v in model.named_parameters() if v.requires_grad}

def load_adapters(model,state):
    params={k:v for k,v in model.named_parameters() if v.requires_grad}
    if set(params)!=set(state): raise ValueError('Adapter parameter mismatch')
    with torch.no_grad():
        for name,p in params.items(): p.copy_(state[name])

def merge(model):
    for name,layer in list(model.named_modules()):
        if isinstance(layer,LoRALinear):
            with torch.no_grad():
                layer.base.weight.copy_((layer.base.weight.float()+layer.scale*(layer.b@layer.a)).bfloat16())
            parent,_,child=name.rpartition('.'); setattr(model.get_submodule(parent),child,layer.base)
    model.requires_grad_(False).eval()

class GraphCache:
    """Only the cache API consumed by the pinned model. Never detaches tensors."""
    def __init__(self,entries=None): self.entries=list(entries or [])
    def __len__(self): return len(self.entries)
    def __getitem__(self,i): return self.entries[i]
    def get_seq_length(self,layer_idx=0):
        return self.entries[layer_idx][0].shape[-2] if layer_idx<len(self.entries) else 0
    def update(self,k,v,i,cache_kwargs=None):
        if i<len(self.entries):
            old=self.entries[i]; k=torch.cat((old[0],k),-2); v=torch.cat((old[1],v),-2)
            self.entries[i]=(k,v)
        elif i==len(self.entries): self.entries.append((k,v))
        else: raise ValueError('Nonsequential cache layer')
        return k,v

def enable_checkpointing(model):
    """Checkpoint pure layer calls; mutate live cache only outside recomputation."""
    for layer in model.model.layers:
        original=layer.forward
        def wrapped(hidden, *args, _original=original, **kwargs):
            cache=kwargs.get('past_key_value')
            if not torch.is_grad_enabled() or not isinstance(cache,GraphCache):
                return _original(hidden,*args,**kwargs)
            frozen=list(cache.entries); update=kwargs.get('update_past_key_values',False)
            idx=_original.__self__.self_attn.layer_idx
            def pure(h):
                local=GraphCache(frozen); kw=dict(kwargs,past_key_value=local)
                output=_original(h,*args,**kw)
                if update: return output,*local[idx]
                return output
            output=checkpoint(pure,hidden,use_reentrant=False,preserve_rng_state=True)
            if update:
                h,k,v=output
                if idx==len(cache.entries): cache.entries.append((k,v))
                else: cache.entries[idx]=(k,v)
                return h
            return output
        layer.forward=wrapped

def hidden_for(model,record,return_cache=False):
    """Replay exact native prefix-write shapes, preserving all prefix gradients."""
    cache=GraphCache(); device=next(model.parameters()).device
    for chunk in record['history']:
        model.model(input_ids=torch.tensor([chunk],device=device), use_cache=True,
                    past_key_values=cache,update_past_key_values=True,block_size=BLOCK)
    if return_cache:
        if not cache.entries:
            raise ValueError('Gradient audit requires a nonempty native prefix cache')
        for key,value in cache.entries:
            key.retain_grad(); value.retain_grad()
    out=model.model(input_ids=torch.tensor([record['canvas']],device=device), use_cache=True,
                    past_key_values=cache,update_past_key_values=False,block_size=BLOCK)
    h=out.last_hidden_state
    shifted=torch.cat((h[:,:1],h[:,:-1]),1)[0]
    return (shifted,cache) if return_cache else shifted

def objective(student_h,teacher_h,head,record,branch,chunk_size=4096):
    """Full-vocabulary CE/KL with checkpointed chunks (no truncated top-k KL)."""
    positions=[i for i in range(record['start'],record['start']+SMALL) if record['canvas'][i]==MASK]
    targets=dict(record['first']+record['second'])
    h=student_h[positions]; th=teacher_h[positions].detach()
    def log_normalizer(x):
        pieces=[]
        for start in range(0,head.weight.shape[0],chunk_size):
            w=head.weight[start:start+chunk_size]
            fn=lambda a,w=w: torch.logsumexp(torch.nn.functional.linear(a,w).float(),-1)
            pieces.append(checkpoint(fn,x,use_reentrant=False) if x.requires_grad else fn(x))
        return torch.logsumexp(torch.stack(pieces),0)
    z=log_normalizer(h)
    with torch.no_grad(): tz=log_normalizer(th)
    selected=[j for j,pos in enumerate(positions) if pos in targets]
    ids=[targets[positions[j]] for j in selected]
    # Match autocast GEMM semantics; only target entries are gathered afterward.
    target_logits=torch.nn.functional.linear(h[selected],head.weight[ids]).float().diagonal()
    lp=target_logits-z[selected]; ce=-lp.sum()
    hinge=torch.relu(lp.new_tensor(math.log(.91))-lp).max()
    remain=[j for j,pos in enumerate(positions) if pos not in targets]
    kl=h.sum()*0
    if remain:
        for start in range(0,head.weight.shape[0],chunk_size):
            w=head.weight[start:start+chunk_size]
            def part(a,normal,w=w):
                with torch.no_grad():
                    tl=torch.nn.functional.linear(th[remain],w).float()-tz[remain,None]
                sl=torch.nn.functional.linear(a,w).float()-normal[:,None]
                return (tl.exp()*(tl-sl)).sum()
            kl=kl+checkpoint(part,h[remain],z[remain],use_reentrant=False)
    if branch=='basic': hinge=hinge*0
    return torch.stack((ce,kl,hinge))

class Distiller(nn.Module):
    def __init__(self,student,teacher,branch):
        super().__init__(); self.student=student
        # Frozen teacher is deliberately outside DDP parameter discovery.
        object.__setattr__(self,'teacher',teacher); self.branch=branch
    def forward(self,record):
        torch.cuda.synchronize(); started=time.perf_counter()
        with torch.no_grad(): th=hidden_for(self.teacher,record)
        torch.cuda.synchronize(); self.teacher_seconds=time.perf_counter()-started
        sh=hidden_for(self.student,record)
        return objective(sh,th,self.student.lm_head,record,self.branch)

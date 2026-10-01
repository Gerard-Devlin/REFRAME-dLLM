"""Process-local joint-view adapter for pinned Flash fused Triton primitives.

Only legal common rows update the OWNED cache. The private preflight restores
all original references and symbols on exit. No third-party source is edited.
"""
from contextlib import contextmanager
from dataclasses import dataclass
import inspect
import math

import torch


@dataclass(frozen=True)
class JointMarker:
    common_positions: torch.Tensor
    tiled: bool = False


@contextmanager
def owned_cache(model):
    saved=[]
    try:
        for block in model.model.transformer.blocks:
            saved.append((block,block.k_cache,block.v_cache))
            block.k_cache=block.k_cache.clone();block.v_cache=block.v_cache.clone()
        yield saved
    finally:
        for block,k,v in saved:block.k_cache,block.v_cache=k,v


def joint_block(original, self, x, block_idx, positions, lengths, softmax_scale=None):
    if not isinstance(lengths[10],JointMarker):
        return original(self,x,block_idx,positions,lengths,softmax_scale)
    marker=lengths[10]
    pos,key,rotary,_,_,mask=positions
    table=lengths[1]
    assert x.shape[0]==128 and table.shape==((4,4) if marker.tiled else (1,4)) and lengths[-1] is True
    assert mask.shape==(128,128) and pos.shape==(128,)
    dim=self.config.d_model;heads=self.config.n_heads;hd=dim//heads
    assert hd==128 and self.q_proj.bias is None and self.k_proj.bias is None and self.v_proj.bias is None
    xn=self.attn_norm(x)
    q,k,v,att=(torch.empty_like(x) for _ in range(4))
    if marker.tiled:
        from .joint_kernels import tiled_qkv,tiled_attention
        tiled_qkv[(4,heads)](xn,q,k,v,pos,self.q_proj.weight,self.k_proj.weight,self.v_proj.weight,
            rotary[0],rotary[1],table,self.k_cache,self.v_cache,marker.common_positions.numel(),
            HALF=hd//2,D_MODEL=dim,HEAD_DIM=hd,BLOCK_M=32,BLOCK_D=32,num_warps=4,num_stages=2)
        tiled_attention[(4,heads)](q,k,v,att,self.k_cache,self.v_cache,key,mask,table,
            softmax_scale or 1/math.sqrt(hd),D_MODEL=dim,HEAD_DIM=hd,
            BLOCK_M=32,BLOCK_N=64,LOCAL=128,num_warps=4,num_stages=2)
    else:
        kernels=original.__globals__
        kernels['_flash_verify_qkv_proj_fwd'][(1,heads)](
            xn,q,k,v,pos,self.q_proj.weight,self.k_proj.weight,self.v_proj.weight,
            rotary[0],rotary[1],table,HALF=hd//2,D_MODEL=dim,HEAD_DIM=hd,
            BLOCK_M=128,BLOCK_D=32,num_warps=4,num_stages=2)
        kernels['_flash_verify_attention_fwd'][(1,heads)](
            q,k,v,att,self.k_cache,self.v_cache,key,mask,table,
            softmax_scale or 1/math.sqrt(hd),D_MODEL=dim,HEAD_DIM=hd,
            BLOCK_M=128,BLOCK_N=64,num_warps=4,num_stages=2)
        # These two scatter kernels are paid work, included in call timings.
        count=marker.common_positions.numel()
        self.k_cache.index_copy_(0,marker.common_positions,k[:count])
        self.v_cache.index_copy_(0,marker.common_positions,v[:count])
    x=x+self.dropout(self.attn_out(att));residual=x
    if self._activation_checkpoint_fn is not None:
        raise RuntimeError('Preflight requires the frozen eval backbone without checkpoint hooks')
    xn=self.ff_norm(x)
    x=self.act(self.ff_proj(xn))*self.up_proj(xn)
    return (residual+self.dropout(self.ff_out(x))).unsqueeze(0)


@contextmanager
def joint_adapter(model):
    module=inspect.getmodule(type(model.model.transformer.blocks[0]))
    original=module.flash_fused_elastic_cache
    def dispatch(*args,**kwargs):return joint_block(original,*args,**kwargs)
    module.flash_fused_elastic_cache=dispatch
    try:yield
    finally:module.flash_fused_elastic_cache=original


def prepare_call(plan, original_kwargs, cache_length, device, *, tiled=False):
    query=torch.tensor([plan.tokens],device=device,dtype=torch.long)
    pos=torch.tensor(plan.positions,device=device,dtype=torch.int32)
    common=pos[:plan.common].long().contiguous()
    outside=torch.ones(cache_length,device=device,dtype=torch.bool)
    outside[common]=False
    keys=outside.nonzero().flatten().to(torch.int32).contiguous()
    rows=[[0,len(keys),start,start+32] for start in range(0,128,32)] if tiled else [[0,len(keys),0,plan.tile]]
    table=torch.tensor(rows,device=device,dtype=torch.int32)
    marker=JointMarker(common,tiled)
    oldpos=original_kwargs['positions'];oldlen=original_kwargs['lengths']
    positions=[pos,keys,oldpos[2],[],oldpos[4],plan.mask().to(device).contiguous()]
    # 64 here is HALF the kernel tile, NOT a generation block-size change.
    lengths=[oldlen[0],table,None,None,None,oldlen[5],1,oldlen[7],64,128,marker,True]
    return query,dict(use_cache=True,positions=positions,lengths=lengths)

"""Keep single-state operator shapes while visiting states operator by operator.

This is a numeric/hardware control, not a drafter or an exactness theorem.
Any weight-cache reuse, added launches and concatenations must be measured.
"""
from contextlib import AbstractContextManager

import torch


class NativeRowOps(AbstractContextManager):
    def __init__(self, model):
        self.model=model
        self.saved=[]
        self.stats=dict(linear_rows=0,normalization_rows=0,attention_rows=0)

    def install(self, obj, name, replacement):
        self.saved.append((obj,name,name in obj.__dict__,obj.__dict__.get(name)))
        setattr(obj,name,replacement)

    def __enter__(self):
        if self.saved:
            raise RuntimeError('Do not enter the same operator context twice')
        if self.model.model.config.weight_tying:
            raise ValueError('This control requires the native explicit Linear output head')
        normalizers={'attn_norm','ff_norm','ln_f','q_norm','k_norm'}
        for name,module in self.model.named_modules():
            kind=('linear_rows' if isinstance(module,torch.nn.Linear) else
                  'normalization_rows' if name.split('.')[-1] in normalizers else None)
            if kind is None:
                continue
            original=module.forward

            def row_forward(x,*args,_original=original,_kind=kind,**kwargs):
                if x.ndim<3 or x.shape[0]==1:
                    return _original(x,*args,**kwargs)
                self.stats[_kind]+=x.shape[0]
                return torch.cat([_original(x[i:i+1],*args,**kwargs)
                                  for i in range(x.shape[0])],0)

            self.install(module,'forward',row_forward)
        # Enter AFTER the counting Flash backend, so every actual row attention
        # call goes through its counter. RoPE and native cache ownership stay
        # untouched; rows must already have independent writable Dual caches.
        for block in self.model.model.transformer.blocks:
            original=block._scaled_dot_product_attention

            def row_attention(q,k,v,*args,_original=original,**kwargs):
                if q.shape[0]==1:
                    return _original(q,k,v,*args,**kwargs)
                if k.shape[0]!=q.shape[0] or v.shape[0]!=q.shape[0]:
                    raise ValueError('Expected explicitly packed per-state K/V rows')
                if (args and args[0] is not None) or kwargs.get('attn_mask') is not None:
                    raise ValueError('This control only supports native unmasked Flash attention')
                self.stats['attention_rows']+=q.shape[0]
                return torch.cat([_original(q[i:i+1],k[i:i+1],v[i:i+1],*args,**kwargs)
                                  for i in range(q.shape[0])],0)

            self.install(block,'_scaled_dot_product_attention',row_attention)
        return self

    def __exit__(self,*exc):
        for obj,name,existed,raw in reversed(self.saved):
            if existed:
                setattr(obj,name,raw)
            else:
                delattr(obj,name)
        self.saved.clear()
        return False

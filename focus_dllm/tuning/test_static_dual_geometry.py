"""Actual native CPU attention/RoPE arithmetic and geometry ownership checks."""
import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple
import unittest

import torch
import torch.nn.functional as F

from .static_dual_geometry import StaticDualGeometry

SOURCE=Path(__file__).resolve().parents[2]/'v1/llada/model/modeling_llada.py'
TREE=ast.parse(SOURCE.read_text(encoding='utf-8'))
BLOCK=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='LLaDABlock')
FUNCTION=next(n for n in BLOCK.body if isinstance(n,ast.FunctionDef) and n.name=='attention')
ROTARY=next(n for n in TREE.body if isinstance(n,ast.ClassDef) and n.name=='RotaryEmbedding')
SPACE=dict(torch=torch,nn=torch.nn,Optional=Optional,Tuple=Tuple,ModelConfig=SimpleNamespace,
    BufferCache=dict,einsum=torch.einsum,_non_meta_init_device=lambda _:torch.device('cpu'))
exec(compile(ast.Module(body=[ROTARY,FUNCTION],type_ignores=[]),str(SOURCE),'exec'),SPACE)


class Block(torch.nn.Module):
    attention=SPACE['attention']
    def __init__(self):
        super().__init__()
        self.config=SimpleNamespace(n_heads=2,effective_n_kv_heads=2,rope=True,
            attention_dropout=0.,d_model=8,rope_theta=10000.,rope_full_precision=True,max_sequence_length=12)
        self.q_norm=self.k_norm=None
        self.attn_out=torch.nn.Identity()
        self.rotary_emb=SPACE['RotaryEmbedding'](self.config,{})
        self.eval()
    def _scaled_dot_product_attention(self,q,k,v,**_):
        return F.scaled_dot_product_attention(q,k,v,is_causal=False)


class Checks(unittest.TestCase):
    def mask(self,width=4,start=2):
        mask=torch.zeros(width,12,dtype=torch.bool);mask[:,start:start+3]=True
        return mask
    def model(self):
        block=Block()
        model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
        return model,block
    def test_mask_and_positions_owned(self):
        mask=self.mask();old=mask.clone();geometry=StaticDualGeometry(mask)
        ptr=geometry.addresses
        geometry.set_block(8)
        self.assertEqual(geometry.addresses,ptr)
        self.assertTrue(torch.equal(mask,old))
        self.assertEqual(geometry.indices[0].tolist(),[8,9,10])
    def test_reject_gaps_empty_different_rows(self):
        masks=[torch.zeros(4,12,dtype=torch.bool),self.mask(),self.mask()]
        masks[1][:,3]=False;masks[2][1,2]=False
        for mask in masks:
            with self.assertRaises(ValueError):StaticDualGeometry(mask)
    def test_stale_geometry_rejected(self):
        for target in ('mask','index'):
            g=StaticDualGeometry(self.mask())
            if target=='mask':g.mask[0,2]=False
            else:g.indices[0][0]=0
            with self.assertRaises(ValueError):g.validate()
    def test_wrong_cache_or_input_domain_rejected(self):
        g=StaticDualGeometry(self.mask());q=torch.zeros(4,3,8)
        cache=tuple(torch.zeros(4,2,12,4) for _ in range(2))
        g.check_call(q,cache,g.mask)
        for value in (None,tuple(torch.zeros(1,2,12,4).expand(4,-1,-1,-1) for _ in range(2))):
            with self.assertRaises(ValueError):g.check_call(q,value,g.mask)
        with self.assertRaises(ValueError):g.check_call(q[:1],cache,g.mask)
        with self.assertRaises(ValueError):g.check_call(q,cache,self.mask())
    def test_actual_native_two_block_arithmetic_and_cache(self):
        torch.manual_seed(234)
        for width in (1,2,4):
            model,block=self.model();mask=self.mask(width)
            g=StaticDualGeometry(mask)
            for start in (2,8):
                g.set_block(start)
                q,k,v=[torch.randn(width,3,8,dtype=torch.double) for _ in range(3)]
                cache=tuple(torch.randn(width,2,12,4,dtype=torch.double) for _ in range(2))
                native=tuple(t.clone() for t in cache);modified=tuple(t.clone() for t in cache)
                reference,_=block.attention(q,k,v,layer_past=native,use_cache=True,replace_position=g.mask)
                with g.scope(model):
                    observed,_=block.attention(q,k,v,layer_past=modified,use_cache=True,replace_position=g.mask)
                self.assertTrue(torch.equal(reference,observed))
                self.assertTrue(all(torch.equal(a,b) for a,b in zip(native,modified)))
                outside=~g.mask[0]
                self.assertTrue(all(torch.equal(a[:,:,outside],b[:,:,outside]) for a,b in zip(cache,modified)))
    def test_graph_like_live_index_buffer_after_update(self):
        model,block=self.model();g=StaticDualGeometry(self.mask())
        q=torch.randn(4,2,3,4,dtype=torch.double)
        k=torch.randn(4,2,12,4,dtype=torch.double)
        with g.scope(model):
            method=block.rotary_emb.forward
            old_end=g.end
            _=method(q,k,old_end)
            ptr=g.indices[0].data_ptr();g.set_block(8)
            self.assertEqual(g.indices[0].data_ptr(),ptr)
            self.assertIs(block.rotary_emb._focus_static_geometry,g)
            observed=method(q,k,old_end)
            reference=SPACE['RotaryEmbedding'].forward(block.rotary_emb,q,k,g.end)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(observed,reference)))
    def test_restore_after_exception_and_nested_rejection(self):
        model,block=self.model();g=StaticDualGeometry(self.mask())
        with self.assertRaises(RuntimeError):
            with g.scope(model):
                with g.scope(model):pass
        self.assertNotIn('attention',block.__dict__)
        self.assertNotIn('forward',block.rotary_emb.__dict__)
        self.assertNotIn('_focus_static_geometry',block.__dict__)
    def test_set_block_boundary(self):
        g=StaticDualGeometry(self.mask())
        for value in (-1,10,1.5):
            with self.assertRaises(ValueError):g.set_block(value)


if __name__=='__main__':unittest.main()

"""CPU operator-shape and restoration checks, not model/latency evidence."""
import unittest
from types import SimpleNamespace

import torch
import torch.nn.functional as F

from focus_dllm.tuning.native_row_ops import NativeRowOps


class Block(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj=torch.nn.Linear(4,4).double()
        self.attn_norm=torch.nn.LayerNorm(4).double()
        self.calls=[]

    def _scaled_dot_product_attention(self,q,k,v,**kwargs):
        self.calls.append(q.shape[0])
        return F.scaled_dot_product_attention(q,k,v,**kwargs)


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model=torch.nn.Module()
        self.model.config=SimpleNamespace(weight_tying=False)
        self.model.transformer=torch.nn.Module()
        self.model.transformer.blocks=torch.nn.ModuleList([Block()])
        self.model.transformer.ff_out=torch.nn.Linear(4,6).double()


class Checks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.model=Model()
        self.block=self.model.model.transformer.blocks[0]
        self.input=torch.randn(4,3,4,dtype=torch.double)

    def test_linear_matches_serial_shapes(self):
        oracle=torch.cat([self.block.q_proj(x[None]) for x in self.input])
        with NativeRowOps(self.model) as control:
            value=self.block.q_proj(self.input)
        self.assertTrue(torch.equal(value,oracle))
        self.assertEqual(control.stats['linear_rows'],4)

    def test_norm_matches_serial_shapes(self):
        oracle=torch.cat([self.block.attn_norm(x[None]) for x in self.input])
        with NativeRowOps(self.model) as control:value=self.block.attn_norm(self.input)
        self.assertTrue(torch.equal(value,oracle))
        self.assertEqual(control.stats['normalization_rows'],4)

    def test_attention_counts_actual_rows(self):
        q=torch.randn(4,1,3,4,dtype=torch.double)
        oracle=torch.cat([self.block._scaled_dot_product_attention(x[None],x[None],x[None]) for x in q])
        self.block.calls.clear()
        with NativeRowOps(self.model) as control:value=self.block._scaled_dot_product_attention(q,q,q)
        self.assertTrue(torch.equal(value,oracle))
        self.assertEqual(self.block.calls,[1,1,1,1])
        self.assertEqual(control.stats['attention_rows'],4)

    def test_single_row_is_untouched(self):
        oracle=self.block.q_proj(self.input[:1])
        with NativeRowOps(self.model) as control:value=self.block.q_proj(self.input[:1])
        self.assertTrue(torch.equal(value,oracle))
        self.assertEqual(sum(control.stats.values()),0)

    def test_restore_class_method_presence(self):
        self.assertNotIn('forward',self.block.q_proj.__dict__)
        self.assertNotIn('_scaled_dot_product_attention',self.block.__dict__)
        with NativeRowOps(self.model):self.block.q_proj(self.input)
        self.assertNotIn('forward',self.block.q_proj.__dict__)
        self.assertNotIn('_scaled_dot_product_attention',self.block.__dict__)

    def test_restore_existing_attention_backend(self):
        original=self.block._scaled_dot_product_attention
        self.block._scaled_dot_product_attention=lambda *args,**kwargs:original(*args,**kwargs)
        wrapper=self.block.__dict__['_scaled_dot_product_attention']
        with NativeRowOps(self.model):pass
        self.assertIs(self.block.__dict__['_scaled_dot_product_attention'],wrapper)

    def test_restore_on_exception(self):
        with self.assertRaises(RuntimeError):
            with NativeRowOps(self.model):raise RuntimeError('deliberate')
        self.assertNotIn('forward',self.block.q_proj.__dict__)

    def test_no_tied_head_silent_skip(self):
        self.model.model.config.weight_tying=True
        with self.assertRaises(ValueError):
            with NativeRowOps(self.model):pass
        self.assertNotIn('forward',self.block.q_proj.__dict__)

    def test_no_custom_mask_silent_fallback(self):
        q=torch.randn(4,1,3,4,dtype=torch.double)
        with self.assertRaises(ValueError):
            with NativeRowOps(self.model):self.block._scaled_dot_product_attention(q,q,q,attn_mask=torch.ones(3,3))
        self.assertNotIn('_scaled_dot_product_attention',self.block.__dict__)


if __name__=='__main__':unittest.main(verbosity=2)

import unittest
from types import SimpleNamespace

import torch

from .flash_readout import Readout,ModelReadout,generator


def dummy(model, x_query, verify=True):
    block_m=32;num_verify=3;seqlen_keep=[58]
    query_masked_pos=[torch.arange(128)]
    output=model(x_query)
    first=output.logits.squeeze(0)[0:32] if verify else output.logits.squeeze(0)[0:128]
    output=model(x_query)
    second=output.logits.squeeze(0)[61:64]
    x=torch.zeros(3);pos_decoded_new_j=torch.tensor([1]);x0_decoded_new_j=torch.tensor([2.])
    x[pos_decoded_new_j]=x0_decoded_new_j
    x[pos_decoded_new_j]=x0_decoded_new_j
    return first,second,x


def unfamiliar(x):return x


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.norm=torch.nn.Identity();self.head=torch.nn.Linear(3,5,bias=False)
        self.model=SimpleNamespace(transformer=SimpleNamespace(ln_f=self.norm))
    def forward(self,x,**kw):return SimpleNamespace(logits=self.head(self.norm(x)))


class FlashReadoutTests(unittest.TestCase):
    def test_virtual_indexing_and_empty(self):
        z=torch.randn(1,32,5);r=Readout(z,48,16,64)
        self.assertTrue(torch.equal(r.squeeze(0)[48:64],z[0,:16]))
        with self.assertRaises(ValueError):r[0:16]
        with self.assertRaises(ValueError):r[48:64:2]
        e=Readout(torch.empty(1,0,5),64,0,64)
        self.assertEqual(e[64:64].shape,(0,5))

    def test_projection_keeps_consumed_logits_and_restores_hook(self):
        model=FakeModel();x=torch.randn(1,64,3)
        full=model(x).logits
        proxy=ModelReadout(model,compact=True)
        value=proxy(x,focus_head_rows=(48,16),lengths=[False]*11+[True])
        torch.testing.assert_close(value.logits[48:64],full[0,48:64])
        self.assertEqual(len(model.norm._forward_hooks),0)
        empty=proxy(x,focus_head_rows=(64,0),lengths=[False]*11+[True])
        self.assertEqual(empty.logits[64:64].shape,(0,5))

    def test_restore_after_bad_rows(self):
        model=FakeModel();proxy=ModelReadout(model,compact=True)
        with self.assertRaises(ValueError):proxy(torch.ones(1,64,3),focus_head_rows=(63,3),lengths=[True]*12)
        self.assertEqual(len(model.norm._forward_hooks),0)

    def test_original_function_and_commit_actions_unchanged(self):
        values=[];actions=[]
        def model(x,focus_head_rows):
            values.append(focus_head_rows);return SimpleNamespace(logits=torch.zeros(1,128,5))
        copied=generator(dummy,lambda p,t:actions.append((p.tolist(),t.tolist())))
        _,_,x=copied(model,None)
        self.assertEqual(values,[(0,32),(61,3)])
        self.assertEqual(actions,[([1],[2.]),([1],[2.])])
        self.assertEqual(x.tolist(),[0,2,0]);self.assertNotIn('_focus_action',dummy.__globals__)
        values.clear();generator(dummy)(model,None,False)
        self.assertEqual(values[0],(0,128))

    def test_unknown_layout_fails(self):
        with self.assertRaises(ValueError):generator(unfamiliar)


if __name__=='__main__':unittest.main()

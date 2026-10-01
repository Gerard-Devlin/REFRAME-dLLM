import unittest
import torch
from .flash_statistics import statistics
from .flash_readout import generator


def pinned_shape(model, verify=True):
    block_m=2;query_masked_pos=[torch.arange(2)];num_verify=1;seqlen_keep=[2]
    output=model(None,lengths=[False]);logits=output.logits.squeeze(0)
    logits_masked_j=logits[:2]
    p_masked=F.softmax(logits_masked_j.to(torch.float64),dim=-1)
    x0_p_masked,x0_masked=torch.max(p_masked,dim=-1)
    normal=(x0_p_masked,x0_masked)
    output=model(None,lengths=[True]);logits=output.logits.squeeze(0)
    logits_verify_j=logits[3:4];x_verify_j=torch.tensor([1])
    p_verify=F.softmax(logits_verify_j.to(torch.float64),dim=-1)
    x0_p_verify=p_verify.gather(1,x_verify_j.unsqueeze(1)).view(-1)
    selected=x0_p_verify
    logits_masked_j=logits[:2]
    p_masked=F.softmax(logits_masked_j.to(torch.float64),dim=-1)
    x0_p_masked,x0_masked=torch.max(p_masked,dim=-1)
    return normal,selected,(x0_p_masked,x0_masked)


import torch.nn.functional as F


class Tests(unittest.TestCase):
    def test_normalization_all_vocabulary_ties_and_targets(self):
        z=torch.tensor([[3.,3.,2.,-200.],[0.,0.,0.,0.]],dtype=torch.bfloat16)
        p=z.double().softmax(-1);expected,top=p.max(-1)
        confidence,actual=statistics(z)
        self.assertTrue(torch.equal(confidence,expected));self.assertTrue(torch.equal(actual,top))
        self.assertEqual(actual.tolist(),[0,0])
        target=torch.tensor([2,3]);confidence,actual=statistics(z,target)
        self.assertTrue(torch.equal(confidence,p.gather(1,target[:,None]).flatten()))
        self.assertTrue(torch.equal(actual,top))

    def test_empty_rows_and_invalid_consumer(self):
        conf,top=statistics(torch.empty(0,8))
        self.assertEqual(conf.numel(),0);self.assertEqual(top.numel(),0)
        with self.assertRaises(ValueError):statistics(torch.empty(1,0))
        with self.assertRaises(ValueError):statistics(torch.zeros(2,8),torch.tensor([1]))
        with self.assertRaises(ValueError):statistics(torch.zeros(2,8),torch.tensor([1.,2.]))

    def test_generator_only_replaces_probability_consumers(self):
        from types import SimpleNamespace
        z=torch.tensor([[[1.,2.,0.],[2.,1.,0.],[1.,1.,1.],[1.,3.,0.]]])
        metadata=[]
        def model(*args,**kw):
            metadata.append(kw.pop('focus_head_rows',None));return SimpleNamespace(logits=z)
        baseline=pinned_shape(model);metadata.clear()
        adapted=generator(pinned_shape,statistics=statistics)(model)
        self.assertEqual(metadata,[(0,2),(3,1)])
        for a,b in zip(baseline,adapted):
            if isinstance(a,tuple):self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))
            else:self.assertTrue(torch.equal(a,b))

    def test_unknown_probability_layout_fails_closed(self):
        from .test_flash_readout import dummy
        with self.assertRaises(ValueError):generator(dummy,statistics=statistics)


if __name__=='__main__':unittest.main()

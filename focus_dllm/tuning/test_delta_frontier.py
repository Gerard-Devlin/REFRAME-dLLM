import unittest
from types import SimpleNamespace
import numpy as np
import torch
from .delta_frontier import Capture,frontier_metrics,influence_columns,rankcorr,relative_change,select_edges


class Tests(unittest.TestCase):
    def test_edges_do_not_cross_boundary_or_repeat(self):
        self.assertEqual(select_edges([{'block':b} for b in [0,0,1,1,1,2,2]]),[0,3,5])
        with self.assertRaises(ValueError):select_edges([{'block':b} for b in [0,1,1]])

    def test_relative_exact_vs_small(self):
        x=torch.tensor([[1.,2.],[1.,0.]])
        a,d,e=relative_change(x,x+torch.tensor([[0.,0.],[.001,0.]]))
        self.assertEqual(e.tolist(),[True,False]);self.assertAlmostEqual(d[1].item(),.001,places=6)

    def test_fixed_coverage_and_mandatory(self):
        result=frontier_metrics([1,.1,.01,0],[0,3,2,1],[1,.01,.0001,0],[True,False,False,False],fraction=.5)
        self.assertEqual(result['coverage'],.5);self.assertEqual(result['omitted_significant'],0)
        self.assertGreater(result['delta_energy_captured'],.99)
        self.assertEqual(rankcorr([0,0,0],[1,2,3]),None)
        self.assertAlmostEqual(rankcorr([1,1,2],[2,2,4]),1)

    def test_attention_columns_reference_and_no_mutation(self):
        torch.manual_seed(7)
        q=torch.randn(1,2,5,4);k=torch.randn_like(q);weights=torch.randn(2,2,3)
        before=[t.clone() for t in (q,k,weights)]
        actual=influence_columns(q,k,[0,3],weights,'cpu',tile=2)
        p=torch.softmax(q[0]@k[0].transpose(-1,-2)/2,dim=-1)[:,:,[0,3]]
        torch.testing.assert_close(actual,(p@weights).mean(0))
        for old,new in zip(before,(q,k,weights)):self.assertTrue(torch.equal(old,new))

    def test_hooks_restored_on_exception(self):
        class Block(torch.nn.Module):
            def _scaled_dot_product_attention(self,q,k,v,**kw):return q+k+v
            def forward(self,x):
                h=x[:,None];a=self._scaled_dot_product_attention(h,h,h)
                return a[:,0],None
        block=Block();model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
        x=torch.randn(1,3,4);expected=block(x)[0]
        try:
            with Capture(model) as capture:
                self.assertTrue(torch.equal(block(x)[0],expected));self.assertEqual(set(capture.rows[0]),{'h','out','q','k','v'})
                raise RuntimeError('test')
        except RuntimeError:pass
        self.assertNotIn('_scaled_dot_product_attention',block.__dict__)
        self.assertFalse(block._forward_hooks);self.assertFalse(block._forward_pre_hooks)
        self.assertTrue(torch.equal(block(x)[0],expected))


if __name__=='__main__':unittest.main()

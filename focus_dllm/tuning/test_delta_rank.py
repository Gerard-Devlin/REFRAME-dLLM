import unittest
from types import SimpleNamespace

import numpy as np
import torch

from .delta_rank import (CaptureComponents, audit_legacy_row_energy, full_spectrum, remove_energetic_rows,
                         residual_audit, spectrum_summary)


class Tests(unittest.TestCase):
    def test_known_spectrum(self):
        result, energy = full_spectrum(torch.diag(torch.tensor([4.,2.,1.], dtype=torch.float64)))
        np.testing.assert_allclose(energy, [16,4,1], atol=1e-12)
        self.assertEqual([result[k] for k in ('r90','r95','r99')], [2,2,3])
        self.assertAlmostEqual(result['stable_rank'],21/16)

    def test_rectangular_matches_full_svd(self):
        torch.manual_seed(21)
        for shape in ((9,17),(17,9)):
            x=torch.randn(*shape,dtype=torch.float64)
            before=x.clone();result,energy=full_spectrum(x)
            np.testing.assert_allclose(energy,torch.linalg.svdvals(x).square().numpy(),rtol=1e-12,atol=1e-12)
            self.assertTrue(torch.equal(before,x))
            self.assertLess(result['trace_relative_error'],1e-12)

    def test_rank_one_and_zero(self):
        x=torch.arange(1,10,dtype=torch.float64)[:,None]@torch.arange(1,15,dtype=torch.float64)[None]
        result,_=full_spectrum(x)
        self.assertEqual(result['r99'],1)
        for shape in ((7,4),(0,4)):
            result,_=full_spectrum(torch.zeros(shape))
            self.assertEqual(result['r99'],0);self.assertEqual(result['stable_rank'],0)

    def test_sparse_plus_low_rank_tail(self):
        x=torch.ones(8,6,dtype=torch.float64)
        x[0]=torch.arange(1,7,dtype=torch.float64)*20
        x[1]=torch.arange(6,0,-1,dtype=torch.float64)*20
        tail,selection=remove_energetic_rows(x)
        self.assertEqual(set(selection['removed_positions']),{0,1})
        self.assertEqual(full_spectrum(tail)[0]['r99'],1)
        self.assertLess(selection['tail_energy_fraction'],.01)

    def test_tail_ties_keep_original_order(self):
        x=torch.eye(8)
        tail,selection=remove_energetic_rows(x)
        self.assertEqual(selection['removed_positions'],[0,1])
        self.assertTrue(torch.equal(tail,x[2:]))
        self.assertAlmostEqual(selection['tail_energy_fraction'],.75)

    def test_invalid_spectra_and_tail(self):
        for energy in ([1,-1],[1,np.nan],[1,2]):
            with self.assertRaises(ValueError):spectrum_summary(energy,2,2)
        with self.assertRaises(ValueError):full_spectrum(torch.tensor([[float('nan')]]))
        with self.assertRaises(ValueError):remove_energetic_rows(torch.ones(2,2),1.1)

    def test_bf16_rounding_audit(self):
        torch.manual_seed(8)
        states=[]
        for _ in range(2):
            state={k:torch.randn(1,5,4).bfloat16() for k in ('input','attention','mlp')}
            state['hidden']=(state['input']+state['attention'])+state['mlp'];states.append(state)
        result=residual_audit(*states)
        self.assertTrue(result['bf16_residual_reconstruction_bitwise'])
        self.assertGreater(result['delta_rounding_energy_fraction'],0)
        states[1]['hidden'][0,0,0]+=1
        with self.assertRaises(AssertionError):residual_audit(*states)

    def test_legacy_energy_audit_keeps_original_arithmetic(self):
        torch.manual_seed(77)
        old=torch.randn(8,4096).bfloat16();new=torch.randn_like(old)
        saved=(new.float()-old.float()).norm(dim=-1).square().tolist()
        result=audit_legacy_row_energy(old,new,saved)
        self.assertTrue(result['original_fp32_row_energy_bitwise_equal'])
        self.assertGreater(result['fp32_vs_fp64_row_energy_max_relative_gap'],0)
        saved[0]+=1
        with self.assertRaises(AssertionError):audit_legacy_row_energy(old,new,saved)

    def test_component_hooks_do_not_change_output_and_cleanup(self):
        class Block(torch.nn.Module):
            def __init__(self):
                super().__init__();self.attn_out=torch.nn.Linear(4,4);self.ff_out=torch.nn.Linear(4,4)
            def forward(self,x):
                y=x+self.attn_out(x);return y+self.ff_out(y),None
        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__();self.block=Block()
                self.model=SimpleNamespace(transformer=SimpleNamespace(blocks=[self.block]))
        model=Model().eval();x=torch.randn(1,5,4);expected=model.block(x)[0]
        with self.assertRaisesRegex(RuntimeError,'cleanup'):
            with CaptureComponents(model) as capture:
                self.assertTrue(torch.equal(expected,model.block(x)[0]))
                self.assertEqual(set(capture.rows[0]),{'input','hidden','attention','mlp'})
                residual_audit(capture.rows[0],capture.rows[0])
                raise RuntimeError('cleanup')
        for module in (model.block,model.block.attn_out,model.block.ff_out):
            self.assertFalse(module._forward_hooks);self.assertFalse(module._forward_pre_hooks)

    def test_training_rejected_without_hooks(self):
        with self.assertRaises(ValueError):
            with CaptureComponents(torch.nn.Linear(2,2)):pass

    def test_single_changed_key_value_formula_unchanged_queries(self):
        # Real-arithmetic identity does not extend to changed queries or deep rank.
        torch.manual_seed(17)
        q,k,v=[torch.randn(7,5,dtype=torch.float64) for _ in range(3)]
        knew,vnew=k.clone(),v.clone();j=3;knew[j]+=1;vnew[j]-=.7
        scores=q@k.T/np.sqrt(5);prob=scores.softmax(-1);old=prob@v
        r=((q@knew.T/np.sqrt(5))-scores)[:,j].exp()
        incremental=(old+prob[:,j,None]*(r[:,None]*vnew[j]-v[j]))/(1+prob[:,j]*(r-1))[:,None]
        torch.testing.assert_close(incremental,(q@knew.T/np.sqrt(5)).softmax(-1)@vnew,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from .joint_views import layout,label_reachability,public_cache_write_positions,pending_candidates
from .joint_engine import owned_cache,joint_adapter,prepare_call


def plan(search=3):
    return layout([9,10,11,12],[126336]*4,[0,8],[3,4],list(range(9,9+search)),
                  [5]*search,cache_length=64)


class JointViewTests(unittest.TestCase):
    def test_no_speculative_label_can_reach_public_rows(self):
        for search in (0,1,2,3):
            value=plan(search)
            paths=label_reachability(value)
            self.assertFalse(bool(paths[:value.common].any()))

    def test_own_and_future_labels_never_reach_verification(self):
        value=plan();paths=label_reachability(value)
        self.assertFalse(bool(torch.triu(paths[value.verify_begin:value.verify_begin+value.search]).any()))
        self.assertTrue(bool(paths[value.verify_begin+2,0]))
        self.assertTrue(bool(paths[value.data_begin+2,2]))

    def test_root_has_exact_common_attention_context(self):
        value=plan();mask=value.mask()
        self.assertTrue(torch.equal(mask[value.proposal_rows[0]],mask[value.verify_begin]))
        self.assertEqual(value.tokens[value.verify_begin],value.tokens[value.proposal_rows[0]])
        self.assertEqual(value.positions[value.verify_begin],value.positions[value.proposal_rows[0]])

    def test_private_rows_and_padding_are_never_public_cache_writes(self):
        value=plan();writes=public_cache_write_positions(value)
        self.assertEqual(writes,(9,10,11,12,0,8))
        self.assertFalse(bool(value.mask()[:,value.verify_begin:].any()))
        self.assertEqual(len(value.tokens),128)

    def test_budget_and_positions(self):
        value=layout(range(32),[126336]*32,range(32,96),[1]*64,range(16),[2]*16,cache_length=96)
        self.assertEqual(value.common+2*value.search,128)
        with self.assertRaises(ValueError):layout([1],[126336],[1],[2],[],[])
        with self.assertRaises(ValueError):layout([1],[126336],[],[],[2],[3])
        with self.assertRaises(ValueError):layout([1],[126336],[],[],[1],[3],cache_length=1)

    def test_special_and_non_mask_inputs_rejected(self):
        with self.assertRaises(ValueError):layout([1],[7],[],[],[1],[3])
        with self.assertRaises(ValueError):layout([1],[126336],[2],[126336],[],[])
        for token in (126336,126081):
            with self.assertRaises(ValueError):layout([1],[126336],[],[],[1],[token],forbidden={126081})

    def test_pending_predictions_are_filtered_without_future_information(self):
        old=[dict(position=1,token=3,confidence=.8),dict(position=2,token=9,confidence=.99),
             dict(position=3,token=7,confidence=.9)]
        self.assertEqual(pending_candidates(old,[1,2],{9}),[old[0]])
        self.assertEqual(pending_candidates([],range(32)),[])

    def test_owned_cache_restores_even_on_failure(self):
        block=SimpleNamespace(k_cache=torch.ones(3,2),v_cache=torch.zeros(3,2))
        model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[block])))
        k,v=block.k_cache,block.v_cache
        with self.assertRaises(RuntimeError):
            with owned_cache(model):
                block.k_cache.fill_(2);raise RuntimeError('intentional')
        self.assertIs(block.k_cache,k);self.assertIs(block.v_cache,v)
        self.assertTrue(torch.equal(k,torch.ones(3,2)))

    def test_adapter_restores_symbol_and_delegates_normal_paths(self):
        original=lambda *args:17
        module=SimpleNamespace(flash_fused_elastic_cache=original)
        model=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(blocks=[object()])))
        with patch('focus_dllm.tuning.joint_engine.inspect.getmodule',return_value=module):
            with self.assertRaises(RuntimeError):
                with joint_adapter(model):
                    result=module.flash_fused_elastic_cache(None,None,0,None,[None]*12)
                    self.assertEqual(result,17)
                    raise RuntimeError('intentional')
        self.assertIs(module.flash_fused_elastic_cache,original)

    def test_query_tiling_retains_all_local_keys_and_public_ownership(self):
        value=plan()
        kwargs=dict(positions=[None,None,[torch.ones(64,128),torch.ones(64,128)],[],torch.zeros(64),None],
                    lengths=[[],None,None,None,None,[0],1,64,32,128,None,False])
        a,ka=prepare_call(value,kwargs,64,'cpu')
        b,kb=prepare_call(value,kwargs,64,'cpu',tiled=True)
        self.assertTrue(torch.equal(a,b))
        self.assertTrue(torch.equal(ka['positions'][-1],kb['positions'][-1]))
        self.assertTrue(torch.equal(ka['positions'][1],kb['positions'][1]))
        self.assertEqual(kb['lengths'][1].shape,(4,4))
        self.assertFalse(set(public_cache_write_positions(value))&set(kb['positions'][1].tolist()))
        q=torch.randn(128,8,dtype=torch.double);k=torch.randn_like(q);v=torch.randn_like(q)
        mask=value.mask()
        dense=(q@k.T).masked_fill(~mask,float('-inf')).softmax(-1)@v
        tiled=torch.cat([(q[start:start+32]@k.T).masked_fill(~mask[start:start+32],float('-inf')).softmax(-1)@v
                         for start in range(0,128,32)])
        torch.testing.assert_close(dense,tiled,rtol=1e-12,atol=1e-12)


if __name__=='__main__':unittest.main()

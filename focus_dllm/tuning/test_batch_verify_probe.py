"""CPU packing/scope/decision checks; no model acceleration evidence."""
import importlib.util
from pathlib import Path
import unittest

import torch

spec = importlib.util.spec_from_file_location('preflight', Path(__file__).with_name('batch_verify_probe.py'))
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


class ProbeChecks(unittest.TestCase):
    def states(self):
        return [torch.tensor([[-1,-1,-1,9,9]]),torch.tensor([[2,-1,-1,9,9]]),
                torch.tensor([[2,3,-1,9,9]])]

    def cache(self):
        return [(torch.arange(24.).reshape(1,2,3,4),torch.zeros(1,2,3,4))]

    def test_native_window(self):
        preflight.validate_window(self.states(),3,-1)

    def test_no_cross_block(self):
        states=self.states();states[-1][0,4]=8
        with self.assertRaises(ValueError):preflight.validate_window(states,3,-1)

    def test_fixed_tokens_never_change(self):
        states=self.states();states[-1][0,0]=7
        with self.assertRaises(ValueError):preflight.validate_window(states,3,-1)

    def test_no_completed_state(self):
        states=self.states();states[-1][0,2]=4
        with self.assertRaises(ValueError):preflight.validate_window(states,3,-1)

    def test_no_duplicate_state(self):
        with self.assertRaises(ValueError):preflight.validate_window([self.states()[0]]*2,3,-1)

    def test_no_reverting_to_mask(self):
        states=self.states();states[-1][0,0]=-1
        with self.assertRaises(ValueError):preflight.validate_window(states,3,-1)

    def test_no_multiple_requests(self):
        with self.assertRaises(ValueError):preflight.validate_window([self.states()[0].repeat(2,1)],3,-1)

    def test_fixed_prefix_views(self):
        cache=self.cache(); snapshot=cache[0][0].clone()
        packed, past=preflight.pack_states(self.states(),cache)
        self.assertEqual(tuple(past[0][0].shape),(3,2,3,4))
        self.assertEqual(past[0][0].data_ptr(),cache[0][0].data_ptr())
        self.assertTrue(torch.equal(past[0][0][2],snapshot[0]))
        packed[1,0]=55
        self.assertEqual(self.states()[0][0,0],-1)
        self.assertTrue(torch.equal(cache[0][0],snapshot))

    def test_reject_batched_formal_prefix(self):
        cache=[tuple(t.repeat(2,1,1,1) for t in self.cache()[0])]
        with self.assertRaises(ValueError):preflight.pack_states(self.states(),cache)

    def test_reject_mismatched_prefix_lengths(self):
        cache=self.cache()+[(torch.zeros(1,2,4,4),torch.zeros(1,2,4,4))]
        with self.assertRaises(ValueError):preflight.pack_states(self.states(),cache)

    def test_owned_branches_do_not_alias(self):
        cache=self.cache();_,packed=preflight.pack_states(self.states(),cache,owned=True)
        packed[0][0][0,0,1,0]=999
        self.assertNotEqual(float(packed[0][0][1,0,1,0]),999.)
        self.assertNotEqual(float(cache[0][0][0,0,1,0]),999.)

    def test_single_owned_branch_still_isolates_teacher(self):
        cache=self.cache();_,packed=preflight.pack_states(self.states()[:1],cache,owned=True)
        self.assertNotEqual(packed[0][0].data_ptr(),cache[0][0].data_ptr())
        self.assertTrue(torch.equal(packed[0][0],cache[0][0]))

    def test_threshold_and_argmax_fallback(self):
        logits=torch.tensor([[0.,.1,.2],[0.,.3,.2],[0.,.1,.2]],dtype=torch.double)
        pos,tokens,confidence,state=preflight.decision(logits,torch.tensor([-1,-1,7]),3,-1,.9)
        self.assertEqual(pos.tolist(),[1]); self.assertEqual(tokens.tolist(),[1])
        self.assertEqual(state.tolist(),[-1,1,7]); self.assertEqual(confidence.numel(),2)

    def test_native_inclusive_threshold(self):
        # The equality position is NOT the argmax fallback; strict > would
        # exclude it and fail this test.
        logits=torch.tensor([[0.,2.],[0.,3.]],dtype=torch.double)
        threshold=float(torch.softmax(logits[0],0)[1])
        pos,_,_,_=preflight.decision(logits,torch.tensor([-1,-1]),2,-1,threshold)
        self.assertEqual(pos.tolist(),[0,1])

    def test_empty_active_region(self):
        with self.assertRaises(ValueError):preflight.decision(torch.zeros(2,4),torch.tensor([1,2]),2,-1,.9)


if __name__=='__main__':unittest.main(verbosity=2)

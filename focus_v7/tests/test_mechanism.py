from types import SimpleNamespace
import unittest

import torch

from focus_v7.mechanism import full_call, instrument, private_cache


def sample_decider(values):
    decision = decide(values)
    return decision + 1


def decide(values):
    return sum(values)


class MechanismTests(unittest.TestCase):
    def test_instrumentation_keeps_decision_and_observes_before_return(self):
        seen=[]
        traced=instrument(sample_decider,lambda state:seen.append(state['decision']))
        self.assertEqual(traced([1,2]),sample_decider([1,2]))
        self.assertEqual(seen,[3])

    def test_private_cache_restores_objects_after_exception(self):
        k=torch.arange(12).reshape(4,3).float();v=k+1
        block=SimpleNamespace(k_cache=k,v_cache=v)
        with self.assertRaisesRegex(RuntimeError,'deliberate'):
            with private_cache([block]):
                self.assertNotEqual(block.k_cache.data_ptr(),k.data_ptr())
                block.k_cache.fill_(100)
                raise RuntimeError('deliberate')
        self.assertIs(block.k_cache,k);self.assertIs(block.v_cache,v)
        self.assertEqual(k[0,0],0)

    def test_full_reference_covers_each_initialized_position_once(self):
        canvas=torch.full((128,),126336)
        canvas[:9]=torch.arange(9)
        canvas[96:]=126081
        current=dict(raw=SimpleNamespace(device='cpu'),canvas=canvas,maximum=128,
                     state=dict(seqlen_k=[112],rotary_emb_pos=[None,None],attn_scores=torch.zeros(128)))
        query,pos,lengths=full_call(current,[30,19,70])
        self.assertEqual(pos[0][:3].tolist(),[30,19,70])
        self.assertEqual(sorted(pos[0].tolist()),list(range(112)))
        self.assertTrue(torch.equal(query[0],canvas[pos[0]]))
        self.assertEqual(lengths[-1],False)
        self.assertEqual(lengths[4][:,1].unique().tolist(),[112])
        self.assertEqual(lengths[2][0,2:].tolist(),[0,3])
        self.assertEqual(lengths[3][0,2].item(),3)
        self.assertEqual(lengths[3][-1,-1].item(),112)
        for bad in ([30,30],[-1],[120],[3]):
            with self.assertRaises(ValueError):full_call(current,bad)


if __name__=='__main__':unittest.main()

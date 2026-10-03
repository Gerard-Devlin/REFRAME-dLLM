import unittest
from types import SimpleNamespace
import torch
from .temporal_batch import MASK_ID
from .temporal_batch_ceiling import Workload,action,audit,clone_cache


class Tests(unittest.TestCase):
    def test_original_threshold_and_fallback_actions(self):
        state=torch.full((32,),MASK_ID,dtype=torch.long);targets=torch.arange(32)
        logits=torch.zeros(32,8,dtype=torch.float64);logits[:,1]=1.;logits[3,2]=2.
        positions,values,_,next_state=action(logits,state,targets)
        self.assertEqual(positions.tolist(),[3]);self.assertEqual(values.tolist(),[2])
        self.assertEqual(next_state[3],2);self.assertTrue((state==MASK_ID).all())

    def test_bitwise_and_decision_equivalence_are_distinct(self):
        state=torch.full((1,32),MASK_ID,dtype=torch.long);targets=torch.arange(32)[None,:]
        logits=torch.zeros(32,8);logits[:,1]=2.
        report=audit([logits],[logits+.25],[state],[targets])
        self.assertFalse(report['all_logits_bitwise_equal'])
        self.assertTrue(report['all_commit_actions_equal'])
        self.assertEqual(report['max_logit_error'],.25)

    def test_dual_branch_ownership_and_serial_cache_isolation(self):
        past=[(torch.zeros(1,2,64,4),torch.ones(1,2,64,4))]
        branches=clone_cache(past,2);branches[0][0][0].fill_(3)
        self.assertTrue((past[0][0]==0).all());self.assertTrue((branches[0][0][1]==0).all())
        class Fake:
            def __call__(self,ids,**kwargs):
                kwargs['past_key_values'][0][0].add_(1)
                return SimpleNamespace(logits=torch.zeros(ids.shape[0],32,8))
        rows=[torch.full((1,32),MASK_ID,dtype=torch.long) for _ in range(2)]
        targets=[torch.arange(32)[None,:] for _ in rows]
        engine=Workload(Fake(),rows,targets,False,past,torch.zeros(1,64,dtype=torch.bool))
        engine.prepare_batch();engine.raw(False);engine.raw(True)
        self.assertTrue((past[0][0]==0).all())
        self.assertEqual(float(engine.serial_cache[0][0].mean()),2.)
        self.assertEqual(float(engine.batch_cache[0][0].mean()),1.)


if __name__=='__main__':unittest.main()

import unittest
import numpy as np
import torch
from .delta_frontier_probe import check_action,describe_drift,region_masks


class Tests(unittest.TestCase):
    def test_real_action_shapes_and_position_offset(self):
        canvas=torch.full((1,34),126336);canvas[0,:2]=1
        logits=torch.zeros(1,32,7);logits[:,:,3]=20
        target=torch.arange(2,34)
        actual,expected=check_action(logits,canvas,target,dict(commit_positions=list(range(32)),commit_values=[3]*32))
        self.assertEqual(actual,[(p+2,v) for p,v in expected])

    def test_regions_partition_generation(self):
        x=[3,4]+[7]*32+[126336]*64
        r=region_masks(x,2,1)
        self.assertEqual(int(r['prompt'].sum()),2);self.assertEqual(int(r['decoded'].sum()),32)
        self.assertEqual(int(r['active'].sum()),32);self.assertEqual(int(r['future_mask'].sum()),32)
        self.assertTrue(np.all(sum(r[n].astype(int) for n in ['prompt','decoded','active','future_mask'])==1))

    def test_drift_energy_ceiling_is_not_tolerance(self):
        value=describe_drift(torch.tensor([3.,0.,0.,0.]),torch.tensor([.003,0.,0.,0.]),
            torch.tensor([False,True,True,True]),{'all':np.ones(4,bool)})
        self.assertEqual(value['oracle_99pct_energy_coverage'],.25)
        self.assertEqual(value['all']['within_tolerance']['0.01'],1.)
        self.assertEqual(value['all']['exact_unchanged_fraction'],.75)


if __name__=='__main__':unittest.main()

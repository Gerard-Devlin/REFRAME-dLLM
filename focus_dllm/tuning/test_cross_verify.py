import unittest

import torch

from .cross_verify import make_views,acceptance


class CrossVerifyTests(unittest.TestCase):
    def setUp(self):
        self.canvas=torch.tensor([[9,9,9,9,9,9,9,9,9,9]])
        self.positions=torch.tensor([2,5,1,7]);self.drafts=torch.tensor([1,2,3,4])
        self.views=make_views(self.canvas,self.positions,self.drafts,mask_id=9,special_ids={0})

    def test_own_positions_masked_and_only_opposite_visible(self):
        for row in (0,1):
            own=self.views.owners==row
            self.assertTrue(torch.equal(self.views.canvas[row,self.positions[own]],torch.full((2,),9)))
            self.assertTrue(torch.equal(self.views.canvas[row,self.positions[~own]],self.drafts[~own]))
        self.assertTrue(torch.equal(self.canvas,torch.full_like(self.canvas,9)))

    def test_committed_and_other_masks_unchanged(self):
        canvas=self.canvas.clone();canvas[0,0]=6
        views=make_views(canvas,self.positions,self.drafts,mask_id=9)
        remaining=torch.tensor([0,3,4,6,8,9])
        self.assertTrue(torch.equal(views.canvas[:,remaining],canvas[:,remaining].repeat(2,1)))

    def test_visible_self_copy_cannot_be_accepted(self):
        logits=torch.zeros(2,4,10)
        for i,owner in enumerate(self.views.owners):logits[1-owner,i,self.drafts[i]]=20
        accepted,p=acceptance(logits,self.views)
        self.assertFalse(bool(accepted.any()));self.assertTrue(torch.equal(p,torch.full((4,),.1,dtype=torch.double)))

    def test_masked_prediction_must_match_and_be_confident(self):
        logits=torch.zeros(2,4,10)
        for i,owner in enumerate(self.views.owners):logits[owner,i,self.drafts[i]]=20
        logits[0,0,6]=30;logits[1,1].zero_()
        accepted,_=acceptance(logits,self.views)
        self.assertEqual(accepted.tolist(),[False,False,True,True])

    def test_two_candidates_both_have_other_context(self):
        views=make_views(self.canvas,self.positions[:2],self.drafts[:2],mask_id=9)
        self.assertEqual(views.owners.tolist(),[0,1])
        self.assertEqual(views.canvas[0,5].item(),2);self.assertEqual(views.canvas[1,2].item(),1)

    def test_ownership(self):
        self.positions.fill_(0);self.drafts.fill_(8)
        self.assertEqual(self.views.positions.tolist(),[2,5,1,7]);self.assertEqual(self.views.drafts.tolist(),[1,2,3,4])

    def test_special_and_boundary_exclusions(self):
        for positions,drafts in ((torch.tensor([0,32]),torch.tensor([1,2])),
                (torch.tensor([2,2]),torch.tensor([1,2])),
                (torch.tensor([2,5]),torch.tensor([9,2])),
                (torch.tensor([2,5]),torch.tensor([0,2]))):
            with self.assertRaises(ValueError):make_views(self.canvas,positions,drafts,mask_id=9,special_ids={0})
        canvas=self.canvas.clone();canvas[0,2]=1
        with self.assertRaises(ValueError):make_views(canvas,self.positions,self.drafts,mask_id=9)

    def test_mismatched_logits_rejected(self):
        with self.assertRaises(ValueError):acceptance(torch.zeros(1,4,10),self.views)
        with self.assertRaises(ValueError):acceptance(torch.zeros(2,4,3),self.views)


if __name__=='__main__':unittest.main()

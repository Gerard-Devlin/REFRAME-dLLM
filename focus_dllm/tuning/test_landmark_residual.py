import unittest

import torch

from .landmark_residual import LocalTransport,error_metrics


class LocalTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.old=(torch.randn(1,2,9,4),torch.randn(1,2,9,4))
        self.source=torch.tensor([1,3,6,8]);self.dropped=torch.tensor([2,4,5,7])
        self.plan=LocalTransport.prepare(self.old,self.source,self.dropped)

    def test_no_unavailable_labels(self):
        fresh=self.old[0]+torch.randn_like(self.old[0])*.1
        expected=self.plan.predict(self.old[0],fresh)
        modified=fresh.clone();modified[:,:,self.dropped]=float('nan')
        modified[:,:,:1]=float('nan')
        self.assertTrue(torch.equal(expected,self.plan.predict(self.old[0],modified)))

    def test_constant_field_and_zero_change(self):
        for old in self.old:
            self.assertTrue(torch.equal(self.plan.predict(old,old),old[:,:,self.dropped]))
            change=torch.randn(1,2,1,4)
            self.assertTrue(torch.allclose(self.plan.predict(old,old+change),old[:,:,self.dropped]+change,atol=1e-6))

    def test_convexity_and_domain(self):
        self.assertTrue((self.plan.weights>=0).all())
        self.assertTrue(torch.allclose(self.plan.weights.sum(-1),torch.ones(2,4)))
        self.assertTrue((self.plan.neighbors<self.source.numel()).all())
        self.assertEqual(self.plan.neighbors.shape,(2,4,4))

    def test_source_permutation_and_no_mutations(self):
        perm=torch.tensor([2,0,3,1])
        other=LocalTransport.prepare(self.old,self.source[perm],self.dropped)
        fresh=self.old[0]+torch.randn_like(self.old[0])*.1
        versions=[t._version for t in (*self.old,fresh,self.source,self.dropped)]
        self.assertTrue(torch.allclose(self.plan.predict(self.old[0],fresh),other.predict(self.old[0],fresh),atol=1e-6))
        self.assertEqual(versions,[t._version for t in (*self.old,fresh,self.source,self.dropped)])

    def test_geometry_reads_only_old(self):
        repeat=LocalTransport.prepare(self.old,self.source,self.dropped)
        self.assertTrue(torch.equal(self.plan.neighbors,repeat.neighbors))
        self.assertTrue(torch.equal(self.plan.weights,repeat.weights))
        self.source.fill_(0);self.dropped.fill_(0)
        self.assertEqual(self.plan.source.tolist(),[1,3,6,8])
        self.assertEqual(self.plan.dropped.tolist(),[2,4,5,7])

    def test_degenerate_history_and_one_landmark(self):
        old=(torch.zeros_like(self.old[0]),torch.zeros_like(self.old[1]))
        plan=LocalTransport.prepare(old,torch.tensor([1]),self.plan.dropped)
        fresh=old[0].clone();fresh[:,:,1]=2
        self.assertTrue(torch.equal(plan.predict(old[0],fresh),torch.full((1,2,4,4),2.)))

    def test_error_metrics_labels_separate(self):
        old=self.old[0];fresh=old.clone();fresh[:,:,self.plan.dropped]+=1
        energy,local,mean,count=error_metrics(self.plan,old,fresh).tolist()
        self.assertEqual(energy,32);self.assertEqual(local,32);self.assertEqual(mean,32);self.assertEqual(count,32)

    def test_reject_overlap_gaps_shapes(self):
        for src,dst in ((torch.tensor([]),self.plan.dropped),(torch.tensor([1,1]),self.plan.dropped),
            (self.plan.source,torch.tensor([1])),(self.plan.source,torch.tensor([9]))):
            with self.assertRaises(ValueError):LocalTransport.prepare(self.old,src,dst)
        with self.assertRaises(ValueError):LocalTransport.prepare((self.old[0].expand(2,-1,-1,-1),self.old[1].expand(2,-1,-1,-1)),self.plan.source,self.plan.dropped)
        with self.assertRaises(ValueError):self.plan.predict(self.old[0][:,:,:8],self.old[0][:,:,:8])


if __name__=='__main__':unittest.main()

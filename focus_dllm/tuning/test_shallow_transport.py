import unittest

import torch

from .shallow_transport import ShallowTransport


class ShallowTransportTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(1234)
        self.old=torch.randn(1,9,6);self.new=self.old+torch.randn_like(self.old)*.1
        self.source=torch.tensor([3,5,7,10]);self.dropped=torch.tensor([4,6,8,9])
        self.shape=(1,2,12,4)
        self.plan=ShallowTransport.prepare(self.old,self.new,self.source,self.dropped,
            prefix_length=3,kv_shape=self.shape)

    def test_no_fresh_dropped_labels_or_prefix(self):
        old=torch.randn(self.shape);fresh=old+torch.randn_like(old)*.1
        expected=self.plan.predict(old,fresh)
        fresh[:,:,self.dropped]=float('nan');fresh[:,:,:3]=float('nan')
        self.assertTrue(torch.equal(expected,self.plan.predict(old,fresh)))

    def test_current_input_changes_weights(self):
        other=ShallowTransport.prepare(self.old,self.old,self.source,self.dropped,prefix_length=3,kv_shape=self.shape)
        self.assertFalse(torch.equal(self.plan.weights,other.weights))
        self.assertTrue(torch.allclose(other.weights,torch.full((4,4),.25)))

    def test_constant_field_and_zero(self):
        old=torch.randn(self.shape);change=torch.randn(1,2,1,4)
        self.assertTrue(torch.equal(self.plan.predict(old,old),old[:,:,self.dropped]))
        self.assertTrue(torch.allclose(self.plan.predict(old,old+change),old[:,:,self.dropped]+change,atol=1e-6))
        self.assertTrue(torch.allclose(self.plan.weights.sum(-1),torch.ones(4),atol=1e-6))

    def test_linear_field_with_spanning_landmarks(self):
        old_hidden=torch.zeros(1,6,2)
        features=torch.tensor([[[0.,0.],[1.,0.],[0.,1.],[1.,1.],[.2,.3],[.7,.1]]])
        src=torch.tensor([0,1,2,3]);dst=torch.tensor([4,5])
        plan=ShallowTransport.prepare(old_hidden,features,src,dst,prefix_length=0,
            kv_shape=(1,1,6,2),regularization=.0001)
        transform=torch.tensor([[2.,1.],[-1.,3.]])
        labels=(features@transform).unsqueeze(1)+.5
        prediction=plan.predict(torch.zeros_like(labels),labels)
        self.assertTrue(torch.allclose(prediction,labels[:,:,dst],atol=2e-4))

    def test_source_permutation_and_versions(self):
        perm=torch.tensor([3,1,0,2]);versions=[t._version for t in (self.old,self.new,self.source,self.dropped)]
        other=ShallowTransport.prepare(self.old,self.new,self.source[perm],self.dropped,prefix_length=3,kv_shape=self.shape)
        old=torch.randn(self.shape);fresh=old+torch.randn_like(old)*.1
        self.assertTrue(torch.allclose(self.plan.predict(old,fresh),other.predict(old,fresh),atol=2e-6))
        self.assertEqual(versions,[t._version for t in (self.old,self.new,self.source,self.dropped)])

    def test_one_landmark(self):
        plan=ShallowTransport.prepare(self.old,self.new,self.source[:1],self.dropped,prefix_length=3,kv_shape=self.shape)
        self.assertTrue(torch.equal(plan.weights,torch.ones(4,1)))

    def test_owned_positions(self):
        self.source.fill_(0);self.dropped.fill_(0)
        self.assertEqual(self.plan.source.tolist(),[3,5,7,10]);self.assertEqual(self.plan.dropped.tolist(),[4,6,8,9])

    def test_reject_domains(self):
        for src,dst in ((torch.tensor([2]),self.dropped),(self.source,torch.tensor([12])),
            (self.source,self.source),(self.source,torch.tensor([],dtype=torch.long))):
            with self.assertRaises(ValueError):ShallowTransport.prepare(self.old,self.new,src,dst,prefix_length=3,kv_shape=self.shape)
        with self.assertRaises(ValueError):ShallowTransport.prepare(self.old,self.new,self.source,self.dropped,prefix_length=2,kv_shape=self.shape)
        with self.assertRaises(ValueError):self.plan.predict(torch.randn(1,2,11,4),torch.randn(1,2,11,4))


if __name__=='__main__':unittest.main()

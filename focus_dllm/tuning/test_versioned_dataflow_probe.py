import unittest
import torch
from .versioned_dataflow_probe import compare,replacement,parse_profile_trace


class Tests(unittest.TestCase):
    def test_multistream_gpu_annotations_do_not_overwrite_full_scope(self):
        # Actual Kineto format: same named CPU scope plus shorter GPU stream scopes.
        events=[dict(ph='X',cat='user_annotation',name='profile_flow',ts=1000.,dur=100.),
            dict(ph='X',cat='gpu_user_annotation',name='profile_flow',ts=1020.,dur=10.),
            dict(ph='X',cat='kernel',name='producer',ts=1005.,dur=20.),
            dict(ph='X',cat='kernel',name='consumer',ts=1020.,dur=20.),
            dict(ph='X',cat='kernel',name='outside',ts=1110.,dur=10.)]
        parsed=parse_profile_trace(dict(traceEvents=events));r=parsed['regions']['profile_flow']
        self.assertEqual(r['kernels'],2);self.assertEqual(r['summed_kernel_us'],40.)
        self.assertEqual(r['union_kernel_us'],35.);self.assertEqual(r['overlap_kernel_us'],5.)
        self.assertEqual(parsed['unassigned_kernels'],1)

    def test_duplicate_or_absent_cpu_scope_is_not_silently_accepted(self):
        cpu=dict(ph='X',cat='user_annotation',name='profile_flow',ts=1.,dur=10.)
        with self.assertRaisesRegex(ValueError,'Duplicate'):parse_profile_trace(dict(traceEvents=[cpu,cpu]))
        with self.assertRaisesRegex(ValueError,'Missing'):
            parse_profile_trace(dict(traceEvents=[dict(cpu,cat='gpu_user_annotation')]))

    def test_compare_detects_bf16_last_bit(self):
        x=torch.tensor([[1.,2.]],dtype=torch.bfloat16);y=x.clone();y[0,0]=1.0078125
        result=compare(x,y)
        self.assertFalse(result['bitwise_equal']);self.assertEqual(result['changed_fraction'],.5)
        self.assertEqual(result['max_absolute'],.0078125)

    def test_replacement_uses_current_input_and_restores_on_exception(self):
        class Block(torch.nn.Module):
            def forward(self,x,**kwargs):return x+1,None
        block=Block();owned=torch.zeros(2)
        with self.assertRaisesRegex(RuntimeError,'restore'):
            with replacement(block,owned,lambda:owned*2):
                self.assertTrue(torch.equal(block(torch.tensor([1.,2.]))[0],torch.tensor([2.,4.])))
                self.assertTrue(torch.equal(block(torch.tensor([3.,4.]))[0],torch.tensor([6.,8.])))
                with self.assertRaises(ValueError):block(owned,use_cache=True)
                raise RuntimeError('restore')
        self.assertNotIn('forward',block.__dict__)
        self.assertTrue(torch.equal(block(torch.tensor([1.,2.]))[0],torch.tensor([2.,3.])))


if __name__=='__main__':unittest.main()

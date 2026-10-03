import unittest
from types import SimpleNamespace
import torch
from .temporal_batch import reconstruct_states,window_indices,choose_cached_window,compact_head,extract_block,MASK_ID


class Tests(unittest.TestCase):
    def test_reconstruct_never_reveals_future_reference(self):
        first=[MASK_ID]*32;second=[1]+[MASK_ID]*31
        trace=[dict(call=0,block=0,canvas=first,commit_positions=[0],commit_values=[1]),
               dict(call=1,block=0,canvas=second,commit_positions=list(range(1,32)),commit_values=list(range(2,33)))]
        states,targets=reconstruct_states([77],trace,list(range(1,33)))
        self.assertEqual(states[0],[77]+[MASK_ID]*32)
        self.assertEqual(states[1],[77,1]+[MASK_ID]*31)
        self.assertEqual(targets[0],list(range(1,33)))
        with self.assertRaises(ValueError):reconstruct_states([77],trace,[99]*32)
        trace[1]['commit_positions']=[0];trace[1]['commit_values']=[2]
        with self.assertRaises(ValueError):reconstruct_states([77],trace,list(range(1,33)))

    def test_fixed_window_no_repeated_synthetic_states(self):
        self.assertEqual(window_indices(48),list(range(16,32)))
        self.assertEqual(window_indices(20),list(range(4,20)))
        with self.assertRaises(ValueError):window_indices(15)
        windows=[dict(block=2,ordinary_total=16),dict(block=0,ordinary_total=4),dict(block=1,ordinary_total=16)]
        self.assertEqual(choose_cached_window(windows)['block'],1)

    def test_row_heads_and_hook_restoration(self):
        norm=torch.nn.Identity()
        fake=SimpleNamespace(model=SimpleNamespace(transformer=SimpleNamespace(ln_f=norm)))
        h=torch.randn(2,64,4);positions=torch.stack([torch.arange(32),torch.arange(32,64)])
        with compact_head(fake,positions):
            out=norm(h)
            self.assertTrue(torch.equal(out[0],h[0,:32]))
            self.assertTrue(torch.equal(out[1],h[1,32:]))
        self.assertTrue(torch.equal(norm(h),h))
        with self.assertRaises(RuntimeError):
            with compact_head(fake,positions):raise RuntimeError('Private failure')
        self.assertEqual(len(norm._forward_hooks),0)
        self.assertTrue(torch.equal(extract_block(h,positions),torch.stack([h[0,:32],h[1,32:]])))


if __name__=='__main__':unittest.main()

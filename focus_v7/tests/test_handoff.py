import unittest
from types import SimpleNamespace

import torch

from focus_v7.handoff import Update,build_call,insert,summarize,transition


class HandoffTests(unittest.TestCase):
    def current(self):
        canvas=torch.tensor([10]*40+[126336]*40)
        state=dict(active_batch=[0],block_m=32,seqlen_k=[80],rotary_emb_pos=None,info=[],
                   attn_scores=torch.zeros(80),start_layer=[32],query_tracked_blocks=None,
                   num_active=1,max_length=80,block_n=128,elastic_cache=None)
        return dict(state=state,canvas=canvas)

    def test_transition_full_set_and_tie_order(self):
        self.assertEqual(transition([5,2,3],[.9,.8,.95],[11,12,13]),Update((3,5),(13,11)))
        self.assertEqual(transition([5,2],[.4,.4],[11,12]),Update((2,),(12,)))

    def test_mask_invalid_confidence_and_rewrite_fail_closed(self):
        with self.assertRaises(ValueError):transition([1],[float('nan')],[3])
        with self.assertRaises(ValueError):transition([1],[.9],[126336])
        with self.assertRaises(ValueError):insert([1],[3],Update((1,),(4,)))
        with self.assertRaises(ValueError):insert([1],[126336],Update((2,),(4,)))

    def test_actual_two_tile_geometry_and_external_versions(self):
        current=self.current();tracked=list(range(32));window=list(range(40,72));change=Update((41,),(999,))
        query,pos,lens=build_call(current,tracked,window,change,paired=True)
        self.assertEqual(query.shape,(1,128));self.assertEqual(pos[-1].shape,(128,64))
        self.assertEqual(lens[1].tolist(),[[0,16,0,64],[0,16,64,128]])
        self.assertEqual(int(query[0,1]),126336);self.assertEqual(int(query[0,65]),999)
        self.assertEqual(pos[0][:64].tolist(),pos[0][64:].tolist())
        self.assertFalse(torch.isin(pos[1],pos[0]).any())
        self.assertEqual(len(torch.unique(pos[1]))+64,80)

    def test_paired_input_is_exactly_independent_input_concatenation(self):
        current=self.current();t=list(range(32));w=list(range(40,72));u=Update((41,),(999,))
        joined,_,_=build_call(current,t,w,u,paired=True)
        first,_,_=build_call(current,t,w);second,_,_=build_call(current,t,w,u)
        self.assertTrue(torch.equal(joined,torch.cat((first,second),dim=1)))

    def test_no_cross_branch_label_path_in_32_layers(self):
        edges=torch.zeros((128,128),dtype=torch.bool)
        edges[:64,:64]=True;edges[64:,64:]=True
        reach=torch.zeros((128,1),dtype=torch.bool);reach[65]=True
        for _ in range(32):reach|=(edges.long()@reach.long())>0
        self.assertFalse(reach[:64].any());self.assertTrue(reach[64:].all())

    def test_uninitialized_or_duplicate_physical_positions_rejected(self):
        with self.assertRaises(ValueError):build_call(self.current(),list(range(32)),list(range(31,63)))
        with self.assertRaises(ValueError):build_call(self.current(),list(range(32)),list(range(65,97)))

    def test_gate_does_not_use_input_only_agreement_as_speed_evidence(self):
        rows=[dict(proposal_matches_fresh_update=True,packed_actions_equal=True)]*16
        timing=[dict(single_mean_seconds=1,packed_mean_seconds=1.5)]*2
        self.assertTrue(summarize(rows,timing)['gate_passed'])
        rows[0]=dict(proposal_matches_fresh_update=True,packed_actions_equal=False)
        self.assertFalse(summarize(rows,timing)['gate_passed'])
        rows=[dict(proposal_matches_fresh_update=True,packed_actions_equal=True)]*16
        self.assertFalse(summarize(rows,[dict(single_mean_seconds=1,packed_mean_seconds=2)]*2)['gate_passed'])


if __name__=='__main__':unittest.main()

import unittest
from types import SimpleNamespace
import torch

from focus_v6.audit import label_reachability
from focus_v7.packet import prefix_mask
from focus_v6.audit import Layout
from focus_v7.query_budget import (BudgetLayout,TrackedScheduler,budget_mask,
    build_budget_call,budget_commit,budget_promotion)
from focus_v7.revision import select_commit,promotion


class QueryBudgetTests(unittest.TestCase):
    def test_all_layer_label_exclusion_and_one_version(self):
        for m,k in [(32,8),(16,16),(17,8),(8,8),(3,3),(1,1)]:
            layout=BudgetLayout(m,k);mask=budget_mask(layout)
            reachable=label_reachability(mask,layout,32)
            self.assertFalse(bool(reachable[:layout.clean.stop].any()))
            expected=torch.arange(k)[None,:]<torch.arange(k)[:,None]
            self.assertTrue(torch.equal(reachable[layout.audit],expected))
            physical=list(range(layout.tracked+m))+list(range(layout.tracked,layout.tracked+k))*2
            for row in mask:
                keys=[p for p,b in zip(physical,row) if b]
                self.assertEqual(len(keys),len(set(keys)))
                self.assertEqual(len(keys),layout.tracked+m)

    def test_legacy_geometry_and_decisions_are_unchanged(self):
        for k in (1,8,16):
            new=BudgetLayout(k,k);old=Layout(k)
            self.assertTrue(torch.equal(budget_mask(new),prefix_mask(old)))
            for confidence in (.3,.93,.995):
                p=[confidence]*k;drafts=list(range(100,100+k));ids=drafts.copy()
                ids[-1]+=20
                a=select_commit(p,ids,drafts,clean_p=p,clean_ids=ids)
                b=budget_commit(p,ids,drafts,clean_p=p,clean_ids=ids,capacity=16)
                self.assertEqual(a,b)
                positions=list(range(1000,1000+k));tracked=list(range(new.tracked))
                self.assertEqual(promotion(old,positions,tracked,a),budget_promotion(new,positions,tracked,b))

    def test_actual_call_geometry_covers_clean_unselected_positions(self):
        x=torch.full((120,),126336,dtype=torch.long);x[:64]=17
        state=dict(active_batch=[0],block_m=32,num_decoded=[64],full_pos=torch.arange(120)[None,:],
            x=x,x_draft=torch.arange(120),mask_id=126336,seqlen_k=[120],rotary_emb_pos=None,
            info=[],attn_scores=torch.zeros(120),start_layer=[32],query_tracked_blocks=None,
            num_active=1,max_length=120,block_n=128,elastic_cache=None)
        query,positions,_,layout,clean,drafts=build_budget_call(state,8,32)
        self.assertEqual(query.shape,(1,64));self.assertEqual(clean.tolist(),list(range(64,96)))
        self.assertEqual(drafts.tolist(),list(range(64,72)))
        self.assertTrue(torch.equal(positions[-1],budget_mask(layout)))
        self.assertFalse(bool(torch.isin(positions[1],torch.unique(positions[0])).any()))
        self.assertEqual(len(torch.unique(torch.cat((positions[0],positions[1])))),120)
        x[80]=22
        with self.assertRaises(ValueError):build_budget_call(state,8,32)

    def test_repair_capacity_caps_simultaneous_clean_roots(self):
        layout=BudgetLayout(32,8);ids=list(range(100,132));p=[.999]*32
        decision=budget_commit(p[:8],ids[:8],ids[:8],clean_p=p,clean_ids=ids,capacity=16)
        self.assertEqual(decision.accepted,8);self.assertEqual(decision.progress,16)
        clean=list(range(1000,1032));tracked=list(range(16))
        rows,dest,dirty=budget_promotion(layout,clean,tracked,decision)
        self.assertEqual(len(dirty),8)
        source=dict(zip(dest,rows))
        for i in range(8):self.assertEqual(source[clean[i]],layout.draft.start+i)
        for p in dirty:self.assertNotIn(p,source)
        for i in range(16,32):self.assertEqual(source[clean[i]],layout.clean.start+i)
        scheduler=TrackedScheduler('age');committed=[clean[i] for i in decision.indices]
        chosen=scheduler.select(tracked+committed,committed,16)
        self.assertEqual(chosen,committed)

    def test_rejected_draft_does_not_block_independent_unselected_root(self):
        decision=budget_commit([.2]*8,[700]*8,list(range(8)),clean_p=[.2]*31+[.99],
            clean_ids=list(range(100,132)),capacity=16)
        self.assertEqual(decision.accepted,0);self.assertEqual(decision.indices,(31,))
        self.assertEqual(decision.tokens,(131,))
        _,dest,dirty=budget_promotion(BudgetLayout(32,8),list(range(32)),list(range(100,116)),decision)
        self.assertEqual(dirty,(31,));self.assertNotIn(31,dest)

    def test_old_context_rotation_and_mandatory_identity_repairs(self):
        scheduler=TrackedScheduler('age');known=list(range(40));changed=list(range(32,40))
        first=scheduler.select(known,changed,16)
        self.assertEqual(first,changed+list(range(8)))
        scheduler.installed(first)
        second=scheduler.select(known,changed,16)
        self.assertEqual(second,changed+list(range(8,16)))
        scheduler.installed(second)
        third=scheduler.select(known,changed,16)
        self.assertEqual(third,changed+list(range(16,24)))
        with self.assertRaises(ValueError):scheduler.select(known,list(range(17)),16)

    def test_invalid_probabilities_and_versions_fail_closed(self):
        with self.assertRaises(ValueError):BudgetLayout(32,16)
        with self.assertRaises(ValueError):budget_commit([float('nan')],[1],[1],clean_p=[.9],clean_ids=[1],capacity=16)
        with self.assertRaises(ValueError):budget_commit([.2],[1],[1],clean_p=[.9],clean_ids=[126336],capacity=16,forbidden=(126336,))

    def test_shadow_readout_uses_only_projected_rows(self):
        from focus_dllm.tuning.flash_readout import Readout
        from focus_v7.budget_evaluate import projected_pair,maximum_error
        layout=BudgetLayout(32,8);compact=torch.arange(48*5,dtype=torch.float32).reshape(1,48,5)
        value=SimpleNamespace(logits=Readout(compact,16,48,64))
        clean,audit=projected_pair(value,layout)
        self.assertEqual(clean.shape,(32,5));self.assertEqual(audit.shape,(8,5))
        self.assertTrue(torch.equal(audit,compact[0,40:]))
        mutated=clean.clone();mutated[:,2]=-torch.inf
        self.assertEqual(maximum_error(clean,mutated,mask_id=2),0)
        mutated[0,1]+=1
        self.assertEqual(maximum_error(clean,mutated,mask_id=2),1)
        mutated[0,0]=torch.nan
        with self.assertRaises(AssertionError):maximum_error(clean,mutated,mask_id=2)


if __name__=='__main__':unittest.main()

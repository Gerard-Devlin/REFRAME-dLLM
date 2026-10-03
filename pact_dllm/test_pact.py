"""Independent optimizer, information-flow and cache-identity regression checks."""
import itertools
import math
import random
import unittest
from types import SimpleNamespace
from unittest.mock import patch
import torch
from .graph import DAG, build_dag
from .planner import maximum_closure, joint_plan
from .cache import Ledger
from .engine import prepare, Runtime
from focus_dllm.tuning.firebreak.attention import dense_reference


class PlannerTests(unittest.TestCase):
    def test_bruteforce_cyclic_and_shared_graphs(self):
        rng = random.Random(28493)
        for n in range(1,9):
            for _ in range(25):
                w=[rng.uniform(-2,2) for _ in range(n)]
                req=[tuple(j for j in range(n) if j!=i and rng.random()<.18) for i in range(n)]
                mandatory=(rng.randrange(n),) if rng.random()<.3 else ()
                feasible=[]
                for mask in range(1<<n):
                    s={i for i in range(n) if mask&(1<<i)}
                    if set(mandatory)<=s and all(set(req[i])<=s for i in s):
                        feasible.append(sum(w[i] for i in s))
                got=maximum_closure(w,req,mandatory)
                self.assertAlmostEqual(got.objective,max(feasible),places=9)

    def test_shared_refresh_is_paid_once(self):
        dag=DAG(((),()))
        p=joint_plan(dag,[.9,.9],[(0,),(0,)],[1.],price=.6,query_cost=0.)
        self.assertEqual(p['candidates'],(0,1));self.assertEqual(p['tiles'],(0,))
        self.assertAlmostEqual(p['objective'],1.2)

    def test_parent_cost_cannot_be_skipped(self):
        p=joint_plan(DAG(((),(0,))),[0.,.9],[(),(0,)],[4.],price=.5)
        self.assertEqual(p['candidates'],())

    def test_mandatory_negative_cost(self):
        self.assertEqual(maximum_closure([-100,1],[(),(0,)],mandatory=(0,)).selected,(0,1))

    def test_reject_invalid_cost(self):
        for value in (float('nan'),float('inf')):
            with self.assertRaises(ValueError):maximum_closure([value],[()])
        with self.assertRaises(ValueError):joint_plan(DAG(((),)),[1],[(0,)],[-1])
        with self.assertRaises(ValueError):maximum_closure([1],[(1,)])


class GraphTests(unittest.TestCase):
    def test_failure_is_local(self):
        # 0->1; independent 2->3. Failure1 does not cancel branch2->3.
        g=DAG(((),(0,),(),(2,)))
        self.assertEqual(g.closed_accept([True,False,True,True]),(0,2,3))
        self.assertEqual(g.closed_accept([False,True,True,True]),(2,3))

    def test_all_pass_subsets_are_closed(self):
        g=DAG(((),(0,),(),(0,2),(1,3)))
        for passes in itertools.product((False,True),repeat=5):
            result=g.closed_accept(passes)
            for i in result:
                self.assertTrue(passes[i]);self.assertTrue(set(g.parents[i])<=set(result))

    def test_32_layer_paths_are_only_ancestors(self):
        g=DAG(((),(0,),(),(0,2),(1,3)))
        paths=g.label_paths(32)
        self.assertEqual(paths[5:],g.ancestors())
        for i,p in enumerate(paths[5:]):self.assertFalse(p&(1<<i))

    def test_execution_subset_requires_ancestors(self):
        g=DAG(((),(0,),(),(2,)))
        self.assertEqual(g.subset([2,3]).parents,((),(0,)))
        with self.assertRaises(ValueError):g.subset([1,3])

    def test_cycle_and_future_parent_rejected(self):
        for rows in (((1,),()),((0,),), ((),(0,0))):
            with self.assertRaises(ValueError):DAG(rows)

    def test_proxy_graph_respects_bound(self):
        g=build_dag([[1.]*16 for _ in range(16)],max_parents=2,max_ancestors=4)
        self.assertTrue(all(a.bit_count()<=4 for a in g.ancestors()))
        self.assertTrue(all(len(p)<=2 for p in g.parents))


class LedgerTests(unittest.TestCase):
    def test_committed_identity_is_dirty_even_after_mask_refresh(self):
        l=Ledger([10,126336]);t=l.ticket([1]);l.mark_refreshed(t)
        l.change([1],[23]);self.assertEqual(l.dirty(),(1,))
        fresh=l.ticket([1]);l.mark_refreshed(fresh);self.assertEqual(l.dirty(),())

    def test_draft_promotion_rejected(self):
        l=Ledger([10,20])
        with self.assertRaises(ValueError):l.mark_refreshed(l.ticket([1],'draft'))
        self.assertEqual(l.cached_tokens,[10,20])

    def test_stale_context_rejected_before_metadata_write(self):
        l=Ledger([10,20]);t=l.ticket([0]);l.change([1],[21])
        with self.assertRaises(ValueError):l.mark_refreshed(t)
        self.assertEqual(l.observed_epoch,[0,0])

    def test_unchanged_commit_preserves_epoch(self):
        l=Ledger([10,20]);l.change([1],[20]);self.assertEqual(l.epoch,0)

    def test_invalid_commit_has_no_partial_change(self):
        l=Ledger([10,20])
        with self.assertRaises(ValueError):l.change([0,9],[11,21])
        self.assertEqual(l.tokens,[10,20])


class LayoutTests(unittest.TestCase):
    def setUp(self):
        self.g=DAG(((),(0,),(),(2,)))
        self.canvas=[11]+[126336]*4+[12]
        self.r=prepare(self.canvas,range(6),None,'cpu',candidate_positions=(1,2,3,4),candidate_tokens=(20,21,22,23),dag=self.g)

    def test_exactly_one_key_version_per_position(self):
        # Key rows: four draft variants, six clean positions.
        for choice in self.r.choices.tolist():
            for i,p in enumerate(range(6)):
                versions=int(choice[4+i])+(int(choice[p-1]) if 1<=p<=4 else 0)
                self.assertEqual(versions,1)
                self.assertEqual(int(self.r.mapping[p]),-2)

    def test_background_has_no_draft_label_paths(self):
        self.assertFalse(self.r.choices[8:,:4].any())
        self.assertTrue(self.r.choices[8:,4:].all())

    def test_verification_own_and_unrelated_labels_excluded(self):
        a=self.g.ancestors()
        for i,row in enumerate(self.r.choices[4:8,:4].tolist()):
            self.assertEqual(sum((1<<j) for j,v in enumerate(row) if v),a[i])

    def test_verification_never_becomes_key(self):
        self.assertTrue(set(range(4,8)).isdisjoint(self.r.private_rows.tolist()))

    def test_dense_attention_matches_explicit_version_selection(self):
        torch.manual_seed(4183)
        m,b=len(self.r.ids),len(self.r.private_rows)
        q=torch.randn(m,2,8);bk=torch.randn(6,2,8);bv=torch.randn_like(bk)
        k=torch.randn(b,2,8);v=torch.randn_like(k)
        actual=dense_reference(q,bk,bv,k,v,self.r.mapping,self.r.choices)
        for r,choices in enumerate(self.r.choices):
            pick=choices.nonzero().flatten()
            ref=torch.einsum('hd,nhd->hn',q[r],k[pick])/math.sqrt(8)
            out=torch.einsum('hn,nhd->hd',ref.softmax(-1),v[pick])
            self.assertTrue(torch.allclose(actual[r],out,atol=2e-6))

    def test_normal_refresh_has_unique_clean_bank(self):
        rows=prepare(self.canvas,(0,2,5),None,'cpu')
        self.assertEqual(rows.private_rows.tolist(),[0,1,2])
        self.assertTrue(rows.choices.all())
        self.assertEqual(rows.mapping.tolist(),[-2,-1,-2,-1,-1,-2])

    def test_invalid_missing_clean_mask_rejected(self):
        with self.assertRaises(ValueError):
            prepare(self.canvas,(0,5),None,'cpu',candidate_positions=(1,),candidate_tokens=(20,),dag=DAG(((),)))


def toy_model():
    torch.manual_seed(51577)
    def linear(a,b):return torch.nn.Linear(a,b,bias=False)
    blocks=[]
    for _ in range(32):
        blocks.append(SimpleNamespace(attn_norm=torch.nn.LayerNorm(16),q_proj=linear(16,16),
            k_proj=linear(16,16),v_proj=linear(16,16),dropout=torch.nn.Identity(),attn_out=linear(16,16),
            ff_norm=torch.nn.LayerNorm(16),ff_proj=linear(16,24),up_proj=linear(16,24),
            act=torch.nn.SiLU(),ff_out=linear(24,16)))
    config=SimpleNamespace(n_heads=2,d_model=16,weight_tying=False,scale_logits=False)
    transformer=SimpleNamespace(blocks=blocks,wte=torch.nn.Embedding(126337,16),
        emb_drop=torch.nn.Identity(),ln_f=torch.nn.LayerNorm(16),ff_out=linear(16,32))
    return SimpleNamespace(model=SimpleNamespace(config=config,transformer=transformer))


def toy_projection(block,xn,ready,h,d):
    return (block.q_proj(xn).view(-1,h,d),
            block.k_proj(xn).index_select(0,ready.private_rows).view(-1,h,d),
            block.v_proj(xn).index_select(0,ready.private_rows).view(-1,h,d))


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.patch=patch('pact_dllm.engine.native_projection',toy_projection);self.patch.start()
        self.addCleanup(self.patch.stop)
        self.model=toy_model();self.canvas=[11,126336,126336,12]
        self.runtime=Runtime(self.model,None,attention_reference=True)
        self.runtime.prefill(self.canvas,(1,2))

    def test_full_refresh_matches_independent_full_recompute(self):
        self.runtime.commit((1,),(20,))
        rows=prepare(self.runtime.ledger.tokens,range(4),None,'cpu')
        updated=self.runtime.run(rows);self.runtime.promote_clean(updated)
        other=Runtime(self.model,None,attention_reference=True)
        full=other.prefill(self.runtime.ledger.tokens,range(4))
        self.assertTrue(torch.allclose(updated['logits'],full['logits'],atol=1e-6))
        for a,b in zip(self.runtime.cache,other.cache):
            self.assertTrue(all(torch.allclose(x,y,atol=1e-6) for x,y in zip(a,b)))

    def test_missing_dirty_refresh_rejected(self):
        self.runtime.commit((1,),(20,))
        with self.assertRaises(ValueError):self.runtime.run(prepare(self.runtime.ledger.tokens,(2,),None,'cpu'))

    def test_stale_transaction_does_not_write_any_layer(self):
        result=self.runtime.run(prepare(self.canvas,range(4),None,'cpu'))
        before=[(k.clone(),v.clone()) for k,v in self.runtime.cache]
        self.runtime.commit((1,),(20,))
        with self.assertRaises(ValueError):self.runtime.promote_clean(result)
        self.assertTrue(all(torch.equal(k,a) and torch.equal(v,b) for (k,v),(a,b) in zip(self.runtime.cache,before)))

    def test_true_32_layer_draft_noninterference(self):
        g=DAG(((),(0,)))
        ready=prepare(self.canvas,range(4),None,'cpu',candidate_positions=(1,2),candidate_tokens=(20,21),dag=g)
        before=self.runtime.run(ready);ready.ids[0]=22;after=self.runtime.run(ready)
        self.assertTrue(torch.equal(before['logits'][0],after['logits'][0]))
        for a,b in zip(before['updates'],after['updates']):
            self.assertTrue(all(torch.equal(x,y) for x,y in zip(a,b)))


class PoolTests(unittest.TestCase):
    def test_identity_scan_is_once_not_per_position(self):
        from .signals import measure
        ledger=Ledger([11]*256+[126336]*16)
        original=ledger.dirty;calls=[]
        def watched():calls.append(1);return original()
        ledger.dirty=watched
        runtime=SimpleNamespace(ledger=ledger,cache=[None]*3+[(torch.randn(272,2,128).bfloat16(),None)])
        measure(runtime,torch.randn(16,2,128).bfloat16(),tuple(range(256,272)),tuple(range(256,272)))
        self.assertEqual(len(calls),1,'Cache planning must not scan the whole canvas for every position')

    def test_vectorized_pool_matches_per_tile_reference(self):
        from .signals import pooled_keys
        torch.manual_seed(87254)
        for n in (1,3,4,5,8,15,256,1001):
            keys=torch.randn(n,2,128).bfloat16()
            tiles=[tuple(range(i,min(i+4,n))) for i in range(0,n,4)]
            expected=torch.stack([keys[list(t)].float().mean(0) for t in tiles])
            self.assertTrue(torch.equal(pooled_keys(keys,tiles),expected))

    def test_noncontiguous_positions_and_ragged_tail(self):
        from .signals import pooled_keys
        keys=torch.randn(31,2,128)
        tiles=[(1,5,7,13),(18,20,21,22),(25,30)]
        self.assertTrue(torch.equal(pooled_keys(keys,tiles),torch.stack([keys[list(t)].mean(0) for t in tiles])))

    def test_empty_pool(self):
        from .signals import pooled_keys
        self.assertEqual(pooled_keys(torch.randn(4,2,128),()).shape,(0,2,128))


class QualityTests(unittest.TestCase):
    def test_unknown_scores_do_not_become_a_full_accuracy(self):
        from .quality import accuracy
        result=accuracy([True,False,None])
        self.assertIsNone(result['accuracy']);self.assertEqual(result['known_accuracy'],.5)
        self.assertEqual(result['accuracy_bounds'],[1/3,2/3])

    def test_paired_missing_scores_stay_excluded_and_visible(self):
        from .quality import paired
        result=paired([True,False,None],[False,False,True],draws=100)
        self.assertEqual(result['accuracy_delta'],.5)
        self.assertEqual(result['paired_examples'],2);self.assertEqual(result['excluded_unknown'],1)
        with self.assertRaises(ValueError):paired([True],[True,False])

    def test_frozen_comparison_rejects_wrong_prompt_order_before_scoring(self):
        import tempfile,json
        from pathlib import Path
        from .quality import load_baselines
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); data=root/'data.json';data.write_text('[]')
            (root/'cpu_frozen_development128_20261003.json').write_text(json.dumps(
                {'datasets':{'humaneval':{'development_ids':['a','b'],'sha256':'unused'}}}))
            with self.assertRaisesRegex(ValueError,'IDs/data'):
                load_baselines(root,'humaneval',[{'id':'b'},{'id':'a'}],data,256,'model','rev')


if __name__=='__main__':unittest.main()

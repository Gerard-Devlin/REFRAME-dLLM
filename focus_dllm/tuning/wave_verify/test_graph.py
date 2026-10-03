import itertools
import unittest
from .graph import reference_mask,required_rows,analyze,optimal_batches


class GraphTests(unittest.TestCase):
    def test_first_output_needs_all_later_mask_rows(self):
        mask = reference_mask(32,16)
        needed = required_rows(mask,[48],32)
        self.assertTrue(set(range(48,64)).issubset(needed[-2]))
        self.assertEqual(len(needed[0]),63)
        self.assertNotIn(47,needed[0])  # last draft is not a MASK ancestor

    def test_other_batch_geometries_also_need_mask_tail(self):
        for search in (1,4,8,16):
            tracked=64-2*search
            needed=required_rows(reference_mask(32,search),[tracked+search],32)
            self.assertTrue(set(range(tracked+search,64)).issubset(needed[0]))

    def test_preserving_graph_leaves_no_large_prefix_only_savings(self):
        result=analyze()
        self.assertEqual(result['early_required_candidate_mask_rows'],16)
        self.assertEqual(result['early_required_draft_rows'],15)
        self.assertLess(result['prefix_vs_full_target_row_proxy_saving'],.02)

    def test_dependency_can_change_first_output(self):
        # A positive normalized linear attention model witnesses a real path;
        # pruning a reachable late MASK is NOT graph equivalence.
        mask=reference_mask(4,2);target=6
        def run(values):
            for _ in range(3):
                values=[old+sum(v for v,on in zip(values,row) if on)/sum(row)
                        for old,row in zip(values,mask)]
            return values[target]
        base=[0.]*8;changed=list(base);changed[7]=1.
        self.assertGreater(run(changed)-run(base),0.)

    def test_dp_matches_all_partitions_for_given_tables(self):
        n=5;cost=[[None]*(n+1) for _ in range(n+1)];survive=[[None]*(n+1) for _ in range(n+1)]
        for i in range(n):
            for j in range(i+1,n+1):
                cost[i][j]=1+.2*(j-i);survive[i][j]=.7**(j-i)
        values,ends=optimal_batches(cost,survive)
        possible=[]
        for flags in itertools.product((False,True),repeat=n-1):
            boundaries=[0]+[i+1 for i,on in enumerate(flags) if on]+[n]
            total=0.;reach=1.
            for i,j in zip(boundaries,boundaries[1:]):
                total+=reach*cost[i][j];reach*=survive[i][j]
            possible.append(total)
        self.assertAlmostEqual(values[0],min(possible))


if __name__=='__main__':unittest.main()

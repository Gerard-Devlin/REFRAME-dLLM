import unittest
from focus_v6.audit import Layout
from focus_v7.revision import Commit,promotion,select_commit


class RevisionTests(unittest.TestCase):
    def test_clean_conflict_stops_dependent_prefix(self):
        plan=select_commit([.99,.99,.99],[1,2,3],[1,2,3],clean_p=[.2,.99,.95],clean_ids=[1,9,8])
        self.assertEqual(plan.accepted,1)
        self.assertEqual(plan.tokens,(1,9,8))
        self.assertEqual(plan.indices,(0,1,2))

    def test_no_low_probability_audit_correction(self):
        plan=select_commit([.1,.3],[8,9],[1,2],clean_p=[.4,.7],clean_ids=[3,4])
        self.assertEqual((plan.accepted,plan.indices,plan.tokens,plan.kind),(0,(1,),(4,),'clean_fallback'))

    def test_independent_clean_tail_survives_prefix_failure(self):
        plan=select_commit([.99,.1,.99],[1,9,3],[1,2,3],clean_p=[.2,.2,.95],clean_ids=[1,9,7])
        self.assertEqual(plan.indices,(0,2));self.assertEqual(plan.tokens,(1,7))
        layout=Layout(3);tracked=list(range(layout.tracked));candidates=[101,102,103]
        rows,dest,dirty=promotion(layout,candidates,tracked,plan)
        self.assertEqual(dirty,(103,));self.assertNotIn(103,dest)
        self.assertEqual(rows[dest.index(101)],layout.draft.start)
        self.assertEqual(rows[dest.index(102)],layout.clean.start+1)
        self.assertNotIn(layout.draft.start+1,rows)

    def test_progress_capacity_and_no_duplicate_commits(self):
        plan=select_commit([.1]*16,[1]*16,[2]*16,clean_p=[.99]*16,clean_ids=[3]*16)
        self.assertEqual(plan.progress,16);self.assertEqual(len(set(plan.indices)),16)
        layout=Layout(16)
        rows,dest,dirty=promotion(layout,list(range(100,116)),list(range(16)),plan)
        self.assertEqual(len(dirty),16);self.assertEqual(dest,tuple(range(16)))

    def test_forbidden_and_invalid_values_fail_closed(self):
        with self.assertRaises(ValueError):
            select_commit([.1],[1],[2],clean_p=[.99],clean_ids=[126336],forbidden=(126336,))
        with self.assertRaises(ValueError):
            select_commit([float('nan')],[1],[1],clean_p=[.9],clean_ids=[1])
        with self.assertRaises(ValueError):
            promotion(Layout(3),[100,101,102],list(range(55)),Commit(1,(1,),(9,),'bad'))


if __name__=='__main__':unittest.main()

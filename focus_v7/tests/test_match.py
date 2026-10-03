import unittest
from focus_v7.greedy import decide
from focus_v7.packet import decide as budget_decide, promotion_plan
from focus_v6.audit import Layout


class MatchTests(unittest.TestCase):
    def test_low_confidence_argmax_is_separate_policy(self):
        p,tokens = [.6,.7,.6],[3,4,5]
        self.assertEqual(decide(p,tokens,tokens).accepted,3)
        self.assertEqual(budget_decide(p,tokens,tokens).accepted,0)

    def test_first_mismatch_and_no_later_commit(self):
        result=decide([.9,.2,.99],[3,9,5],[3,4,5])
        self.assertEqual((result.accepted,result.tokens,result.correction),(1,(3,9),9))
        rows,dest,dirty=promotion_plan(Layout(3),(100,101,102),tuple(range(55)),result)
        self.assertEqual(dirty,(101,))
        self.assertNotIn(101,dest)

    def test_match_does_not_bypass_special_token_policy(self):
        result=decide([1.,1.],[3,126336],[3,126336],forbidden=(126336,))
        self.assertEqual(result.tokens,(3,))

    def test_invalid_statistics_fail_closed(self):
        with self.assertRaises(ValueError):
            decide([float('nan')],[1],[1])


if __name__=='__main__':
    unittest.main()

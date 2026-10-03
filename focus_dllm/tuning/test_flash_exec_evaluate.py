import unittest
from .flash_exec_evaluate import validate,paired_intervals,canvas_metadata


class Tests(unittest.TestCase):
    def test_independent_selection_contract(self):
        validate(64,16,[256,512])
        for args in ((0,16,[256]),(64,0,[256]),(64,16,[128]),(64,16,[256,256])):
            with self.assertRaises(ValueError):validate(*args)

    def test_complete_pairing_and_intervals(self):
        value=paired_intervals([0,1,1],[0,1,1],[2.,4.,6.],[1.,2.,3.])
        self.assertEqual(value['accuracy_difference_95ci_pp'],[0.,0.])
        self.assertEqual(value['speedup_95ci'],[2.,2.])
        with self.assertRaises(ValueError):paired_intervals([1],[1,0],[2.],[1.])
        with self.assertRaises(ValueError):paired_intervals([1],[1],[2.],[0.])

    def test_canvas_only_uses_real_in_budget_commits(self):
        data=canvas_metadata([([9,10,11,14],[126081,7,126081,126081]),([11],[8])],10,4)
        self.assertEqual(data['raw_token_ids'],[7,8,126336,126336])
        self.assertTrue(data['budget_without_eos']);self.assertIsNone(data['first_eos_in_budget'])
        data=canvas_metadata([([12],[126081])],10,4)
        self.assertEqual(data['first_eos_in_budget'],2)
        with self.assertRaises(ValueError):canvas_metadata([([10,11],[7])],10,4)


if __name__=='__main__':unittest.main()

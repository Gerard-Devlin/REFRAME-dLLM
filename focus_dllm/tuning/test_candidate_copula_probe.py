import unittest
import torch
from .candidate_copula_probe import conditional_canvas, aggregate


class Tests(unittest.TestCase):
    def test_own_mask_and_original_immutable(self):
        canvas=torch.tensor([[9,126336,126336,5]])
        saved=canvas.clone()
        value=conditional_canvas(canvas,1,2,17)
        self.assertEqual(value.tolist(),[[9,126336,17,5]])
        self.assertTrue(torch.equal(canvas,saved))
        with self.assertRaises(ValueError):conditional_canvas(canvas,1,1,2)
        with self.assertRaises(ValueError):conditional_canvas(canvas,1,0,2)

    def test_gate_clusters_prompts_not_mc_repetitions(self):
        records=[dict(task=task,id=prompt,aligned_interaction_nats=.1,independent_interaction_nats=0.,
                      shuffled_interaction_nats=.03,largest_marginal_mc_error=.003,topk_pair_mass_independent=.6)
                 for task in ('humaneval','mbpp','math') for prompt in (0,1) for _ in range(4)]
        value=aggregate(records)
        self.assertEqual(value['prompts'],6);self.assertEqual(value['states'],24)
        self.assertTrue(value['mechanism_gate_pass'])
        records[0]['aligned_interaction_nats']=-2.
        self.assertFalse(aggregate(records)['mechanism_gate_pass'])
        self.assertFalse(aggregate(records[:4])['mechanism_gate_pass'])
        self.assertFalse(aggregate([])['mechanism_gate_pass'])


if __name__=='__main__':unittest.main()

import unittest
import torch

from focus_v7.generation import Horizon, select_tracked, valid_predictions, render


class GenerationTests(unittest.TestCase):
    def test_identity_repair_is_not_displaced_by_background_score(self):
        result = select_tracked(list(range(80)), [2, 3, 6], 16)
        self.assertEqual(result[:3], [2, 3, 6])
        self.assertEqual(len(set(result)), 16)
        with self.assertRaises(ValueError):
            select_tracked(list(range(80)), list(range(17)), 16)
        self.assertEqual(select_tracked(list(range(80)),list(range(16)),16),list(range(16)))

    def test_rejected_identity_is_not_legal_background(self):
        with self.assertRaises(ValueError):
            select_tracked([0, 1, 2], [7], 2)

    def test_horizon_uses_first_discovered_group_not_first_sorted_eos(self):
        state = Horizon(256)
        state.observe([11, 8, 20], [9, 9, 2], 9)
        self.assertEqual((state.limit, state.discovered), (12, 11))
        state.observe([5], [9], 9)
        self.assertEqual((state.limit, state.discovered), (12, 11))

    def test_fixed_work_still_records_eos_without_free_early_stop(self):
        state = Horizon(256, fixed=True)
        state.observe([11], [9], 9)
        self.assertEqual((state.limit, state.discovered), (256, 11))

    def test_mask_cannot_masquerade_as_progress_but_eos_can(self):
        logits = torch.tensor([[100., 2., 3.], [0., 4., 3.]])
        probability, token = valid_predictions(logits, 0)
        self.assertEqual(token, [2, 1])
        self.assertTrue(all(0 < p <= 1 for p in probability))

    def test_render_keeps_official_two_stage_output_semantics(self):
        class Tokenizer:
            def decode(self, ids, skip_special_tokens=False):
                if not skip_special_tokens:
                    return 'code STOP tail'
                return 'clean'
            def __call__(self, text):
                self.text = text
                return {'input_ids': [3]}
        tokenizer = Tokenizer()
        self.assertEqual(render(tokenizer, [1,2], ('STOP',)), 'clean')
        self.assertEqual(tokenizer.text, 'code ')


if __name__ == '__main__':
    unittest.main()

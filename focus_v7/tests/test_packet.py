import unittest
import torch

from focus_v6.audit import Layout, label_reachability
from focus_v7.packet import prefix_mask, decide, promotion_plan, require_identity_repair
from focus_v7.cache import promote


class PacketTests(unittest.TestCase):
    def test_all_layer_source_isolation(self):
        for k in (2, 4, 8, 16):
            layout = Layout(k)
            labels = label_reachability(prefix_mask(layout), layout, layers=32)
            self.assertFalse(labels[:layout.tracked+k].any())
            for i in range(k):
                self.assertEqual(labels[layout.draft.start+i].tolist(), [j <= i for j in range(k)])
                self.assertEqual(labels[layout.audit.start+i].tolist(), [j < i for j in range(k)])

    def test_exactly_one_version_per_position(self):
        layout = Layout(16)
        mask = prefix_mask(layout)
        for q in range(layout.width):
            for i in range(layout.candidates):
                self.assertEqual(int(mask[q, layout.clean.start+i])
                                 + int(mask[q, layout.draft.start+i])
                                 + int(mask[q, layout.audit.start+i]), 1)

    def test_first_failure_retains_prefix_and_one_correction(self):
        value = decide([.99, .98, .1, .999], [10, 11, 22, 13], [10, 11, 12, 13])
        self.assertEqual((value.accepted, value.tokens, value.correction), (2, (10, 11, 22), 22))

    def test_budget_exhaustion_is_not_whole_epoch_abort(self):
        value = decide([.9, .9, .9], [1, 2, 3], [1, 2, 3])
        self.assertEqual(value.tokens, (1, 2, 3))
        self.assertEqual((value.accepted, value.correction), (2, 3))

    def test_all_pass_and_no_special_tail(self):
        self.assertIsNone(decide([1., 1.], [3, 4], [3, 4]).correction)
        self.assertEqual(decide([.1], [126336], [2], forbidden=(126336,)).tokens, ())

    def test_bad_probabilities_fail_closed(self):
        for p in (-.1, 1.1, float('nan')):
            with self.assertRaises(ValueError):
                decide([p], [1], [1])

    def test_promotion_never_installs_rejected_or_corrected_draft(self):
        layout = Layout(4)
        value = decide([.99, .1, .99, .99], [1, 9, 3, 4], [1, 2, 3, 4])
        rows, dest, dirty = promotion_plan(layout, (100, 101, 102, 103), tuple(range(layout.tracked)), value)
        self.assertEqual(dirty, (101,))
        self.assertNotIn(101, dest)
        self.assertEqual(rows[dest.index(100)], layout.draft.start)
        self.assertEqual(rows[dest.index(102)], layout.clean.start+2)
        self.assertFalse(set(range(layout.draft.start+1, layout.draft.stop)) & set(rows))

    def test_required_repair_not_silent_stale_identity(self):
        self.assertTrue(require_identity_repair((3,), (2, 3), {3: 9}, (4, 9)))
        for positions, ids in (((2,), (4,)), ((2, 3), (4, 0))):
            with self.assertRaises(ValueError):
                require_identity_repair((3,), positions, {3: 9}, ids)

    def test_actual_tensor_transaction(self):
        captured = [(torch.arange(12).view(4, 3).float(), torch.ones(4, 3))]*2
        bank = [(torch.zeros(7, 3), torch.zeros(7, 3)) for _ in range(2)]
        promote(bank, captured, torch.tensor([0, 2]), torch.tensor([1, 6]))
        self.assertTrue(torch.equal(bank[0][0][6], captured[0][0][2]))
        self.assertEqual(float(bank[0][0][2].sum()), 0.)
        with self.assertRaises(ValueError):
            promote(captured, captured, torch.tensor([0]), torch.tensor([0]))
        with self.assertRaises(ValueError):
            promote(bank, captured, torch.tensor([0, 1]), torch.tensor([2, 2]))

    def test_packet_capacity_includes_identity_repair(self):
        for k in (8, 16):
            self.assertGreaterEqual(Layout(k).tracked, k)

    def test_summarizer_does_not_credit_baseline_safe_commits(self):
        from focus_v7.probe import summarize
        rows = [dict(k=k, task=task, tokens=[1], teacher_final_matches=[True],
                     packet_wall_ms=10., packet_gpu_ms=9., safe_progress=9,
                     official_accepted=1, regular_ms=10., verify_ms=10., accepted=0)
                for k in (8, 16) for task in ('humaneval', 'mbpp', 'math')]
        summary = summarize(rows)
        self.assertAlmostEqual(summary['8']['fixed_state_progress_cost_ratio'], .2)
        self.assertFalse(summary['8']['gate_passed'])


if __name__ == '__main__':
    unittest.main()

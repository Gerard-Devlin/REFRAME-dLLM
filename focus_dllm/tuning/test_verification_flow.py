import unittest

import torch

from .verification_flow import view_mask, proposal_paths, rejection_record, observe_cumulative


def dummy_verify(x0_p_verify, x_verify_j, query_pos_flat, p_verify, gamma=.8):
    acc_seqlen_verify = 0
    T, S = 0, x0_p_verify.numel()
    x0_p_verify = x0_p_verify.cumprod(dim=0)
    return x0_p_verify >= gamma


def no_cumulative(x):
    return x


class VerificationFlowTests(unittest.TestCase):
    def test_direct_block_does_not_block_multilayer_path(self):
        mask = view_mask(4, 2)
        tracked = 4
        one = proposal_paths(mask, tracked, 2, 1)[tracked + 2:]
        two = proposal_paths(mask, tracked, 2, 2)[tracked + 2:]
        self.assertFalse(bool(one[0, 0]))
        self.assertTrue(bool(two[0, 0]))
        self.assertFalse(bool(two[1, 1]))

    def test_isolated_mask_excludes_own_and_future_labels_at_all_depths(self):
        for search in (0, 1, 2, 4, 16):
            mask = view_mask(32, search, isolated=True)
            tracked = 64 - 2 * search
            for layers in (0, 1, 2, 4, 32):
                paths = proposal_paths(mask, tracked, search, layers)
                self.assertFalse(bool(paths[:tracked].any()))
                self.assertFalse(bool(torch.triu(paths[tracked + search:]).any()))
            self.assertEqual(mask.shape, (64, 64))

    def test_prior_context_allowed(self):
        paths = proposal_paths(view_mask(4, 2, isolated=True), 4, 2, 3)
        self.assertTrue(bool(paths[-1, 0]))

    def test_budget_only_is_not_wrong_token(self):
        record = rejection_record([.95] * 6, [1] * 6, list(range(6)), [1] * 6)
        self.assertEqual(record['accepted'], 4)
        self.assertTrue(record['budget_only_rejection'])
        self.assertEqual(record['discarded_high_confidence_indices'], [5])

    def test_rejected_head_tail_is_conditional(self):
        record = rejection_record([.3, .99, .99], [1, 2, 3], [2, 3, 4], [4, 2, 3])
        self.assertEqual(record['accepted'], 0)
        self.assertTrue(record['rejected_argmax_mismatch'])
        self.assertEqual(record['discarded_high_confidence_indices'], [1, 2])
        self.assertFalse(record['budget_only_rejection'])

    def test_eos_and_zero_excluded_from_tail(self):
        record = rejection_record([.1, .99, 0.], [1, 9, 3], [0, 1, 2], [2, 9, 3], forbidden={9})
        self.assertEqual(record['discarded_high_confidence_indices'], [])

    def test_empty_and_all_accepted(self):
        self.assertIsNone(rejection_record([], [], [], [])['first_rejected'])
        self.assertEqual(rejection_record([1., .9], [1, 2], [0, 1], [1, 2])['accepted'], 2)

    def test_validation(self):
        for probabilities in ([float('nan')], [-.1], [1.1]):
            with self.assertRaises(ValueError):rejection_record(probabilities, [1], [0], [1])
        with self.assertRaises(ValueError):view_mask(32, 17)
        with self.assertRaises(ValueError):rejection_record([.9], [], [1], [1])

    def test_observer_does_not_change_function_or_input(self):
        records = []
        def observer(*args):records.append([value.clone() if isinstance(value, torch.Tensor) else value for value in args])
        clone = observe_cumulative(dummy_verify, observer)
        probability = torch.tensor([.95, .95, .5], dtype=torch.double)
        draft = torch.tensor([1, 2, 3]); positions = torch.tensor([7, 8, 9]); logits = torch.eye(3)
        clean = dummy_verify(probability, draft, positions, logits)
        seen = clone(probability, draft, positions, logits)
        self.assertTrue(torch.equal(clean, seen))
        self.assertTrue(torch.equal(probability, torch.tensor([.95, .95, .5], dtype=torch.double)))
        self.assertEqual(len(records), 1)
        self.assertNotIn('_focus_observe', dummy_verify.__globals__)

    def test_unfamiliar_source_fails(self):
        with self.assertRaises(ValueError):observe_cumulative(no_cumulative, lambda *_:None)


if __name__ == '__main__':unittest.main()

import unittest
import torch

from .candidate_copula import (factors, shuffled_factors, category_probabilities,
    draw_tokens, draw_categories, pair_table, interaction, choose_pair)


class Tests(unittest.TestCase):
    def test_identity_covariance_rank_deficient_and_zero(self):
        b = torch.tensor([[[1., 2., 3.], [1., 2., 3.], [0., 0., 0.]], [[0., 0., 0.]]*3])
        for gamma in (0., .8, 1.):
            value = factors(b, gamma)
            total = value.shared @ value.shared.transpose(-1, -2) + value.residual @ value.residual.transpose(-1, -2)
            self.assertTrue(torch.allclose(total, torch.eye(3).double().expand(2, -1, -1), atol=1e-10, rtol=0))

    def test_shuffling_preserves_covariance_but_changes_cross_mapping(self):
        b = torch.arange(48).reshape(2, 3, 8).double()
        value = factors(b)
        changed = shuffled_factors(value, torch.tensor([[2, 1, 0], [1, 2, 0]]))
        total = changed.shared @ changed.shared.transpose(-1, -2) + changed.residual @ changed.residual.transpose(-1, -2)
        self.assertTrue(torch.allclose(total, torch.eye(3).double().expand(2, -1, -1), atol=1e-10, rtol=0))
        self.assertFalse(torch.allclose(value.shared[0] @ value.shared[1].T, changed.shared[0] @ changed.shared[1].T))

    def test_full_vocabulary_marginals_and_tail_not_truncated(self):
        logits = torch.tensor([[0., -.3, -.5, -.8, -1.], [.2, -.1, -.3, -.4, -.5]])
        b = torch.tensor([[[1., 2.], [2., 1.]], [[1., 2.], [2., 1.]]])
        expected, top, mass = category_probabilities(logits, 2)
        self.assertTrue((mass[:, -1] > .4).all())
        for perm in (None, torch.tensor([[1, 0], [0, 1]])):
            values, choices = draw_tokens(logits, b, 60000, torch.Generator().manual_seed(912), permutations=perm)
            self.assertTrue((choices == 2).any())
            for position in range(2):
                counts = torch.bincount(values[:, position], minlength=5).double()/len(values)
                self.assertLess(float((counts-expected[position]).abs().max()), .01)
                self.assertTrue(all(int(v) not in top[position].tolist() for v in values[choices[:, position] == 2, position]))

    def test_zero_coupling_independent_and_large_joint_draws(self):
        mass = torch.tensor([[.2, .3, .5], [.4, .2, .4]], dtype=torch.float64)
        value = factors(torch.ones(2, 2, 3), gamma=0.)
        table = pair_table(value, mass, 60000, 1337)
        self.assertLess(float((table - mass[0, :, None] * mass[1, None, :]).abs().max()), .006)

    def test_joint_covariance_sampler_matches_literal_construction(self):
        b = torch.tensor([[[1., 0.], [0., 1.]], [[1., 0.], [0., 1.]]])
        mass = torch.tensor([[.4, .4, .2], [.3, .5, .2]], dtype=torch.float64)
        value = factors(b)
        samples = draw_categories(value, mass, 60000, torch.Generator().manual_seed(814))
        literal = torch.bincount(samples[:, 0]*3+samples[:, 1], minlength=9).reshape(3, 3).double()/len(samples)
        table = pair_table(value, mass, 60000, 321)
        self.assertLess(float((table-literal).abs().max()), .012)
        self.assertGreater(float(table[0, 0]+table[1, 1]), .34)

    def test_no_tail_and_extreme_logit_stability(self):
        z = torch.tensor([[1000., -1000., -1100.], [0., 0., 0.]])
        _, _, mass = category_probabilities(z, 3)
        self.assertTrue(torch.equal(mass[:, -1], torch.zeros(2).double()))
        result, choices = draw_tokens(z, torch.ones(2, 3, 2), 256, torch.Generator().manual_seed(1))
        self.assertTrue((result[:, 0] == 0).all()); self.assertFalse((choices == 3).any())

    def test_interaction_centering_and_no_marginal_reward(self):
        p = torch.tensor([[.2, .3, .5], [.4, .2, .4]], dtype=torch.float64)
        raw = torch.tensor([[1., -1., 0.], [-2., 4., 0.], [0., 0., 0.]], dtype=torch.float64)
        centered = interaction(raw, p)
        self.assertTrue(torch.allclose(centered @ p[1], torch.zeros(3).double(), atol=1e-14))
        self.assertTrue(torch.allclose(p[0] @ centered, torch.zeros(3).double(), atol=1e-14))
        self.assertAlmostEqual(float(p[0] @ centered @ p[1]), 0.)

    def test_preselected_pair_special_top1_and_empty(self):
        z = torch.tensor([[4., 0., 0.], [0., 0., 0.], [1., 1., 0.], [3., 0., 0.]])
        self.assertEqual(choose_pair(z, torch.arange(4), set()).tolist(), [1, 2])
        self.assertIsNone(choose_pair(z, torch.arange(4), {0}))
        self.assertIsNone(choose_pair(z[:1], torch.arange(1), set()))

    def test_invalid_parameters_fail_closed(self):
        for gamma in (-1., 2.):
            with self.assertRaises(ValueError): factors(torch.ones(2, 2, 2), gamma)
        with self.assertRaises(ValueError): factors(torch.ones(2, 2, 2), regularization=0)
        with self.assertRaises(ValueError): factors(torch.full((2, 2, 2), float('nan')))
        with self.assertRaises(ValueError): category_probabilities(torch.ones(2, 3), 2, temperature=0.)
        with self.assertRaises(ValueError): shuffled_factors(factors(torch.ones(2, 2, 2)), torch.tensor([[0, 0], [0, 1]]))


if __name__ == '__main__':
    unittest.main()

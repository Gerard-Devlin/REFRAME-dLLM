import unittest
import torch
from .layout import layout, prefix_union, dependency_closed, cumulative_prefixes
from .attention import dense_reference
from .engine import decide
from .boundary import adapted


def boundary_fixture(logits, query_masked_pos, block_m):
    acc_seqlen_masked, j = 0, 0
    logits_masked_j = logits[acc_seqlen_masked : acc_seqlen_masked + block_m]
    return logits_masked_j


class ProvenanceTests(unittest.TestCase):
    def test_all_depths_and_ragged_groups(self):
        for count in (1, 2, 3, 5, 16, 31, 32):
            for groups in (1, 2, 4):
                p = layout(range(10, 10+count), range(100, 100+count),
                           cache_length=50, group_count=groups)
                for depth in (1, 2, 32, 64):
                    deps = p.dependencies(depth)
                    for i, group in enumerate(p.groups):
                        expected = sum(1 << j for j in range(i) if p.groups[j] == group)
                        self.assertEqual(deps[count+i], expected)
                        self.assertFalse(deps[2*count+i] & (1 << i))

    def test_one_version_per_original_position(self):
        p = layout([7, 2, 9, 1], [30, 40, 50, 60], cache_length=12, group_count=2)
        for query, choices in enumerate(p.choices()):
            versions = p.position_versions(query)
            self.assertEqual(len(versions), 12)
            for j, pos in enumerate(p.positions):
                self.assertEqual(versions[pos], ('draft', j) if choices[j] else ('base', pos))

    def test_observers_are_not_keys(self):
        p = layout([0, 1, 2, 3], [10, 11, 12, 13], cache_length=4, group_count=2)
        self.assertTrue(all(len(row) == p.count for row in p.choices()))
        self.assertTrue(all(source in ('base', 'draft') for row in range(12)
                            for source, _ in p.position_versions(row)))

    def test_veto_before_prefix_preserves_closure(self):
        p = layout(range(8), range(20, 28), cache_length=8, group_count=2)
        iso_flags = [True]*8
        veto_flags = [True, False, True, True, True, True, False, True]
        accepted = prefix_union(p.groups, [a and b for a, b in zip(iso_flags, veto_flags)])
        self.assertEqual(accepted, (0, 4, 5))
        self.assertTrue(dependency_closed(p, accepted))
        self.assertFalse(dependency_closed(p, (0, 2, 4)))

    def test_cumulative_budget_is_not_per_token_gate(self):
        self.assertEqual(cumulative_prefixes([0]*3, [.9]*3, [True]*3, .8), (0, 1))
        self.assertEqual(prefix_union([0]*3, [True]*3), (0, 1, 2))

    def test_invalid_special_and_duplicate_candidates(self):
        for pos, token in (([1, 1], [2, 3]), ([1], [126336]), ([30], [2]), ([], [])):
            with self.assertRaises(ValueError):
                layout(pos, token, cache_length=10, group_count=2, forbidden=(126336,))


class AttentionAndDecisionTests(unittest.TestCase):
    def test_dense_version_selection(self):
        # A one-position draft query MUST receive exactly its draft value.
        q = torch.ones(2, 1, 2)
        base_k = torch.zeros(1, 1, 2); base_v = torch.full((1, 1, 2), 9.)
        draft_k = torch.ones(1, 1, 2); draft_v = torch.full((1, 1, 2), 3.)
        mapping = torch.tensor([0]); choices = torch.tensor([[True], [False]])
        out = dense_reference(q, base_k, base_v, draft_k, draft_v, mapping, choices)
        self.assertTrue(torch.equal(out[:, 0, 0], torch.tensor([3., 9.])))
        self.assertTrue(torch.isfinite(out).all())

    def test_js_symmetry_and_identical_views(self):
        p = layout([0, 1], [0, 1], cache_length=2, group_count=2)
        a = torch.tensor([[8., 0., -1.], [0., 8., -1.]])
        b = torch.tensor([[7., 0., -1.], [0., 7., -1.]])
        x = decide(dict(iso=a, cross=b), p)
        y = decide(dict(iso=b, cross=a), p)
        self.assertEqual(x['accepted'], [0, 1])
        self.assertEqual(x['js_nats'], y['js_nats'])
        self.assertTrue(all(abs(v) < 1e-15 for v in decide(dict(iso=a, cross=a), p)['js_nats']))

    def test_cross_cannot_rescue_isolated_failure(self):
        p = layout([0, 1], [0, 1], cache_length=2, group_count=2)
        a = torch.tensor([[0., 8.], [0., 8.]])
        b = torch.tensor([[8., 0.], [0., 8.]])
        self.assertEqual(decide(dict(iso=a, cross=b), p)['accepted'], [1])

    def test_nonfinite_logits_fail_closed(self):
        p = layout([0], [0], cache_length=1, group_count=1, cross=False)
        with self.assertRaises(ValueError):
            decide(dict(iso=torch.tensor([[float('nan'), 1.]]), cross=None), p)

    def test_ragged_boundary_keeps_actual_masked_extent(self):
        patched = adapted(boundary_fixture)
        logits = torch.arange(40)
        self.assertTrue(torch.equal(patched(logits, [torch.zeros(26)], 32), logits[:26]))
        self.assertTrue(torch.equal(patched(logits, [torch.zeros(32)], 32), logits[:32]))


if __name__ == '__main__':
    unittest.main()

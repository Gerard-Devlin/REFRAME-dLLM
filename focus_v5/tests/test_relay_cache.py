import unittest

import torch

from focus_v5.cost_model import Cycle, joint_row_ceiling, row_ceiling
from focus_v5.relay_cache import (
    JointRelayLayout,
    RelayCacheTransaction,
    flash_verify_mask,
    joint_relay_mask,
    proposal_paths,
    relay_verify_mask,
    validate_joint_relay_noninterference,
    validate_relay_noninterference,
)


class RelayMaskTests(unittest.TestCase):
    def test_pinned_flash_has_multilayer_future_leakage(self):
        # Three proposals are the smallest counterexample: draft 0 reads the
        # later MASK row 2, which reads proposal 1 on the next layer.
        block, search = 8, 3
        tracked = 2 * block - 2 * search
        paths = proposal_paths(flash_verify_mask(block, search), tracked, search, 2)
        # The first draft row acquires proposal 1 through the complementary view.
        self.assertTrue(bool(paths[tracked, 1]))

    def test_relay_is_safe_at_all_relevant_depths(self):
        for search in (0, 1, 2, 4, 16):
            for layers in (0, 1, 2, 4, 16, 32):
                result = validate_relay_noninterference(32, search, layers)
                self.assertEqual(result["draft_paths"].shape, (search, search))

    def test_verify_sees_earlier_but_not_self(self):
        block, search = 4, 2
        tracked = 4
        paths = proposal_paths(relay_verify_mask(block, search), tracked, search, 32)
        verify = paths[tracked + search :]
        self.assertFalse(bool(verify[0].any()))
        self.assertTrue(bool(verify[1, 0]))
        self.assertFalse(bool(verify[1, 1]))

    def test_joint_layout_is_constant_width_and_isolated(self):
        for search in (0, 1, 2, 4, 8, 16):
            layout = JointRelayLayout(32, search)
            self.assertEqual(layout.total, 96)
            self.assertEqual(joint_relay_mask(32, search).shape, (96, 96))
            for layers in (0, 1, 2, 8, 32):
                result = validate_joint_relay_noninterference(32, search, layers)
                self.assertFalse(bool(result["paths"][: layout.clean.stop].any()))


class TransactionTests(unittest.TestCase):
    def _cache(self):
        public = [(torch.arange(30).reshape(1, 5, 6).float(), torch.arange(30, 60).reshape(1, 5, 6).float())]
        clean = [(torch.full((1, 3, 6), 11.0), torch.full((1, 3, 6), 22.0))]
        draft = [(torch.full((1, 2, 6), 101.0), torch.full((1, 2, 6), 202.0))]
        return public, clean, draft

    def test_prefix_commit_and_rejected_rollback(self):
        public, clean, draft = self._cache()
        before = [tensor.clone() for tensor in public[0]]
        transaction = RelayCacheTransaction(public, clean, torch.tensor([1, 3, 4]), draft, torch.tensor([1, 3]))
        transaction.commit(1)
        self.assertTrue(torch.equal(public[0][0][:, 1], draft[0][0][:, 0]))
        self.assertTrue(torch.equal(public[0][1][:, 3], clean[0][1][:, 1]))
        self.assertTrue(torch.equal(public[0][0][:, 4], clean[0][0][:, 2]))
        self.assertFalse(torch.equal(public[0][1][:, 4], before[1][:, 4]))

    def test_zero_commit_is_noop(self):
        public, clean, draft = self._cache()
        RelayCacheTransaction(public, clean, torch.tensor([1, 3, 4]), draft, torch.tensor([1, 3])).commit(0)
        self.assertTrue(torch.equal(public[0][0][:, 1], clean[0][0][:, 0]))
        self.assertTrue(torch.equal(public[0][1][:, 3], clean[0][1][:, 1]))

    def test_double_close_fails(self):
        public, clean, draft = self._cache()
        transaction = RelayCacheTransaction(public, clean, torch.tensor([1, 3, 4]), draft, torch.tensor([1, 3]))
        transaction.rollback()
        with self.assertRaises(RuntimeError):
            transaction.commit(1)


class CostTests(unittest.TestCase):
    def test_ceiling(self):
        result = row_ceiling([Cycle(64, 64, 4, 4), Cycle(60, 64, 8, 6)])
        self.assertEqual(result["baseline_query_rows"], 252)
        self.assertEqual(result["removable_identity_refresh_rows"], 10)
        self.assertAlmostEqual(result["optimistic_row_speedup"], 252 / 242)
        joint = joint_row_ceiling([Cycle(160, 64, 4, 4), Cycle(128, 64, 8, 6)])
        self.assertEqual(joint["relay_query_rows"], 192)
        self.assertAlmostEqual(joint["optimistic_row_speedup"], 416 / 192)


if __name__ == "__main__":
    unittest.main()

import unittest
import torch

from focus_v6.audit import Layout, build_call, instrument, isolated_mask, label_reachability, naive_own_exclusion_mask
from focus_v6.epoch import EpochLedger, atomic_pass, expected_rate_ratio


def dummy_generator(model):
    return model()


class AuditTest(unittest.TestCase):
    def test_all_layer_no_own_label_and_clean_background(self):
        for k in (2, 4, 8, 16):
            layout = Layout(k)
            paths = label_reachability(isolated_mask(layout, main_rows=32), layout)
            self.assertFalse(bool(paths[:layout.tracked + k].any()))
            self.assertTrue(torch.equal(paths[layout.draft], torch.eye(k, dtype=torch.bool)))
            self.assertTrue(torch.equal(paths[layout.audit], ~torch.eye(k, dtype=torch.bool)))
            self.assertTrue(bool(paths[64:].all()))

    def test_naive_peer_reads_break_own_exclusion_at_layer_two(self):
        layout = Layout(4)
        one = label_reachability(naive_own_exclusion_mask(layout), layout, 1)
        two = label_reachability(naive_own_exclusion_mask(layout), layout, 2)
        self.assertFalse(bool(one[layout.audit].diagonal().any()))
        self.assertTrue(bool(two[layout.audit].diagonal().all()))

    def test_each_query_has_one_live_identity_per_candidate_position(self):
        layout = Layout(8)
        mask = isolated_mask(layout)
        for i in range(8):
            rows = [layout.clean.start+i, layout.draft.start+i, layout.audit.start+i]
            # Clean/draft/audit each represents the same position. The audit
            # path retains exactly one of these versions for every candidate.
            self.assertTrue(bool((mask[layout.audit, rows].sum(-1) == 1).all()))

    def test_geometry_excludes_private_positions_from_external_cache(self):
        maximum, decoded = 384, 100
        x = torch.full((maximum,), 126336, dtype=torch.long)
        x[:decoded] = torch.arange(decoded)+10
        drafts = x.clone(); drafts[decoded:decoded+8] = torch.arange(8)+1000
        state = dict(x=x, x_draft=drafts, full_pos=torch.arange(maximum).reshape(1, maximum),
                     num_decoded=[decoded], num_verify=8, mask_id=126336, seqlen_k=[384],
                     start_layer=[32], query_tracked_blocks=torch.empty(0, dtype=torch.int32),
                     active_batch=[0], num_active=1, max_length=maximum, block_m=32, block_n=128,
                     elastic_cache=None, rotary_emb_pos=[torch.empty(0), torch.empty(0)],
                     info=[], attn_scores=torch.zeros(maximum))
        for k in (2, 4, 8):
            query, positions, lengths, layout, selected, proposal = build_call(state, k)
            self.assertEqual(query.shape[1], 64)
            self.assertEqual(positions[0].numel(), 64)
            self.assertFalse(bool(torch.isin(positions[1], selected).any()))
            self.assertEqual(lengths[1][0, 3].item(), 64)
            self.assertEqual(proposal.tolist(), list(range(1000, 1000+k)))

    def test_observer_source_shape_fails_closed(self):
        with self.assertRaises(ValueError):
            instrument(dummy_generator, lambda *a: None, lambda *a: None, lambda *a: None)


class EpochTest(unittest.TestCase):
    def ledger(self):
        return EpochLedger([11, 126336, 126336], {"layer": ("old-k", "old-v")}, special_ids={126081})

    def test_failed_epoch_discards_all_provisional_cache_and_tokens(self):
        ledger = self.ledger()
        epoch = ledger.begin([1, 2], [20, 21])
        ledger.stage(epoch, {"layer": ("new-k", "new-v")})
        ledger.finish(epoch, False)
        self.assertEqual(ledger.canvas, (11, 126336, 126336))
        self.assertEqual(ledger.cache["layer"], ("old-k", "old-v"))
        self.assertEqual(ledger.version, 0)

    def test_atomic_commit_and_stale_epoch_rejected(self):
        ledger = self.ledger()
        epoch = ledger.begin([1, 2], [20, 21])
        ledger.stage(epoch, {"layer": ("new-k", "new-v")})
        ledger.finish(epoch, True)
        self.assertEqual(ledger.canvas, (11, 20, 21))
        self.assertEqual(ledger.version, 1)
        with self.assertRaises(ValueError):
            ledger.stage(epoch, {})

    def test_eos_never_speculatively_shortens_canvas(self):
        with self.assertRaises(ValueError):
            self.ledger().begin([1], [126081])

    def test_commit_requires_cache_built_under_same_epoch(self):
        ledger = self.ledger()
        epoch = ledger.begin([1], [20])
        with self.assertRaises(ValueError):
            ledger.finish(epoch, True)

    def test_mutable_tensor_storage_cannot_masquerade_as_version_handle(self):
        with self.assertRaises(ValueError):
            EpochLedger([126336], {"layer": torch.zeros(2)})
        ledger = self.ledger()
        epoch = ledger.begin([1], [20])
        with self.assertRaises(ValueError):
            ledger.stage(epoch, {"layer": ["mutable"]})

    def test_pending_epoch_cannot_be_replaced_before_audit(self):
        ledger = self.ledger()
        ledger.begin([1], [20])
        with self.assertRaises(RuntimeError):
            ledger.begin([2], [21])

    def test_probability_budget_and_top1_are_both_required(self):
        self.assertTrue(atomic_pass([.99, .99], [2, 3], [2, 3]))
        self.assertFalse(atomic_pass([.99, .7], [2, 3], [2, 3]))
        self.assertFalse(atomic_pass([.99, .99], [2, 4], [2, 3]))
        self.assertFalse(atomic_pass([0., 1.], [2, 3], [2, 3]))

    def test_recovery_cost_can_reverse_nominal_speed_gain(self):
        values = dict(baseline_ms=58., joint_ms=35., recovery_ms=35., safe_progress=1.,
                      epoch_size=4, baseline_progress=3.)
        self.assertGreater(expected_rate_ratio(pass_fraction=.9, **values), 1.)
        self.assertLess(expected_rate_ratio(pass_fraction=.1, **values), 1.)


if __name__ == "__main__":
    unittest.main()

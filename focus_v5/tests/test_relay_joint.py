import unittest

import torch

from focus_v5.relay_joint import accepted_prefix, build_joint_call, padded_joint_mask


class RelayJointTest(unittest.TestCase):
    def state(self, search=8):
        block = 32
        decoded = 100
        maximum = 384
        full = torch.arange(maximum).reshape(1, maximum)
        tokens = torch.full((maximum,), 126336, dtype=torch.long)
        tokens[:decoded] = torch.arange(decoded) + 10
        drafts = tokens.clone()
        drafts[decoded:decoded + block] = torch.arange(block) + 1000
        return {
            "model": object(), "x": tokens, "x_draft": drafts, "full_pos": full,
            "num_decoded": [decoded], "num_verify": search, "seqlen_k": [384],
            "start_layer": [32], "query_tracked_blocks": torch.empty(0, dtype=torch.int32),
            "active_batch": [0], "num_active": 1, "max_length": maximum,
            "block_m": block, "block_n": 128, "elastic_cache": None,
            "rotary_emb_pos": [torch.empty(0), torch.empty(0)], "info": [],
            "attn_scores": torch.zeros(maximum), "gamma": .8,
        }

    def test_padded_mask_hides_padding(self):
        mask = padded_joint_mask(32, 8)
        self.assertEqual(tuple(mask.shape), (128, 128))
        self.assertFalse(bool(mask[96:].any()))
        self.assertFalse(bool(mask[:, 96:].any()))

    def test_joint_geometry_and_public_keys(self):
        call = build_joint_call(self.state())
        self.assertEqual(tuple(call.input_ids.shape), (1, 96))
        self.assertEqual(call.positions[0].numel(), 96)
        self.assertEqual(call.lengths[1].tolist(), [[0, 304, 0, 96]])
        self.assertEqual(call.lengths[8], 64)
        self.assertFalse(bool(torch.isin(call.positions[1], torch.cat((call.tracked_positions, call.clean_positions))).any()))

    def test_acceptance_uses_only_verifier_rows(self):
        call = build_joint_call(self.state(search=2))
        logits = torch.full((1, 96, 1300), -10.0)
        drafts = call.input_ids[0, call.layout.draft]
        logits[0, call.layout.verify.start, drafts[0]] = 10
        logits[0, call.layout.verify.start + 1, drafts[1]] = 10
        result = accepted_prefix(logits, call, .8)
        self.assertEqual(result["accepted"], 2)


if __name__ == "__main__":
    unittest.main()

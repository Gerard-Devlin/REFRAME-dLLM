import math
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .focus_v4_runtime import Options, DeviceRotary, select_support, commit
from ..llada_common import MASK_ID
from ..llada_pruning import choose_support, LLaDABlockForward, PositionedRotary


class FakeRotary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(rope_full_precision=True)

    def get_rotary_embedding(self, length, device):
        phases = torch.arange(length, device=device).float()[None, None, :, None]/10
        return phases.sin().expand(1, 1, length, 4), phases.cos().expand(1, 1, length, 4)

    def apply_rotary_pos_emb(self, sin, cos, value):
        a, b = value.chunk(2, -1)
        return value*cos+torch.cat((-b, a), -1)*sin


class Tests(unittest.TestCase):
    def test_gpu_selection_matches_original_geometry(self):
        torch.manual_seed(3)
        for length in (64, 96, 256):
            for keep in (.3125, .5, .75):
                q = torch.randn(1, 2, length, 4)
                k = torch.randn(1, 2, length+7, 4)
                for targets in ([0], [0, 3, 7, 15, 31], list(range(32))):
                    relevance = LLaDABlockForward._relevance({'q':q, 'k':k}, targets)[7:]
                    expected = choose_support(relevance, targets, keep, list(range(32, length)),
                                              [n for n in range(32) if n not in targets])
                    actual = select_support(q, k, torch.tensor(targets), length, 7, keep)
                    self.assertEqual(actual.tolist(), expected)

    def test_invalid_empty_selection_and_options(self):
        with self.assertRaises(ValueError):
            select_support(torch.ones(1, 1, 64, 4), torch.ones(1, 1, 64, 4),
                           torch.empty(0, dtype=torch.long), 64, 0, .5)
        for option in (Options(layer=32), Options(keep=1), Options(keep=0),
                       Options(graph=True,graph_start=0,graph_warmups=0),Options(graph_start=-1)):
            with self.assertRaises(ValueError):option.validate(32)

    def test_prepared_and_dynamic_position_rope_is_exact_and_read_only(self):
        torch.manual_seed(8)
        past = torch.randn(1, 2, 5, 4, dtype=torch.bfloat16)
        reference = past.clone()
        q = torch.randn(1, 2, 3, 4, dtype=torch.bfloat16)
        k = torch.cat((past, torch.randn_like(q)), -2)
        positions = torch.tensor([0, 2, 7])
        qpos = positions+5
        kpos = torch.cat((torch.arange(5), qpos))
        base = FakeRotary()
        for prepared in (False, True):
            engine = DeviceRotary(base, qpos, kpos, 13, past, prepared)
            for values in ([0, 2, 7], [0, 3, 6]):
                positions.copy_(torch.tensor(values))
                qpos.copy_(positions+5)
                kpos[5:].copy_(qpos)
                expected = PositionedRotary(base, positions, 8, 5)(q, k)
                actual = engine(q, k)
                self.assertTrue(all(torch.equal(a, b) for a, b in zip(actual, expected)))
                self.assertTrue(torch.equal(past, reference))

    def test_commit_native_boundary_fallback_and_tie(self):
        x = torch.full((1, 6), MASK_ID)
        target = torch.tensor([1, 3, 5])
        logits = torch.tensor([[[3., 0.], [1., 1.], [3., 0.]]])
        confidence = logits.double().softmax(-1)[0, 0, 0].item()
        positions, tokens = commit(x, target, logits, confidence)
        self.assertEqual(positions.tolist(), [1, 5])
        self.assertEqual(tokens.tolist(), [0, 0])
        self.assertEqual(x[0, 3].item(), MASK_ID)
        for fused in (False, True):
            x.fill_(MASK_ID)
            positions, tokens = commit(x, target, logits, .99999, fused)
            self.assertEqual(positions.tolist(), [1])
            self.assertEqual(tokens.tolist(), [0])

    def test_segment_exception_restores_rotary(self):
        from .focus_v4_runtime import BlockEngine
        block = SimpleNamespace(rotary_emb='original')
        with self.assertRaises(RuntimeError):
            with BlockEngine.rotary_scope(None, [block], ['private']):
                self.assertEqual(block.rotary_emb, 'private')
                raise RuntimeError('test cleanup')
        self.assertEqual(block.rotary_emb, 'original')

    def test_eager_segment_no_hidden_setup_forward(self):
        from .focus_v4_runtime import Segment
        calls=[]
        segment=Segment(lambda:calls.append(True),False)
        segment.prepare(warmups=0)
        self.assertEqual(calls,[])
        segment.call()
        self.assertEqual(calls,[True])
        segment.close()

    def test_paired_report_does_not_hide_unknown_scores_or_action_changes(self):
        from .focus_v4_evaluate import paired_summary
        rows = []
        for index in range(2):
            rows.append(dict(focus_control=dict(seconds=2.,nfe=10,truncated=False),
                io_borrow=dict(seconds=1.,nfe=10,truncated=False),
                parity={'io_borrow':dict(tokens=True,actions=index==0,nfe=True)},
                clean={'io_borrow':[dict(tokens_equal_trace=True,nfe=10)]},
                historical={n:dict(text=index==0,nfe=index==0) for n in ('focus_control','io_borrow')}))
        score = dict(correct={n:[True,None] for n in ('focus_control','io_borrow')},policy='test')
        report = paired_summary(rows,score,['focus_control','io_borrow'])
        self.assertEqual(report['metrics']['io_borrow']['unresolved_scores'],1)
        self.assertEqual(report['metrics']['io_borrow']['scored_examples'],1)
        self.assertFalse(report['exact_trajectory']['io_borrow'])
        self.assertTrue(report['score_agreement']['io_borrow'])
        self.assertEqual(report['speedups']['io_borrow'],2.)
        self.assertEqual(report['historical_agreement']['io_borrow']['text'],1)

    def test_duplicate_frozen_ids_are_rejected(self):
        from .focus_v4_evaluate import historical_records
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            out=root/'math_g256_ours/output';out.mkdir(parents=True)
            row=json.dumps(dict(id='x',flash_focus_head=dict(nfe=1,text='a')))
            (out/'rank_elastic.jsonl').write_text(row+'\n'+row+'\n')
            with self.assertRaises(ValueError):historical_records(root,'math',256)


if __name__ == '__main__':unittest.main()

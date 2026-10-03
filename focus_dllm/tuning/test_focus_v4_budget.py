import unittest
from types import SimpleNamespace

import torch

from .focus_v4_budget import BudgetEngine, budget_order, select_budget
from .focus_v4_runtime import DEFAULT_OPTIONS, Segment
from .focus_v4_quality import NAMES, paired_quality, summarize, validate_protocol


class Tests(unittest.TestCase):
    def test_query_specific_budget_does_not_hide_a_query_in_the_mean(self):
        future = torch.tensor([[.6, .2, 0., 0.], [0., 0., .075, .075]], dtype=torch.float64)
        order, count, remaining = budget_order(future, .1)
        self.assertEqual(order.tolist(), [0, 1, 2, 3])
        self.assertEqual(int(count), 3)
        # Global-mean mass accepts2 positions; the second query loses .15.
        self.assertLessEqual(float(remaining[:, 2].mean()), .1)
        self.assertGreater(float(remaining[1, 2]), .1)
        self.assertTrue((remaining[:, int(count)]<=.1).all())

    def test_shortest_order_prefix_and_monotone_budget(self):
        torch.manual_seed(412)
        for queries in (1, 5, 32):
            future = torch.softmax(torch.randn(queries, 17), -1)*.7
            previous = 18
            for budget in (0., .001, .02, .1, .4, .7, 1.):
                order, required, remaining = budget_order(future, budget)
                count = int(required)
                self.assertLessEqual(count, previous)
                self.assertTrue((remaining[:, count]<=budget).all())
                if count:
                    self.assertTrue((remaining[:, count-1]>budget).any())
                self.assertEqual(sorted(order.tolist()), list(range(17)))
                previous = count

    def test_zero_and_all_discarded_budgets(self):
        future = torch.tensor([[.2, .1, .05]], dtype=torch.float64)
        self.assertEqual(int(budget_order(future, 0.)[1]), 3)
        self.assertEqual(int(budget_order(future, 1.)[1]), 0)
        zeros = torch.zeros(3, 4)
        order, count, _ = budget_order(zeros, 0.)
        self.assertEqual(int(count), 0)
        self.assertEqual(order.tolist(), [0, 1, 2, 3])

    def test_invalid_budget_or_geometry(self):
        for budget in (-.1, 1.1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                budget_order(torch.ones(2, 4), budget)
        for value in (torch.ones(3), torch.empty(0, 5), torch.empty(2, 0)):
            with self.assertRaises(ValueError):
                budget_order(value, .1)

    def test_actual_qk_selection_protects_entire_current_block_and_prefix(self):
        torch.manual_seed(39)
        q = torch.randn(1, 3, 96, 4)
        k = torch.randn(1, 3, 103, 4)
        original_q, original_k = q.clone(), k.clone()
        for targets in (torch.tensor([31]), torch.tensor([0, 7, 18, 31])):
            for budget in (0., .02, .2, 1.):
                kept, dropped = select_budget(q, k, targets, 96, 7, budget)
                self.assertEqual(kept[:32].tolist(), list(range(32)))
                self.assertEqual(kept.tolist(), sorted(set(kept.tolist())))
                self.assertLessEqual(float(dropped), budget+1e-7)
                self.assertTrue(torch.equal(q, original_q) and torch.equal(k, original_k))
                if budget==0.:
                    self.assertEqual(kept.tolist(), list(range(96)))
                if budget==1.:
                    self.assertEqual(kept.tolist(), list(range(32)))
        with self.assertRaises(ValueError):
            select_budget(q, k, torch.tensor([0]), 64, 7, .1)

    def test_dynamic_deep_mapping_can_grow_and_shrink_without_losing_absolute_positions(self):
        # Exercise changing maps/head selection, not a mock of Transformer quality.
        engine = object.__new__(BudgetEngine)
        engine.ids = torch.zeros((1, 96), dtype=torch.long)
        engine.past_length, engine.length = 7, 96
        hidden = torch.arange(384).reshape(1, 96, 4).float()
        q, k = torch.zeros(1, 2, 96, 4), torch.zeros(1, 2, 103, 4)
        engine.pre = Segment(lambda:(hidden, q, k), False)
        engine.post = Segment(lambda:engine.hidden, False)
        engine.deep_rope = [SimpleNamespace(query_positions=None, key_positions=None) for _ in range(3)]
        engine.core = SimpleNamespace(config=SimpleNamespace(weight_tying=False, scale_logits=False),
                                      transformer=SimpleNamespace(ff_out=lambda value:value))
        engine.calls, engine.future_counts, engine.future_sizes, engine.dropped = 0, [], [], []
        target = torch.tensor([0, 19, 31])
        lengths = []
        for budget in (.5, .01, 1.):
            engine.budget = budget
            logits = engine.forward(engine.ids, target)
            self.assertTrue(torch.equal(logits, hidden.index_select(1, target)))
            self.assertEqual(engine.qpositions.tolist(), (engine.kept+7).tolist())
            self.assertEqual(engine.kpositions[:7].tolist(), list(range(7)))
            for rotary in engine.deep_rope:
                self.assertIs(rotary.query_positions, engine.qpositions)
                self.assertIs(rotary.key_positions, engine.kpositions)
            lengths.append(engine.kept.numel())
        self.assertGreater(lengths[1], lengths[0])
        self.assertEqual(lengths[2], 32)
        self.assertFalse(DEFAULT_OPTIONS.graph)

    def test_development_protocol_cannot_silently_run_eight_or_old_baselines(self):
        for args in [('v4','humaneval',8,0,256), ('v4','humaneval',128,64,256),
                     ('llada','humaneval',128,0,256), ('v1','humaneval',128,0,256),
                     ('flash','math',128,0,128)]:
            with self.assertRaises(ValueError):
                validate_protocol(*args)
        validate_protocol('v4','humaneval',128,0,256)
        validate_protocol('flash','gsm8k',128,0,512)
        self.assertEqual(NAMES, ['io_borrow','budget05','budget20'])

    def test_unknown_scores_and_incomplete_pairs_stay_explicit(self):
        value = paired_quality([True, None], [True, False], [2.,2.], [1.,1.])
        self.assertEqual(value['paired_examples'], 1)
        self.assertEqual(value['unresolved_pairs'], 1)
        self.assertEqual(value['pooled_speedup'], 2.)
        with self.assertRaises(ValueError):
            paired_quality([True, False], [True], [1.,1.], [1.,1.])

    def test_quality_summary_allows_different_nfe_and_tokens(self):
        rows = []
        for index in range(2):
            row = dict(id=str(index))
            for name,nfe in [('io_borrow',40), ('budget05',30)]:
                row[name] = dict(seconds=2. if name=='io_borrow' else 1., nfe=nfe,
                    token_ids=[1] if name=='io_borrow' else [2], output_tokens=1,
                    first_eos=None, truncated=True)
            rows.append(row)
        cell = dict(correctness={n:{'0':True, '1':False} for n in ('llada','v1','focus')},
                    saved={n:{i:dict(seconds=3.) for i in ('0','1')} for n in ('llada','v1','focus')})
        scored = dict(correct={'io_borrow':[True,False], 'budget05':[True,True]}, policy='test')
        result = summarize(rows, scored, ['io_borrow','budget05'], dict(cells={'humaneval_256':cell}), 'humaneval',256)
        self.assertEqual(result['metrics']['budget05']['accuracy'], 1.)
        self.assertEqual(result['metrics']['budget05']['mean_nfe'], 30)
        self.assertEqual(result['paired']['budget05']['current_io_borrow']['pooled_speedup'], 2.)


if __name__ == '__main__':
    unittest.main()

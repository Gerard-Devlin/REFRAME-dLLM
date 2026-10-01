import unittest
from dataclasses import replace
from .versioned_dataflow import Value,Task,layer_plan,validate_order,ranges


class Tests(unittest.TestCase):
    def test_pipeline_and_phase_orders_both_legal(self):
        tasks,external=layer_plan(3)
        for order in ([f'{kind}:{i}' for kind in ('produce','attention','post') for i in range(3)]+['commit'],
                      [f'produce:{i}' for i in range(3)]+[s for i in range(3) for s in (f'attention:{i}',f'post:{i}')]+['commit']):
            reclaimed=validate_order(tasks,external,order)
            self.assertEqual(len(reclaimed),len(set(reclaimed)))
            self.assertIn(Value('k',2,1),reclaimed)

    def test_attention_cannot_skip_unfinished_kv(self):
        tasks,external=layer_plan(3)
        order=['produce:0','attention:0','produce:1','produce:2','post:0','attention:1','post:1','attention:2','post:2','commit']
        with self.assertRaisesRegex(ValueError,'Unready'):validate_order(tasks,external,order)

    def test_old_kv_is_legal_only_where_prescribed(self):
        tasks,external=layer_plan(3,epoch=4,refreshed={0},cache_epoch=3)
        attention=next(t for t in tasks if t.name=='attention:2')
        self.assertIn(Value('k',0,4),attention.reads);self.assertNotIn(Value('k',0,3),external)
        self.assertIn(Value('k',1,3),attention.reads);self.assertNotIn(Value('k',1,4),attention.reads)
        wrong=tuple(replace(t,reads=tuple(Value(v.role,v.tile,3) if v==Value('k',0,4) else v for v in t.reads)) if t==attention else t for t in tasks)
        order=[f'produce:{i}' for i in range(3)]+[s for i in range(3) for s in (f'attention:{i}',f'post:{i}')]+['commit']
        with self.assertRaisesRegex(ValueError,'wrong-version'):validate_order(wrong,external,order)

    def test_early_commit_rejected(self):
        tasks,external=layer_plan(2)
        with self.assertRaises(ValueError):validate_order(tasks,external,['produce:0','produce:1','attention:0','post:0','commit','attention:1','post:1'])

    def test_duplicate_writers_rejected(self):
        tasks,external=layer_plan(1);bad=tasks+(Task('bad',(),(Value('q',0,1),)),)
        with self.assertRaisesRegex(ValueError,'Conflicting'):validate_order(bad,external,['produce:0','attention:0','post:0','commit','bad'])

    def test_incomplete_plan_rejected(self):
        tasks,external=layer_plan(1)
        with self.assertRaises(ValueError):validate_order(tasks,external,['produce:0'])

    def test_geometry_and_epoch(self):
        self.assertEqual(ranges(257,128),((0,128),(128,256),(256,257)))
        for n,t in ((0,128),(128,0)):
            with self.assertRaises(ValueError):ranges(n,t)
        with self.assertRaises(ValueError):layer_plan(2,epoch=0)
        with self.assertRaises(ValueError):layer_plan(2,refreshed={3})


if __name__=='__main__':unittest.main()

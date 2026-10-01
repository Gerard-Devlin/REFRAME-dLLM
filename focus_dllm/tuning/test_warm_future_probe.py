"""Future-boundary and oracle accounting checks; no model quality claim."""
import unittest
import torch

from .warm_future_probe import NativeObserver, reference_schedule


class ScheduleChecks(unittest.TestCase):
    def events(self):
        return [dict(call=0,block=0,kind='warm',positions=[0],tokens=[10],future=[
            dict(position=2,token=12,confidence=.96),dict(position=3,token=13,confidence=.995)]),
            dict(call=1,block=0,kind='refine',positions=[1],tokens=[11]),
            dict(call=2,block=1,kind='warm',positions=[2],tokens=[12]),
            dict(call=3,block=1,kind='refine',positions=[3],tokens=[13])]

    def run_schedule(self,events):
        return reference_schedule(events,[10,11,12,13],2,[.95,.99],{-1,99},99)

    def test_noncontiguous_future_reference_actions(self):
        s=self.run_schedule(self.events())['thresholds']
        self.assertEqual(s['0.95']['oracle_fully_ready_blocks'],1)
        self.assertEqual(s['0.99']['oracle_fully_ready_blocks'],0)
        self.assertEqual(s['0.99']['oracle_emptied_refine_calls'],1)

    def test_wrong_predictions_are_counted_not_oracle_accepted(self):
        events=self.events();events[0]['future'][1]['token']=88
        s=self.run_schedule(events)['thresholds']['0.95']
        self.assertEqual(s['earliest_wrong'],1)
        self.assertEqual(s['oracle_emptied_refine_calls'],0)

    def test_no_current_block_or_past_observation(self):
        events=self.events();events[0]['future'][0]['position']=1
        with self.assertRaises(ValueError):self.run_schedule(events)

    def test_special_predictions_never_fill_ready_set(self):
        events=self.events();events[0]['future'][1]['token']=99
        s=self.run_schedule(events)['thresholds']['0.95']
        self.assertEqual(s['special_predictions'],1)
        self.assertEqual(s['unique_eligible_positions'],1)
        self.assertEqual(s['oracle_emptied_refine_calls'],0)

    def test_teacher_input_ledger_must_be_complete(self):
        with self.assertRaises(ValueError):self.run_schedule(self.events()[:-1])

    def test_duplicate_teacher_release_rejected(self):
        events=self.events();events[3]['positions']=[2];events[3]['tokens']=[12]
        with self.assertRaises(ValueError):self.run_schedule(events)

    def test_first_prediction_does_not_get_replaced_using_future_truth(self):
        events=self.events();events[0]['future']=[dict(position=4,token=88,confidence=.96)]
        events[2]['future']=[dict(position=4,token=14,confidence=.999)]
        events+=[dict(call=4,block=2,kind='warm',positions=[4],tokens=[14]),
            dict(call=5,block=2,kind='refine',positions=[5],tokens=[15])]
        s=reference_schedule(events,[10,11,12,13,14,15],2,[.95],{-1,99},99)['thresholds']['0.95']
        self.assertEqual(s['earliest_wrong'],1)
        self.assertEqual(s['repeated_conflicts'],1)
        self.assertEqual(s['oracle_emptied_refine_calls'],0)

    def test_post_eos_work_separate(self):
        events=self.events();events[1]['tokens']=[99]
        s=reference_schedule(events,[10,99,12,13],2,[.95],{-1,99},99)
        self.assertEqual(s['first_eos_position'],1)
        self.assertEqual(s['thresholds']['0.95']['post_eos_positions'],2)
        self.assertEqual(s['thresholds']['0.95']['oracle_emptied_refine_before_eos'],0)

    def test_hooks_removed_on_exception_and_input_not_mutated(self):
        model=torch.nn.Identity();prompt=torch.tensor([[1,2]])
        prior=prompt.clone()
        with self.assertRaises(RuntimeError):
            with NativeObserver(model,prompt,4,2,-1):
                self.assertEqual(len(model._forward_hooks),1)
                raise RuntimeError('deliberate')
        self.assertEqual(len(model._forward_hooks),0)
        self.assertEqual(len(model._forward_pre_hooks),0)
        self.assertTrue(torch.equal(prompt,prior))


if __name__=='__main__':unittest.main()

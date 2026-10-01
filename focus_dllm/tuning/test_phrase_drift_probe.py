import unittest
import torch
from .phrase_drift_probe import read_block,finish_trace,MASK_ID


class Tests(unittest.TestCase):
    def test_current_probabilities_and_locked_canvas(self):
        z=torch.full((6,12),-4.,dtype=torch.float64)
        for i in range(6):z[i,i+1]=3.
        canvas=[MASK_ID]*6;canvas[5]=11
        value=read_block(z,canvas)
        self.assertEqual(value['draft'],[1,2,3,4,5,11])
        self.assertFalse(value['active'][5])
        self.assertIsNone(value['placement_scores'][2][2])
        self.assertGreater(value['placement_scores'][0][2],value['placement_scores'][0][3])
        self.assertEqual(canvas,[MASK_ID]*5+[11])

    def test_actual_release_and_fallback(self):
        row=dict(block=0,call=0,full_canvas=[MASK_ID]*4,draft=[1,2,3,4],
                 active=[True]*4,confidence=[.8,.7,.6,.5])
        next_row=dict(block=0,call=1,full_canvas=[1,MASK_ID,MASK_ID,MASK_ID],draft=[1,2,3,4],
                      active=[False,True,True,True],confidence=[1.,.95,.96,.97])
        result=finish_trace([row,next_row],[1,2,3,4],4)
        self.assertEqual(result[0]['commit_positions'],[0])
        self.assertEqual(result[1]['commit_positions'],[1,2,3])

    def test_reject_rollback_and_wrong_actions(self):
        base=dict(block=0,call=0,full_canvas=[1,MASK_ID],draft=[1,2],active=[False,True],confidence=[1.,.7])
        with self.assertRaises(ValueError):finish_trace([dict(base)],[3,2],2)
        with self.assertRaises(ValueError):finish_trace([dict(base)],[1,4],2)
        with self.assertRaises(ValueError):finish_trace([dict(base)],[1,MASK_ID],2)

    def test_partial_placement_score_is_not_full_argmax_path(self):
        p=torch.full((5,12),.9/11,dtype=torch.float64);p[0,1]=.1
        p[1]=.4/11;p[1,1]=.6
        p[2]=0.;p[2,2]=.9;p[2,1]=.1
        p[3]=.05/11;p[3,2]=.95
        p[4]=.01/11;p[4,2]=.99
        value=read_block(p.log(),[MASK_ID]*5)
        self.assertEqual(value['draft'],[1,1,2,2,2])
        self.assertGreater(value['placement_scores'][0][3],value['placement_scores'][0][2])
        moved_full_path=[1,1,1,2,2]
        native=p.max(-1).values.log().sum()
        moved=p.gather(-1,torch.tensor(moved_full_path)[:,None]).log().sum()
        self.assertLess(moved,native)


if __name__=='__main__':unittest.main()

import unittest
from .phrase_drift import candidates,disjoint,spans,analyze_prompt,aggregate

def state(tokens,call=0,canvas=None):
    n=len(tokens)
    return dict(block=0,call=call,canvas=canvas or [126336]*n,draft=tokens,
        active=[True]*n,confidence=[.5]*n,
        placement_scores=[[float(-abs(k-2)) for k in range(5)] for _ in range(n)],
        commit_positions=[],forward_seconds=.01)

class Tests(unittest.TestCase):
    def test_shift_detection_only_current_history_and_fresh_scores(self):
        a=state([9,1,2,3,4,8,7]);b=state([1,2,3,4,8,7,6],1,canvas=[10]+[126336]*6)
        values=candidates(a,b,set())
        self.assertTrue(any(v['tokens']==[1,2,3,4] and v['shift']==-1 for v in values))
        self.assertEqual(candidates(a,dict(b,canvas=a['canvas']),set()),[])
        self.assertTrue(all('final_start' not in v for v in values))

    def test_special_repeat_ambiguous_boundary_and_locked_positions(self):
        self.assertEqual(spans([2,2,2,2],[True]*4,set()),{})
        self.assertEqual(spans([1,2,3,4],[True]*4,{3}),{})
        a=state([1,2,3,4,1,2,3,4]);b=state([9,1,2,3,4,8,7,6],1,canvas=[10]+[126336]*7)
        self.assertFalse(any(v['tokens']==[1,2,3,4] for v in candidates(a,b,set())))
        b['active'][2]=False
        self.assertEqual(candidates(state([1,2,3,4]),state([1,2,3,4],1,canvas=[8]*4),set()),[])
        self.assertEqual(spans([1,2,3,4],[True,False,True,True],set()),{})

    def test_nonoverlap_and_fixed_native_path_comparator(self):
        rows=[dict(confidence=.8,current_start=0,best_current_start=0),
              dict(confidence=.5,current_start=2,best_current_start=2),
              dict(confidence=.6,current_start=4,best_current_start=4)]
        self.assertEqual(len(disjoint(rows)),2)
        a=state([1,2,3,4,8]);b=state([1,2,3,4,8],1,canvas=[10]+[126336]*4)
        self.assertEqual(candidates(a,b,set(),True),[])
        self.assertEqual(len(candidates(a,b,set(),False)),2)

    def test_complete_actions_required_before_oracle(self):
        with self.assertRaises(ValueError):analyze_prompt([],[],set())
        with self.assertRaises(ValueError):analyze_prompt([state([1,2,3,4])],[1,2,3,4],set())
        r=state([1,2,3,4]);r['commit_positions']=[0,1,2,3]
        result=analyze_prompt([r],[1,2,3,4],set())
        self.assertEqual(result['summary']['shifted_candidates'],0)
        self.assertEqual(result['summary']['shifted_cost_oracle']['covered_calls'],0)

    def test_early_content_reference_and_common_phrase_control(self):
        mask=126336
        drafts=[[20,1,2,3,4,25,26,24],[20,21,1,2,3,4,26,24],
                [20,21,1,2,3,4,26,24],[20,21,22,1,2,3,4,24],[20,21,22,1,2,3,4,24]]
        actions=[[0],[7],[1],[2],[3,4,5,6]]
        final=[20,21,22,1,2,3,4,24];canvas=[mask]*8;trace=[]
        for i,(draft,chosen) in enumerate(zip(drafts,actions)):
            row=state(draft,i,canvas=list(canvas));row['active']=[t==mask for t in canvas]
            row['commit_positions']=chosen;trace.append(row)
            for p in chosen:canvas[p]=final[p]
        result=analyze_prompt(trace,final,set())
        early=[r for r in result['shifted'] if r.get('lead_to_absolute_stability',0)>=2]
        self.assertTrue(any(r['tokens']==[1,2,3,4] and r['final_start']==3 for r in early))
        self.assertEqual(result['summary']['early_unique_positions'],4)
        self.assertTrue(all('final_start' not in r for r in candidates(trace[0],trace[1],set())))
        prompts=[dict(trace=trace,analysis=result,forbidden=[]) for _ in range(2)]
        summary=aggregate(prompts)
        self.assertGreater(summary['common_phrase_control']['shared_literal_phrases'],0)
        self.assertEqual(summary['common_phrase_control']['early_noncommon_unique_positions'],0)

    def test_eos_and_repeated_final_locations_cannot_supply_early_reference(self):
        from .phrase_drift import _reference
        candidate=dict(tokens=[1,2,3,4],block=0,call=0,current_start=0,best_current_start=0)
        r=state([1,2,3,4,8,9])
        value=_reference(candidate,[r],[1,2,3,4,126081,8],{i:0 for i in range(6)})
        self.assertTrue(value['correct_current'])
        value=_reference(candidate,[r],[126081,1,2,3,4,8],{i:0 for i in range(6)})
        self.assertFalse(value['content_present_nearby'])

if __name__=='__main__':unittest.main()

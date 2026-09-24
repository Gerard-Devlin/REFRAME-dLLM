import itertools
import random
from types import SimpleNamespace

import pytest
import torch

from relation_update.oracle import ActionObserver, action, max_independent, summarize, aggregate_bounds


def test_weighted_oracle_matches_exhaustive_and_never_skips_adjacent():
    rng = random.Random(9)
    for size in range(10):
        weights = [rng.random() for _ in range(size)]
        allowed = [rng.choice([False, True]) for _ in range(size)]
        expected = max(sum(w*b for w,b in zip(weights,bits))
            for bits in itertools.product([0,1],repeat=size)
            if all(not bit or allowed[i] for i,bit in enumerate(bits))
            and all(not(bits[i] and bits[i+1]) for i in range(size-1)))
        result, selected = max_independent(weights,allowed)
        assert result == pytest.approx(expected)
        assert sum(weights[i] for i in selected) == pytest.approx(expected)
        assert all(allowed[i] for i in selected)
        assert all(b-a>1 for a,b in zip(selected,selected[1:]))


def test_native_action_strict_threshold_tie_forced_and_no_clean_overwrite():
    ids = torch.tensor([2,3,4,5])
    confidence = torch.tensor([.99,.9,.9,.1])
    mask = torch.tensor([False,True,True,True])
    assert action(ids,confidence,mask,.9) == [(1,3)]
    assert action(ids,confidence,torch.zeros(4,dtype=torch.bool),.9) == []
    assert action(ids,confidence,mask,.8) == [(1,3),(2,4)]
    bf16 = torch.tensor([.8984375,.5],dtype=torch.bfloat16)
    assert action(ids[:2],bf16,torch.ones(2,dtype=torch.bool),.9) == [(0,2)]


class ToyNative:
    def forward(self,input_ids,**kwargs):
        return SimpleNamespace(logits=torch.zeros(1,2,12))

    def sample_with_top_p(self,logits,**kwargs):
        p=logits.softmax(-1)
        return p.argmax(-1),p


def native_step(model,state,tokens,confidence,threshold=.9):
    model.forward(input_ids=state,update_past_key_values=False)
    first=int((state[0]==9).nonzero()[0])//2*2
    logits=torch.full((1,2,12),-100.)
    for pos,(token,prob) in enumerate(zip(tokens,confidence)):
        logits[0,pos,token]=torch.log(torch.tensor(prob))
        logits[0,pos,11]=torch.log(torch.tensor(1-prob))
    x,p=model.sample_with_top_p(logits,temperature=0)
    conf=p.gather(-1,x.unsqueeze(-1)).squeeze(-1)[0]
    for index,token in action(x[0],conf,state[0,first:first+2]==9,threshold):
        state[0,first+index]=token


def test_full_observer_verifies_actions_excludes_boundaries_cache_and_last_call():
    model=ToyNative()
    state=torch.tensor([[9,9,9,9]])
    with ActionObserver(model,block_size=4,small_block_size=2,mask_id=9,stop_id=8) as obs:
        native_step(model,state,[1,2],[.8,.7])
        native_step(model,state,[1,2],[.8,.7])
        native_step(model,state,[3,4],[.8,.7])
        native_step(model,state,[3,4],[.8,.7])
        model.forward(input_ids=state,update_past_key_values=True)
    calls=obs.finish()
    assert [c['legal_equal'] for c in calls]==[False,True,False,True,False]
    assert calls[2]['reason']=='subblock_boundary'
    assert calls[4]['reason']=='cache_or_prefill'
    assert all(c['action_verified'] for c in calls[:4])
    assert calls[1]['replay_action']==[(1,2)]


def test_mismatch_is_not_skippable_and_corrupt_action_stops_observer():
    model=ToyNative(); state=torch.tensor([[9,9,9,9]])
    original=model.forward
    with pytest.raises(AssertionError,match='Reconstructed action'):
        with ActionObserver(model,block_size=4,small_block_size=2,mask_id=9,stop_id=8) as obs:
            native_step(model,state,[1,2],[.8,.7])
            native_step(model,state,[1,3],[.8,.7])
            assert not obs.calls[1]['action_equal']
            state[0,1]=4
            model.forward(input_ids=state,update_past_key_values=False)
    assert model.forward==original


def test_eos_and_unverified_terminal_never_skipped():
    model=ToyNative(); state=torch.tensor([[9,9,9,9]])
    with ActionObserver(model,block_size=4,small_block_size=2,mask_id=9,stop_id=8) as obs:
        native_step(model,state,[1,8],[.8,.7])
        native_step(model,state,[1,8],[.8,.7])
    calls=obs.finish()
    assert calls[1]['reason']=='unverified_terminal_action'
    assert not any(c['legal_equal'] for c in calls)


def test_time_bound_includes_non_skippable_cost_and_rejects_bad_timing():
    calls=[dict(forward_seconds=w,legal_equal=a,kind=k,reason=r) for w,a,k,r in (
        (2,False,'prefill_or_cache_write','cache_or_prefill'),
        (1,False,'denoise','no_previous_denoise'),
        (1,True,'denoise','supported'),(1,True,'denoise','supported'))]
    r=summarize(calls,6)
    assert r['selected_calls']==1
    assert r['zero_overhead_modeled_speedup']==pytest.approx(1.2)
    assert r['saved_native_fraction']==pytest.approx(1/6)
    assert not summarize(calls,4)['timing_valid']
    assert summarize(calls,4)['zero_overhead_modeled_speedup'] is None
    totals=aggregate_bounds([summarize(calls,4),summarize(calls,6)])
    assert totals['invalid_timing_prompts']==1
    assert totals['zero_overhead_modeled_speedup'] is None
    assert totals['forward_only_modeled_ceiling']==pytest.approx(10/8)
    assert totals['forward_only_reaches_1_5x'] is False

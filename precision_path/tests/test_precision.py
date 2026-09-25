import copy
import itertools
import json
from types import SimpleNamespace

import pytest
import torch

from precision_path.calibrate import threshold,evaluate,validate,development_curve
from precision_path.common import load_prompts,REVISION
from precision_path.decision import choose,action_list,stability_score,snapshot_cache,assert_cache_unchanged
from precision_path.probe import cost_summary,PairedObserver,bounded_generate
from precision_path.quant import quantize_tensor,FP8Linear


def test_conformal_insufficient_sample_and_strict_ties():
    assert threshold([0.0]*32,.01)==(None,33)
    assert threshold([0.0]*99,.01)==(0.0,99)
    assert threshold([1.,1.,2.],.5)==(1.,2)
    with pytest.raises(ValueError): threshold([float('nan')],.1)
    with pytest.raises(ValueError): threshold([0],0)


def test_rank_bound_by_all_leave_one_out_exchangeable_assignments():
    for scores in ([0.,1.,2.,3.],[0.,0.,1.,1.]):
        for alpha in (.25,.5,.75):
            errors=0
            for i,new in enumerate(scores):
                cutoff,_=threshold(scores[:i]+scores[i+1:],alpha)
                errors+=cutoff is not None and new>cutoff
            assert errors/len(scores)<=alpha


def test_whole_prompt_risk_and_double_payment_on_fallback():
    states=[dict(score=1.,equal=True,bf16_seconds=1.,low_seconds=.4,score_seconds=.02),
            dict(score=2.,equal=False,bf16_seconds=1.,low_seconds=.4,score_seconds=.02)]
    r=evaluate([dict(states=states)],1.)
    assert r['accepted_calls']==1 and r['requests_with_reference_path_acceptance_error']==1
    assert r['ordinary_cost_ratio']==pytest.approx(.92)
    assert evaluate([dict(states=states)],None)['ordinary_cost_ratio']==pytest.approx(1.42)
    curve=development_curve([dict(states=states)])
    assert curve[0]['accept_none'] and curve[0]['accepted_calls']==0
    assert curve[-1]['accepted_calls']==0  # Strict comparison excludes max-score ties.


def test_first_divergence_inclusion_for_finite_deterministic_program():
    # State is the generated prefix; off-reference states may behave arbitrarily.
    for low_bits in itertools.product([0,1],repeat=5):
        scores=[.1,.8,.3,.9,.4]
        risk=max((s for s,b in zip(scores,low_bits) if b),default=0)
        for cutoff in [0.,.2,.5,1.]:
            state=[]
            for i in range(5):
                low=low_bits[i] if not any(state) else 1
                state.append(low if scores[i]>cutoff else 0)
            assert not any(state) or risk>cutoff


def report(role,ids=('a',)):
    rows=[dict(prompt_id=i,complete=True,normal_calls=1,risk_score=.2,
               states=[dict(score=.2,equal=False)]) for i in ids]
    return dict(status='complete',stage='audit',role=role,config_hash='fixed',rows=rows)


def test_calibration_rejects_leakage_config_change_and_partial_trajectories():
    a=report('calibration'); b=report('evaluation',('b',))
    validate(a,b)
    with pytest.raises(ValueError,match='leakage'): validate(a,report('evaluation'))
    b['config_hash']='changed'
    with pytest.raises(ValueError,match='config'): validate(a,b)
    b=report('evaluation',('b',)); b['rows'][0]['normal_calls']=2
    with pytest.raises(ValueError,match='Incomplete'): validate(a,b)
    with pytest.raises(ValueError,match='calibration'): validate(report('development'),report('evaluation',('b',)))


def test_top1_only_does_not_protect_forced_position():
    p=torch.tensor([[.6,.4],[.55,.45]])
    q=torch.tensor([[.51,.49],[.65,.35]])
    mask=torch.ones(2,dtype=torch.bool)
    tokens=p.argmax(-1)
    assert torch.equal(tokens,q.argmax(-1))
    assert not torch.equal(choose(tokens,p,mask,.9)[0],choose(tokens,q,mask,.9)[0])


def test_native_action_strict_threshold_and_clean_mask():
    probs=torch.tensor([[.99,.01],[.9,.1],[.9,.1]])
    tokens=probs.argmax(-1); mask=torch.tensor([False,True,True])
    selected,_=choose(tokens,probs,mask,.9)
    assert action_list(tokens,selected,8)==[(9,0)]


def test_ideal_margin_handles_token_threshold_and_forced_competition():
    torch.manual_seed(72)
    for _ in range(60):
        z=torch.randn(4,9)*3; mask=torch.tensor([False,True,True,True])
        p=z.softmax(-1); tokens=p.argmax(-1); selected,_=choose(tokens,p,mask,.8)
        radius=stability_score(z,mask,selected,.8)
        for _ in range(3):
            shifted=z+(torch.rand_like(z)*2-1)*radius*.8
            q=shifted.softmax(-1); new_tokens=q.argmax(-1); new_selected,_=choose(new_tokens,q,mask,.8)
            assert action_list(tokens,selected)==action_list(new_tokens,new_selected)


def test_cache_mutation_detection_including_growth():
    cache=[(torch.randn(2,3),torch.randn(2,3))]
    saved=snapshot_cache(cache); assert_cache_unchanged(cache,saved)
    cache[0][0][0,0]+=1
    with pytest.raises(AssertionError): assert_cache_unchanged(cache,saved)
    cache.append((torch.zeros(2,3),torch.zeros(2,3)))
    with pytest.raises(AssertionError): assert_cache_unchanged(cache,saved)


def test_real_quantization_dtype_and_cpu_backend_refusal():
    q,s=quantize_tensor(torch.zeros(8,8))
    assert q.dtype==torch.float8_e4m3fn and s>0
    assert torch.equal(q.float(),torch.zeros(8,8))
    with pytest.raises(ValueError,match='CUDA'): FP8Linear(torch.nn.Linear(16,16))


def test_cost_failure_stops_this_backend_even_before_gate():
    rows=[dict(states=[dict(bf16_seconds=1.,low_seconds=.8)])]
    r=cost_summary(rows)
    assert r['rho']==.8 and not r['cost_gate_pass']
    assert r['zero_fallback_all_work_low_ceiling']==1.25


def test_prompt_roles_are_disjoint_and_only_prefix_is_read(tmp_path):
    train=[dict(ids=[i,7,999],prefix=2) for i in range(60)]
    held=[dict(ids=[i,7,888],prefix=2) for i in range(50,70)]
    (tmp_path/'manifest.json').write_text(json.dumps(dict(revision=REVISION)))
    (tmp_path/'train.json').write_text(json.dumps(train))
    (tmp_path/'heldout.json').write_text(json.dumps(held))
    dev=load_prompts(tmp_path,'development',10,1)
    cal=load_prompts(tmp_path,'calibration',10,1)
    test=load_prompts(tmp_path,'evaluation',10,1)
    sets=[{r['id'] for r in rows} for rows in (dev,cal,test)]
    assert not sets[0]&sets[1] and not sets[0]&sets[2] and not sets[1]&sets[2]
    assert all(len(r['ids'])==2 for rows in (dev,cal,test) for r in rows)


class ToyModel:
    def __init__(self,flip=False): self.flip=flip
    def forward(self,input_ids,**kwargs):
        logits=torch.zeros(1,32,4); logits[:,:,int(self.flip)]=2
        return SimpleNamespace(logits=logits)
    __call__=forward
    def sample_with_top_p(self,logits,**kwargs):
        p=logits.softmax(-1); return p.argmax(-1),p


def test_pair_observer_keeps_reference_and_restores_methods(monkeypatch):
    monkeypatch.setattr('precision_path.probe.measured',lambda fn:(fn(),.01))
    reference=ToyModel(); low=ToyModel(True); original=reference.forward
    state=torch.full((1,32),151665,dtype=torch.long)
    with PairedObserver(reference,low,stage='audit',threshold=.9) as observer:
        result=reference.forward(input_ids=state,update_past_key_values=False)
        logits=torch.cat((result.logits[:,:1],result.logits[:,:-1]),dim=1)[:,:8]
        tokens,p=reference.sample_with_top_p(logits,temperature=0)
        assert tokens[0,0]==0
        selected,_=choose(tokens[0],p[0],state[0,:8]==151665,.9)
        state[0,:8][selected]=tokens[0][selected]
        reference.forward(input_ids=state,update_past_key_values=False)
        assert observer.verified==1
    assert reference.forward==original
    assert not observer.rows[0]['equal']
    assert observer.rows[0]['low_action'][0][1]==1


def test_no_progress_generation_fails_and_restores_forward():
    model=ToyModel(); original=model.forward
    def generate(ids,**kwargs):
        while True: model.forward(input_ids=ids)
    model.generate=generate
    with pytest.raises(RuntimeError,match='progress bound'):
        bounded_generate(model,torch.zeros(1,32,dtype=torch.long),dict(max_new_tokens=64,block_size=32))
    assert model.forward==original

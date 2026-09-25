import copy
import math
import pytest
import torch
from step_distill.core import *
from step_distill.model import GraphCache,objective,LoRALinear
from step_distill.trajectory import choose
from step_distill.evaluation import compare

def record():
    return dict(prompt_id='p',history=[[1]*32],canvas=[MASK]*8+[2]*24,start=0,
                first=[[0,3]],second=[[1,4]])

def call(i,canvas,action,seconds=1):
    return dict(index=i,kind='denoise',history=[[1]*32],canvas=canvas,start=0,action=action,seconds=seconds)

def test_windows_no_future_input():
    r=record(); a=call(0,r['canvas'],r['first']); b=call(1,apply_action(a['canvas'],a['action']),r['second'])
    rows=windows(dict(prompt_id='p',calls=[a,b]),{MASK,EOS})
    assert rows[0]['canvas'][1]==MASK
    assert rows[0]['second']==[[1,4]]
    for key in ('canvas','history'):
        assert rows[0][key]==a[key]

@pytest.mark.parametrize('change',['boundary','cache','eos','gap','special','history'])
def test_illegal_pairs(change):
    r=record(); a=call(0,r['canvas'],r['first']); b=call(1,apply_action(a['canvas'],a['action']),r['second'])
    if change=='boundary': b['start']=8
    if change=='cache': b['kind']='cache_write'
    if change=='eos': b['canvas'][10]=EOS
    if change=='gap': b['index']=3
    if change=='special': b['action']=[[1,EOS]]
    if change=='history': b['history']=[[7]*32]
    assert not legal_pair(a,b,{MASK,EOS})

def test_matching_not_double_count():
    canvas=record()['canvas']; calls=[]
    for i in range(5):
        a=[[i,8]]; calls.append(call(i,canvas,a)); canvas=apply_action(canvas,a)
    assert optimal_savings(calls,{MASK,EOS})==2

def test_uniform_windows_include_tail():
    canvas=record()['canvas']; calls=[]
    for i in range(8):
        a=[[i,8]]; calls.append(call(i,canvas,a)); canvas=apply_action(canvas,a)
    w=windows(dict(prompt_id='p',calls=calls),{MASK,EOS},3)
    assert w[0]['first']==[[0,8]] and w[-1]['second']==[[7,8]]

def test_future_leakage_rejected():
    r=record(); r['canvas'][1]=4
    with pytest.raises(ValueError): validate_record(r)

@pytest.mark.parametrize('world',[1,4,5,6])
def test_sampler_exhausts_once_and_tail(world):
    all_ids=[]
    for batch in batches(37,world):
        got=[i for rank in range(world) for i in rank_microbatches(batch,rank,world) if i is not None]
        assert sorted(got)==sorted(batch); all_ids.extend(got)
    assert sorted(all_ids)==list(range(37))

def test_global_normalization_matches_unsharded():
    # DDP averages gradients. World/global-count scale corrects unequal local supervision.
    r=record(); r2=record(); r2['second']=[[1,4],[2,5]]
    counts=loss_counts([r,r2]); world=4
    local_sums=[2.,6.,0.,0.]
    averaged=sum(x*world/counts[0] for x in local_sums)/world
    assert averaged==8/5

def test_graph_cache_keeps_prefix_gradient():
    a=torch.randn(1,1,2,3,requires_grad=True); b=torch.randn_like(a,requires_grad=True)
    c=GraphCache(); c.update(a,a,0); c.update(b,b,0)
    c[0][0].square().sum().backward()
    assert a.grad.abs().sum()>0 and b.grad.abs().sum()>0

@pytest.mark.parametrize('other',[True,False])
def test_chunk_loss_matches_dense_and_second_targets(other):
    torch.manual_seed(1); head=torch.nn.Linear(5,19,bias=False); head.requires_grad_(False)
    h=torch.randn(32,5,requires_grad=True); th=torch.randn(32,5)
    r=record()
    if not other: r['canvas'][2:8]=[2]*6
    actual=objective(h,th,head,r,'release',chunk_size=4)
    z=head(h[:8]).float().log_softmax(-1); t=head(th[:8]).float().log_softmax(-1)
    ce=-z[0,3]-z[1,4]
    kl=(t[2:].exp()*(t[2:]-z[2:])).sum() if other else z.sum()*0
    hinge=torch.relu(torch.tensor(math.log(.91))-torch.stack((z[0,3],z[1,4]))).max()
    expected=torch.stack((ce,kl,hinge))
    assert torch.allclose(actual,expected,atol=2e-6)
    grad=torch.autograd.grad(actual.sum(),h,retain_graph=True)[0]
    ref=torch.autograd.grad(expected.sum(),h)[0]
    assert torch.allclose(grad,ref,atol=2e-6)
    assert grad[1].abs().sum()>0

def test_lora_zero_and_merge_formula():
    torch.manual_seed(4); base=torch.nn.Linear(4,6,bias=False); base.requires_grad_(False)
    lora=LoRALinear(base,rank=2,alpha=4); x=torch.randn(3,4)
    assert torch.equal(lora(x),base(x))
    with torch.no_grad(): lora.b.normal_()
    assert torch.allclose(lora(x),torch.nn.functional.linear(x,base.weight+2*lora.b@lora.a),atol=2e-6)

def test_native_strict_threshold_and_forced_position():
    canvas=[MASK]*8+[3]*24; logits=torch.zeros(8,3)
    logits[3,1]=5
    action=choose(logits,canvas,0,.99)
    assert action==[[3,1]]

def test_lr_and_resume_batch_cursor():
    schedule=batches(51,4)
    cursor=3
    assert schedule[:cursor]+batches(51,4)[cursor:]==schedule
    assert lr_factor(0,100)==.2

def test_screening_requires_speed_and_quality():
    def row(i,time,ok):
        return dict(sample_id=str(i),seconds=time,correct=ok,counts=dict(calls=5,denoise=4,cache_write=1),
                    tokens=[EOS],generated_tokens=5,length_capped=False)
    ref=[row(i,2,True) for i in range(5)]
    assert compare(ref,[row(i,1.5,True) for i in range(5)])['screening_pass']
    assert not compare(ref,[row(i,1.5,i!=0) for i in range(5)])['screening_pass']

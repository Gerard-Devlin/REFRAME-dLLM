from dataclasses import replace
import math
import random
import pytest
from denoising_dag.search import Context,Executor,Prediction,search,compare


CTX=Context('snapshot','revision',tuple(range(8)),'block-causal')


def fixed(states):
    return [Prediction(tuple(range(1,len(s)+1)),(-.2,)*len(s),(.7,)*len(s)) for s in states]


def test_commuting_count_and_preserved_path_multiplicity():
    plain=Executor(CTX,fixed,False)
    memo=Executor(CTX,fixed,True)
    a=search((-1,)*6,plain,-1,3,6,trace=True)
    b=search((-1,)*6,memo,-1,3,6,trace=True)
    assert plain.physical_rows==157
    assert memo.physical_rows==42
    assert a.visits==b.visits==157
    assert a.path==b.path and a.score==b.score
    assert compare(a,b)['pass_']
    assert [x['logical'] for x in memo.layers]==[1,6,30,120]
    assert [x['physical'] for x in memo.layers]==[1,6,15,20]


def test_depth_one_and_order_recording_are_zero_gain_controls():
    memo=Executor(CTX,fixed,True)
    search((-1,)*6,memo,-1,1,6)
    assert memo.logical_rows==memo.physical_rows==7
    def history(states):
        return [Prediction((sum(t!=-1 for t in s),)*len(s),(-.2,)*len(s),(.7,)*len(s)) for s in states]
    memo=Executor(CTX,history,True)
    search((-1,)*8,memo,-1,4,4)
    assert memo.logical_rows==memo.physical_rows==341


def test_interacting_predictors_have_identical_logical_decisions():
    for seed in range(20):
        rng=random.Random(seed)
        bias=[[rng.uniform(-1,1) for _ in range(3)] for _ in range(8)]
        effect=[[[rng.uniform(-.4,.4) for _ in range(3)] for _ in range(8)] for _ in range(8)]
        def predictor(states):
            records=[]
            for s in states:
                values=[[bias[i][v]+sum(effect[i][j][v]*(t+1) for j,t in enumerate(s) if t>=0)
                         for v in range(3)] for i in range(8)]
                probabilities=[[math.exp(v)/sum(math.exp(x) for x in row) for v in row] for row in values]
                records.append(Prediction(tuple(max(range(3),key=lambda j:p[j]) for p in probabilities),
                    tuple(math.log(max(p)) for p in probabilities),
                    tuple(-sum(x*math.log(x) for x in p) for p in probabilities)))
            return records
        a=search((-1,)*8,Executor(CTX,predictor,False),-1,3,4,trace=True)
        b=search((-1,)*8,Executor(CTX,predictor,True),-1,3,4,trace=True)
        assert compare(a,b)['pass_'] and a.score==b.score


def test_keys_include_values_time_revision_positions_and_snapshot():
    executor=Executor(CTX,fixed,True)
    executor.evaluate([(1,-1),(2,-1),(1,-1)])
    assert executor.physical_rows==2 and executor.hits==1
    for change in (dict(snapshot_id='new'),dict(noise_time='1'),dict(model_revision='new'),
                   dict(positions=(2,3)),dict(attention='another'),dict(mode='fp32')):
        executor.context=replace(CTX,**change)
        executor.evaluate([(1,-1)])
    assert executor.physical_rows==8
    executor.context=CTX
    executor.evaluate([(1,-1)])
    assert executor.physical_rows==8


def test_memory_limit_evicts_nothing_and_changes_only_compute_count():
    plain=Executor(CTX,fixed,False)
    limited=Executor(CTX,fixed,True,max_entries=1)
    a=search((-1,)*6,plain,-1,3,6,trace=True)
    b=search((-1,)*6,limited,-1,3,6,trace=True)
    assert len(limited.table)==1
    assert compare(a,b)['pass_']
    with pytest.raises(ValueError,match='exceeds'):
        search((-1,)*6,plain,-1,10,10)


def test_audit_rejects_changed_decisions():
    a=search((-1,)*6,Executor(CTX,fixed,False),-1,2,3,trace=True)
    def different(states):
        return [replace(p,token=(99,)*len(p.token)) for p in fixed(states)]
    b=search((-1,)*6,Executor(CTX,different,True),-1,2,3,trace=True)
    assert not compare(a,b)['pass_']


def test_fully_completed_leaves_require_no_model_call_in_either_path():
    for reuse in (False,True):
        executor=Executor(CTX,fixed,reuse)
        result=search((-1,),executor,-1,3,1,trace=True)
        assert result.state==(1,)
        assert executor.logical_rows==executor.physical_rows==1
        assert result.visits==2

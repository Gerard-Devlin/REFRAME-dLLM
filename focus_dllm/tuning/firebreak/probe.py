"""Private same-state FIREBREAK verification; no new candidate is committed."""
import argparse
import hashlib
import inspect
import json
import statistics
import time
from pathlib import Path
from types import SimpleNamespace
import torch
import torch.nn.functional as F

from .layout import layout
from .engine import prepare, forward, decide
from .attention import streaming, dense_reference
from .boundary import adapted


def cache_equal(blocks, snapshot, n):
    return all(torch.equal(b.k_cache[:n].view(torch.int16), k.view(torch.int16))
               and torch.equal(b.v_cache[:n].view(torch.int16), v.view(torch.int16))
               for b, (k, v) in zip(blocks, snapshot))


def measured(function):
    torch.cuda.synchronize(); started = time.perf_counter()
    result = function()
    torch.cuda.synchronize()
    return result, time.perf_counter()-started


def kernel_check(device):
    torch.manual_seed(19123)
    checks = []
    for count, n in ((1, 1), (5, 13), (16, 129), (31, 193)):
        plan = layout(range(count), range(20, 20+count), cache_length=n, group_count=4)
        h, d, m = 2, 128, plan.count*plan.families
        q = (torch.randn(m, h, d, device=device)*.2).bfloat16()
        bk = (torch.randn(n, h, d, device=device)*.2).bfloat16()
        bv = torch.randn_like(bk)
        dk = (torch.randn(count, h, d, device=device)*.2).bfloat16()
        dv = torch.randn_like(dk)
        mapping = torch.full((n,), -1, device=device, dtype=torch.int32)
        mapping[:count] = torch.arange(count, device=device, dtype=torch.int32)
        choices = torch.tensor(plan.choices(), device=device, dtype=torch.bool)
        actual = streaming(q, bk, bv, dk, dv, mapping, choices)
        reference = dense_reference(q, bk, bv, dk, dv, mapping, choices)
        error = float((actual.float()-reference).abs().max())
        assert torch.isfinite(actual).all() and error < .025, (count, n, error)
        checks.append(dict(candidates=count,cache=n,max_error=error,
                           reference='Dense FP32 diagnostic, not BF16 native equivalence'))
    return checks


def summarize(controls):
    names = ('chain', 'groups2', 'groups4', 'groups4_cross')
    result = dict(windows=len(controls), methods={})
    for name in names:
        rows = [c['methods'][name] for c in controls]
        accepted = sum(len(r['accepted']) for r in rows)
        agreement = sum(r.get('teacher_agree', 0) for r in rows)
        comparable = sum(r.get('teacher_known', 0) for r in rows)
        result['methods'][name] = dict(accepted=accepted,
            diagnostic_teacher_agreement=agreement/comparable if comparable else None,
            teacher_known=comparable, mean_stable_verify_seconds=statistics.mean(r['stable_seconds'] for r in rows) if rows else None,
            mean_gate_seconds=statistics.mean(r['gate_seconds'] for r in rows) if rows else None)
    if controls:
        chain = result['methods']['chain']; cross = result['methods']['groups4_cross']
        # Includes the observed paid proposal, snapshot/meta and gate. This is
        # a SAME-STATE diagnostic ratio, not an online throughput result.
        base_work = sum(c['proposal_seconds']+c['snapshot_seconds']+c['methods']['chain']['setup_seconds']
                        +c['methods']['chain']['stable_seconds']+c['methods']['chain']['gate_seconds'] for c in controls)
        cross_work = sum(c['proposal_seconds']+c['snapshot_seconds']+c['methods']['groups4_cross']['setup_seconds']
                         +c['methods']['groups4_cross']['stable_seconds']+c['methods']['groups4_cross']['gate_seconds'] for c in controls)
        result['matched_progress_per_second_ratio'] = (cross['accepted']/cross_work)/(chain['accepted']/base_work) if chain['accepted'] and cross_work else None
        result['instrumented_matched_chain_seconds'] = base_work
        result['instrumented_cross_seconds'] = cross_work
    result['scope'] = ('Read-only development-state mechanism/cost diagnostic. Teacher agreement is not task '
                       'accuracy. Shadow accepts are not saved model calls or end-to-end acceleration.')
    return result


@torch.no_grad()
def main():
    from ..gpu_contract import check_binding
    from ...common import sha256, write_json
    from ...llada_common import MODEL_ID, REVISION, prompt_ids
    from ..competitors import load_external, load_model, generation_prompt, select_samples
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party', type=Path, required=True)
    parser.add_argument('--datasets', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False)
    binding = check_binding(required=True)
    cls, official, adaptation = load_external(args.third_party, 'flash_verify')
    source = Path(inspect.getsourcefile(inspect.unwrap(official)))
    source_sha = sha256(source)
    model, tokenizer = load_model(SimpleNamespace(method='flash_verify'), cls)
    torch.set_num_threads(1)
    forbidden = set(tokenizer.all_special_ids) | {126336, 126081}
    replacements = [t for t in tokenizer.encode('0 1', add_special_tokens=False) if t not in forbidden]
    assert replacements
    report = dict(model=MODEL_ID, revision=REVISION, binding=binding, official_source_sha256=source_sha,
        adaptation=adaptation, boundary_adapter='Actual masked-window extent; third-party files unchanged',
        implementation={str(p.relative_to(Path(__file__).parent)):sha256(p) for p in Path(__file__).parent.rglob('*.py')},
        configuration=dict(length=256, block=32, threshold=.9, gamma=.8, eta=.05,
                           groups=[1,2,4], seed=51713, prompts_per_task=2, windows_per_prompt=2),
        datasets={task:sha256(path) for task,path in zip(('humaneval','mbpp','math'),args.datasets)},
        kernel_checks=[], prompts=[], controls=[], torch_sdpa_calls=0,
        scope='Private experimental contexts, no online FIREBREAK commits, no task-quality/speed/novelty claim.')
    write_json(args.output/'diagnostic.json', report)
    original_sdpa = F.scaled_dot_product_attention
    def watched(*a, **k):
        report['torch_sdpa_calls'] += 1
        return original_sdpa(*a, **k)
    F.scaled_dot_product_attention = watched
    try:
        report['kernel_checks'] = kernel_check(model.device)
        write_json(args.output/'diagnostic.json', report)
        for task,path in zip(('humaneval','mbpp','math'),args.datasets):
            for sample in select_samples(path,2,0):
                ident = sample.get('id', sample.get('task_id'))
                input_ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                prompt = torch.tensor(input_ids, device=model.device)
                flags = dict(busy=False,calls=0,shadow=0,controls=0,pending=None,
                             proposal_seconds=None,normal_start=None,verify_start=None)
                actions, prompt_controls = [], []
                def action(pos, token):
                    actions.append((pos.detach().cpu().tolist(),token.detach().cpu().tolist()))
                baseline = adapted(official, action)
                def count(_m,_a,_k):
                    flags['shadow' if flags['busy'] else 'calls'] += 1
                count_handle = model.register_forward_pre_hook(count, with_kwargs=True)
                def generate():
                    responses,steps=[None],[0]
                    baseline(model,[prompt],[len(input_ids)],1,responses,steps,gen_length=256,
                        block_length=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
                        verify=True,tokenizer=tokenizer,stop_tokens=[])
                    torch.cuda.synchronize()
                    return responses[0],steps[0]
                try:
                    (clean_text,clean_steps),clean_seconds = measured(generate)
                    clean_calls = flags['calls']; clean_actions = list(actions)
                    flags['calls']=0; actions.clear()
                    def pre(_m, inputs, kwargs):
                        if flags['busy']:
                            return
                        if not kwargs['lengths'][-1]:
                            torch.cuda.synchronize(); flags['normal_start']=time.perf_counter()
                            return
                        if flags['controls'] >= 2:
                            return
                        query=inputs[0][0]; pos=kwargs['positions'][0]
                        search=int((query==126336).sum())
                        tracked=len(query)-2*search
                        if search < 4 or tracked < 0:
                            return
                        raw_tokens=query[tracked:tracked+search].detach().cpu().tolist()
                        raw_positions=pos[tracked:tracked+search].detach().cpu().tolist()
                        paired=[(int(p),int(t)) for p,t in zip(raw_positions,raw_tokens) if t not in forbidden]
                        if len(paired)<4:
                            return
                        positions,tokens=map(tuple,zip(*paired))
                        n=int(torch.cat((pos,kwargs['positions'][1])).max())+1
                        blocks=model.model.transformer.blocks
                        flags['busy']=True
                        try:
                            snapshot,snapshot_seconds=measured(lambda:[(b.k_cache[:n].clone(),b.v_cache[:n].clone()) for b in blocks])
                            results={}
                            first_plan=None; first_prepared=None; first_value=None
                            for name,groups,cross in [('chain',1,False),('groups2',2,False),('groups4',4,False),('groups4_cross',4,True)]:
                                plan=layout(positions,tokens,cache_length=n,group_count=groups,cross=cross,forbidden=forbidden)
                                prepared,setup=measured(lambda:prepare(plan,kwargs['positions'][2],model.device))
                                value,cold=measured(lambda:forward(model,prepared,snapshot));flags['shadow']+=1
                                gate,gate_seconds=measured(lambda:decide(value,plan))
                                if name=='groups4_cross':
                                    first_plan,first_prepared,first_value=plan,prepared,value
                                times=[]
                                for _ in range(3):
                                    repeated,seconds=measured(lambda:forward(model,prepared,snapshot));flags['shadow']+=1
                                    assert torch.equal(value['iso'],repeated['iso'])
                                    if cross:assert torch.equal(value['cross'],repeated['cross'])
                                    times.append(seconds);del repeated
                                results[name]=dict(**gate,setup_seconds=setup,cold_seconds=cold,
                                    stable_seconds=statistics.median(times),gate_seconds=gate_seconds,
                                    query_rows=plan.count*plan.families,private_key_rows=plan.count)
                                if name!='groups4_cross':del value
                            replacement=next(t for t in replacements if t!=tokens[0])
                            altered,_=measured(lambda:forward(model,first_prepared,snapshot,changed_token=replacement));flags['shadow']+=1
                            outside=[i for i,g in enumerate(first_plan.groups) if g!=first_plan.groups[0]]
                            iso_independent=[0]+outside
                            iso_error=float((first_value['iso'][iso_independent].float()-altered['iso'][iso_independent].float()).abs().max())
                            own_cross_error=float((first_value['cross'][0].float()-altered['cross'][0].float()).abs().max())
                            draft_error=float((first_value['draft_hidden'][outside].float()-altered['draft_hidden'][outside].float()).abs().max()) if outside else 0.
                            assert iso_error==own_cross_error==draft_error==0, (iso_error,own_cross_error,draft_error)
                            assert cache_equal(blocks,snapshot,n),'Private verifier wrote the public cache'
                            pending=dict(task=task,id=ident,positions=list(positions),tokens=list(tokens),
                                methods=results,snapshot_seconds=snapshot_seconds,
                                proposal_seconds=flags['proposal_seconds'],cache_length=n,
                                forbidden_iso_max_logit_change=iso_error,own_cross_max_logit_change=own_cross_error,
                                outside_group_draft_max_hidden_change=draft_error,global_cache_unchanged=True,
                                changed_label_index=0,old_label=tokens[0],replacement=replacement,
                                tracked=tracked,search=search)
                            assert pending['proposal_seconds'] is not None
                            flags['pending']=pending; flags['controls']+=1
                            del first_value,altered,snapshot
                        finally:
                            flags['busy']=False
                        torch.cuda.synchronize(); flags['verify_start']=time.perf_counter()
                    def post(_m,inputs,kwargs,output):
                        if flags['busy']:
                            return
                        if not kwargs['lengths'][-1]:
                            torch.cuda.synchronize()
                            if flags['normal_start'] is not None:
                                flags['proposal_seconds']=time.perf_counter()-flags['normal_start']
                            return
                        value=flags['pending']
                        if value is None:
                            return
                        torch.cuda.synchronize(); value['official_verify_seconds']=time.perf_counter()-flags['verify_start']
                        logits=output.logits[0,value['tracked']+value['search']:value['tracked']+2*value['search']]
                        prob=logits.double().softmax(-1)
                        candidate=inputs[0][0,value['tracked']:value['tracked']+value['search']]
                        confidence=prob.gather(1,candidate[:,None]).flatten()
                        value['official_cumulative_accepted']=int((confidence.cumprod(0)>=.8).sum())
                        value['official_high_tail_after_first_failure']=int((confidence[value['official_cumulative_accepted']:]>=.8).sum())
                        prompt_controls.append(value);flags['pending']=None
                        report['controls'].append(value)
                        report['summary']=summarize(report['controls'])
                        write_json(args.output/'diagnostic.json',report)
                        print(json.dumps(dict(task=task,id=ident,windows=len(report['controls']),
                            accepts={k:len(v['accepted']) for k,v in value['methods'].items()})),flush=True)
                    pre_handle=model.register_forward_pre_hook(pre,with_kwargs=True)
                    post_handle=model.register_forward_hook(post,with_kwargs=True)
                    try:seen,seen_steps=generate()
                    finally:pre_handle.remove();post_handle.remove()
                    assert clean_text==seen and clean_steps==seen_steps and clean_calls==flags['calls'] and clean_actions==actions
                    assert flags['pending'] is None and not flags['busy']
                    final={p:t for ps,ts in clean_actions for p,t in zip(ps,ts)}
                    for control in prompt_controls:
                        for result in control['methods'].values():
                            positions=[control['positions'][i] for i in result['accepted']]
                            tokens=[control['tokens'][i] for i in result['accepted']]
                            result['teacher_known']=sum(p in final for p in positions)
                            result['teacher_agree']=sum(final.get(p)==t for p,t in zip(positions,tokens))
                    report['prompts'].append(dict(task=task,id=ident,clean_seconds=clean_seconds,
                        text_sha256=hashlib.sha256(clean_text.encode()).hexdigest(),actual_commits_match=True,
                        clean_calls=clean_calls,observed_calls=flags['calls'],official_iterations=clean_steps,
                        private_full_model_passes=flags['shadow'],windows=len(prompt_controls)))
                    report['summary']=summarize(report['controls'])
                    write_json(args.output/'diagnostic.json',report)
                    print(f'Completed FIREBREAK prompt {len(report["prompts"])}/6',flush=True)
                finally:count_handle.remove()
        assert report['torch_sdpa_calls']==0 and source_sha==sha256(source)
        assert len(report['prompts'])==6 and report['controls']
        write_json(args.output/'diagnostic.json',report)
        (args.output/'complete').write_text('OK\n')
    finally:
        F.scaled_dot_product_attention=original_sdpa


if __name__=='__main__':
    main()

"""Read-only whole-state handoff control on six frozen development trajectories."""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout,suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .budget_evaluate import maximum_error
from .cache import capture_private_kv
from .generation import select_tracked
from .handoff import build_call,summarize,transition
from .mechanism import statistics as predictions
from .query_budget import generate_budget


@torch.no_grad()
def main():
    from focus_dllm.common import sha256,write_json
    from focus_dllm.llada_common import MODEL_ID,REVISION,prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt,load_external,load_model,select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ('third-party','dataset','previous','output'):
        parser.add_argument('--'+key,type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    cls,external,adaptation=load_external(args.third_party,'flash_verify')
    official=Path(inspect.getsourcefile(inspect.unwrap(external)))
    raw,tokenizer=load_model(SimpleNamespace(method='flash_verify'),cls)
    model=ModelReadout(raw,compact=True,minimum=32)
    torch.set_num_threads(1);torch.manual_seed(1234)
    samples=select_samples(args.dataset,16,0)[:6]
    report=dict(model=MODEL_ID,revision=REVISION,binding=check_binding(required=True),
        implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
        official_sha256=sha256(official),dataset_sha256=sha256(args.dataset),adaptation=adaptation,
        config=dict(length=256,seed=51713,offset=0,prompts=6,packet_indices=[0,1,4,8],
                    query_rows_single=64,query_rows_packed=128,head_rows_single=32,head_rows_packed=96),
        scope='Private mechanism control; original coverage_age owns generation. No new online accuracy/speed result.',
        records=[],timing=[],controls=[],prompts=[])
    write_json(args.output/'diagnostic.json',report)
    context={};calls=[0];shadow=[0];private=[False]
    def count(_model,_args):
        calls[0]+=1
        if private[0]:shadow[0]+=1
    handle=raw.register_forward_pre_hook(count)

    def forward(call,paired,*,capture=False):
        query,positions,lengths=call
        if capture:
            with capture_private_kv() as kv:
                value=model(query,use_cache=True,positions=positions,lengths=lengths,
                    focus_head_rows=(0,96 if paired else 32))
        else:
            value=model(query,use_cache=True,positions=positions,lengths=lengths,
                focus_head_rows=(0,96 if paired else 32));kv=None
        logits=value.logits.squeeze(0)
        slices=[logits[:32].detach().clone()]
        if paired:slices.append(logits[64:96].detach().clone())
        return slices,kv

    def observe(current):
        index=len(current['packets']);window=list(current['window'])
        if index not in (0,1,4,8) or len(window)!=32:return
        tracked=select_tracked(current['known'],current['changed'],32)
        proposed=transition(window,[current['proposals'][p][0] for p in window],
                            [current['proposals'][p][1] for p in window])
        started=time.perf_counter()
        first=build_call(current,tracked,window)
        second=build_call(current,tracked,window,proposed)
        paired=build_call(current,tracked,window,proposed,paired=True)
        torch.cuda.synchronize();setup_seconds=time.perf_counter()-started
        saved_objects=[(b.k_cache,b.v_cache) for b in raw.model.transformer.blocks]
        versions=[(k._version,v._version) for k,v in saved_objects]
        saved_current=model.current;private[0]=True
        try:
            packed,packed_kv=forward(paired,True,capture=True)
            one,one_kv=forward(first,False,capture=True)
            two,two_kv=forward(second,False,capture=True)
            expected=[one[0],two[0]]
            packed_stats=[predictions(value) for value in packed]
            serial_stats=[predictions(value) for value in expected]
            base_update=transition(window,serial_stats[0]['confidence'],serial_stats[0]['top1'])
            packed_base=transition(window,packed_stats[0]['confidence'],packed_stats[0]['top1'])
            remaining=[i for i,p in enumerate(window) if p not in set(proposed.positions)]
            def successor(stats):
                if not remaining:return None
                return transition([window[i] for i in remaining],
                    [stats['confidence'][i] for i in remaining],[stats['top1'][i] for i in remaining])
            next_update=successor(serial_stats[1]);packed_next=successor(packed_stats[1])
            errors=[maximum_error(a,b) for a,b in zip(packed,expected)]
            kv_errors=[]
            for part,canonical in enumerate((one_kv,two_kv)):
                kv_errors.append(max(float((joined[part*64:(part+1)*64].float()-single.float()).abs().max())
                    for joined_pair,single_pair in zip(packed_kv,canonical)
                    for joined,single in zip(joined_pair,single_pair)))
            row=dict(**context,packet=index,window=window,tracked=tracked,
                proposed_positions=list(proposed.positions),proposed_tokens=list(proposed.tokens),
                fresh_positions=list(base_update.positions),fresh_tokens=list(base_update.tokens),
                proposal_matches_fresh_update=proposed==base_update,
                packed_actions_equal=base_update==packed_base and next_update==packed_next,
                next_positions=[] if next_update is None else list(next_update.positions),
                next_tokens=[] if next_update is None else list(next_update.tokens),
                projected_eligible_logit_max_errors=errors,private_kv_max_errors=kv_errors,
                branch_top1_matches=[sum(a==b for a,b in zip(x['top1'],y['top1']))
                                    for x,y in zip(packed_stats,serial_stats)],
                first_setup_seconds=setup_seconds,
                successor_condition='Proposed whole-state branch; usable only if full first update matches.')
            if index==0:
                mutated=(paired[0].clone(),paired[1],paired[2])
                j=window.index(proposed.positions[0]);old=int(mutated[0][0,64+j])
                alt=tokenizer.encode('0',add_special_tokens=False)[0]
                if alt==old:alt=tokenizer.encode('1',add_special_tokens=False)[0]
                assert alt!=old and alt!=126336
                mutated[0][0,64+j]=alt
                altered,altered_kv=forward(mutated,True,capture=True)
                error=maximum_error(packed[0],altered[0])
                first_kv_error=max(float((a[:64].float()-b[:64].float()).abs().max())
                    for pair,other in zip(packed_kv,altered_kv) for a,b in zip(pair,other))
                assert error==0 and first_kv_error==0,'private labels reached base branch'
                report['controls'].append(dict(**context,base_logit_error=error,
                    base_private_kv_error=first_kv_error,label_changed=True))
            if context['prompt_index']<2 and index==0:
                timing={name:[] for name in ('single','serial_pair','packed')}
                def run(name):
                    if name=='single':forward(first,False)
                    elif name=='serial_pair':forward(first,False);forward(second,False)
                    else:forward(paired,True)
                for name in timing:run(name)
                names=list(timing)
                for repeat in range(12):
                    for name in names[repeat%3:]+names[:repeat%3]:
                        torch.cuda.synchronize();start=time.perf_counter();run(name)
                        torch.cuda.synchronize();timing[name].append(time.perf_counter()-start)
                report['timing'].append(dict(**context,
                    **{name+'_mean_seconds':statistics.mean(v) for name,v in timing.items()},
                    samples=timing,serial_pair_speedup=statistics.mean(timing['serial_pair'])/statistics.mean(timing['packed']),
                    scope='Stable prepared private forwards/head/clones; includes extra projected tracked rows, excludes online scheduling/commit/setup.'))
            report['records'].append(row)
        finally:
            private[0]=False;model.current=saved_current
        assert all(b.k_cache is k and b.v_cache is v for b,(k,v) in zip(raw.model.transformer.blocks,saved_objects))
        assert versions==[(k._version,v._version) for k,v in saved_objects],'observer mutated public bank'
        row['public_bank_versions_unchanged']=True
        print('WINDOW',context['id'],index,'full_update_match',row['proposal_matches_fresh_update'],
            'actions',row['packed_actions_equal'],'errors',errors,flush=True)

    try:
        for i,sample in enumerate(samples):
            context.clear();context.update(prompt_index=i,id=sample['task_id'])
            calls[0]=0;shadow[0]=0
            ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
            with forbid_sdpa(),suppress_official_prints():
                value=generate_budget(model,tokenizer,external,ids,length=256,tracking='age',observer=observe)
            previous=json.loads((args.previous/f'evaluation/records/{i:04d}_coverage_age.json').read_text())['result']
            assert all(value[key]==previous[key] for key in ('text','raw_token_ids','nfe','packets','query_geometry','tracked_history'))
            assert calls[0]==value['nfe']+shadow[0]
            report['prompts'].append(dict(**context,online_nfe=value['nfe'],shadow_model_calls=shadow[0],
                actual_model_calls=calls[0],text_tokens_actions_nfe_unchanged=True,sdpa_calls=0,
                text_sha256=hashlib.sha256(value['text'].encode()).hexdigest()))
            write_json(args.output/'diagnostic.json',report)
            print('PROMPT',sample['task_id'],'online',value['nfe'],'private',shadow[0],flush=True)
        report['summary']=summarize(report['records'],report['timing'])
        assert report['implementation']=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert sha256(official)==report['official_sha256']
        write_json(args.output/'diagnostic.json',report)
        (args.output/'complete').write_text('OK\n');print('FINAL',json.dumps(report['summary']),flush=True)
    finally:handle.remove()


if __name__=='__main__':main()

"""Finite native-trajectory influence diagnostic, not an event-driven generator."""
import argparse
import gc
import json
import time
from pathlib import Path
import numpy as np
import torch
from .delta_frontier import Capture,frontier_metrics,influence_columns,relative_change,select_edges
from .temporal_batch import reconstruct_states
from .temporal_batch_ceiling import action

CONFIG=dict(length=256,block=32,threshold=.9,prompts_per_task=2,sample_seed=51713,
    edges_per_prompt=3,frontier_budget=.25,significant_relative_drift=.01,
    drift_tolerances=[.001,.01,.05],query_tile=32,control_seed=19073)


def region_masks(canvas,prompt_length,block):
    n=len(canvas);positions=np.arange(n);generated=positions>=prompt_length
    masked=np.asarray(canvas)==126336
    active=(positions>=prompt_length+block*32)&(positions<prompt_length+(block+1)*32)&masked
    return dict(all=np.ones(n,bool),prompt=~generated,decoded=generated&~masked,
                active=active,future_mask=generated&masked&~active)


def describe_drift(absolute,relative,exact,regions):
    a=absolute.numpy();r=relative.numpy();e=exact.numpy();result={}
    for name,region in regions.items():
        if not region.any():continue
        energy=a[region]**2
        result[name]=dict(count=int(region.sum()),exact_unchanged_fraction=float(e[region].mean()),
            relative_median=float(np.median(r[region])),relative_p95=float(np.percentile(r[region],95)),
            relative_max=float(r[region].max()),delta_energy=float(energy.sum()),
            within_tolerance={str(t):float((r[region]<=t).mean()) for t in CONFIG['drift_tolerances']})
    total=float(np.square(a).sum());rank=np.sort(np.square(a))[::-1]
    count99=int(np.searchsorted(np.cumsum(rank),.99*total)+1) if total else 0
    result['oracle_99pct_energy_coverage']=min(count99,len(a))/len(a)
    result['warning']='Relative hidden drift and energy do not certify logits, commit actions or task quality.'
    return result


@torch.no_grad()
def capture(model,canvas):
    with Capture(model) as observer:output=model(canvas)
    assert len(observer.rows)==32 and all(set(r)=={'h','out','q','k','v'} for r in observer.rows)
    return output,observer.rows


def check_action(logits,canvas,target,row):
    positions,values,_,_=action(logits[0],canvas[0],target)
    expected=sorted(zip(row['commit_positions'],row['commit_values']))
    # Saved release positions are generation-relative; caller supplies that offset.
    return sorted(zip(positions.tolist(),values.tolist())),expected


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    from ..common import sha256,write_json
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from ..llada_evaluate import load_model
    from ..llada_backend import LLaDAAttentionBackend
    from .competitors import select_samples,generation_prompt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    binding=check_binding(required=True)
    source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    reference=json.loads(args.reference.read_text())
    refs={(p['task'],str(p['id'])):p for p in reference['prompts']}
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    report=dict(configuration=CONFIG,model=MODEL_ID,revision=REVISION,binding=binding,implementation=source,
        datasets={k:sha256(p) for k,p in data.items()},reference_sha256=sha256(args.reference),prompts=[],
        scope='Six reused development prompts, three predeclared within-block real teacher edges each. '
            'Original full uncached BF16 FlashAttention at B1; no new decoder, cache reuse or sampler. '
            'Only legal paper_prompt and earlier true commits construct inputs, never gold/tests/solutions. '
            'No task quality, trajectory losslessness, novelty or online speedup claim.',
        predictor_scope={'embedding_source':'Changed input embedding norms plus old full attention. Input deltas available at layer0; '
                'computing old attention columns at every deep layer scans all old keys and queries. Cache/storage/propagation costs still required.',
            'hidden_source_oracle':'Current changed-position deep hidden deltas from PAID full teacher, optimistic mechanism reference, not free online information.',
            'value_source_oracle':'Current changed-position deep value deltas from PAID full teacher; K/query/normalization/MLP effects omitted.',
            'shuffled_sources':'Same magnitudes at seeded random positions, no answer access.',
            'uniform':'Flat ranking with same mandatory input-change set and same budget.',
            'old_hidden_norm':'Magnitude-only ranking, without observing current hidden.'},
        cost_scope='Native B1 clean calls timed independently after warmup. Captured activations are CPU copies, '
            'shadow attention uses FP32 tile softmax, and all observation/transfer costs are explicitly paid. '
            'Shadow attention is not a replacement execution kernel or native timing. No complete NxN or full activation trace persisted.',
        falsification='Tiny input changes becoming broad deep drift, weak fixed-budget predictor recall, or large omitted drift '
            'reject a simple sparse-frontier assumption. A positive energy signal is only permission for a further cost/decision diagnostic, '
            'not proof of action/quality preservation. No automatic generator or parameter sweep follows.')
    write_json(args.output/'diagnostic.json',report)
    model,tokenizer=load_model('cuda:0');torch.set_num_threads(1);torch.manual_seed(1234)
    with LLaDAAttentionBackend(model,'flash') as backend:
        for task,path in data.items():
            for sample in select_samples(path,2,0,seed=CONFIG['sample_seed']):
                ident=sample.get('id',sample.get('task_id'));saved=refs[(task,str(ident))]
                ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                states,targets=reconstruct_states(ids,saved['trace'],saved['generated_token_ids'])
                # Seeded edge choice uses only call/block indices, not current deltas or future token quality.
                edges=select_edges(saved['trace']);entry=dict(task=task,id=str(ident),prompt_tokens=len(ids),
                    teacher_nfe=saved['clean_nfe'],edges=[])
                report['prompts'].append(entry)
                warm=torch.tensor([states[edges[0]]],device=model.device)
                model(warm);torch.cuda.synchronize();del warm
                for edge_number,i in enumerate(edges):
                    canvases=[torch.tensor([states[j]],device=model.device) for j in (i,i+1)]
                    changed=np.asarray(states[i])!=np.asarray(states[i+1]);s=np.flatnonzero(changed)
                    assert 0<len(s)<=32 and np.all(s>=len(ids))
                    clean=[];timings=[]
                    for j,x in zip((i,i+1),canvases):
                        before=x.clone();torch.cuda.synchronize();start=time.perf_counter();out=model(x)
                        torch.cuda.synchronize();timings.append(time.perf_counter()-start)
                        target=torch.tensor(targets[j],device=model.device)
                        selected=out.logits.index_select(1,target).detach().cpu()
                        got,expected=check_action(selected.to(model.device),x,target,saved['trace'][j])
                        assert got==[(p+len(ids),v) for p,v in expected],'Native replay no longer matches original teacher action'
                        assert torch.equal(before,x);clean.append(selected);del out
                    start=time.perf_counter();oldout,old=capture(model,canvases[0]);torch.cuda.synchronize()
                    capture_times=[time.perf_counter()-start]
                    assert canvases[0].tolist()==[states[i]],'Observation mutated previous canvas'
                    target=torch.tensor(targets[i],device=model.device)
                    assert torch.equal(oldout.logits.index_select(1,target).cpu(),clean[0]),'Observation changed native logits'
                    del oldout
                    start=time.perf_counter();newout,new=capture(model,canvases[1]);torch.cuda.synchronize()
                    capture_times.append(time.perf_counter()-start)
                    assert canvases[1].tolist()==[states[i+1]],'Observation mutated current canvas'
                    target=torch.tensor(targets[i+1],device=model.device)
                    assert torch.equal(newout.logits.index_select(1,target).cpu(),clean[1]),'Observation changed native logits'
                    del newout
                    n=len(states[i]);rng=np.random.default_rng(CONFIG['control_seed']+edge_number)
                    shuffled=rng.choice(n,len(s),replace=False)
                    regions=region_masks(states[i],len(ids),saved['trace'][i]['block'])
                    embed_abs,embed_rel,embed_exact=relative_change(old[0]['h'][0],new[0]['h'][0])
                    assert embed_exact.numpy()[~changed].all(),'Initial hidden differs outside changed input tokens'
                    record=dict(call_before=i,call_after=i+1,block=saved['trace'][i]['block'],changed_positions=s.tolist(),
                        input_changed_fraction=float(changed.mean()),native_clean_call_seconds=timings,
                        captured_call_seconds=capture_times,logits_bitwise_equal=True,native_release_actions_equal=True,
                        embedding_drift=describe_drift(embed_abs,embed_rel,embed_exact,regions),layers=[])
                    for layer,(previous,current) in enumerate(zip(old,new)):
                        absolute,relative,exact=relative_change(previous['out'][0],current['out'][0])
                        source_abs,_,_=relative_change(previous['h'][0],current['h'][0])
                        value_delta=(current['v'][0].float()-previous['v'][0].float()).norm(dim=-1)[:,s]
                        heads=previous['q'].shape[1];weights=torch.zeros(heads,2*len(s),4)
                        weights[:,:len(s),0]=embed_abs[s][None]
                        weights[:,:len(s),1]=source_abs[s][None]
                        weights[:,:len(s),2]=value_delta
                        weights[:,len(s):,3]=embed_abs[s][None]
                        torch.cuda.synchronize();start=time.perf_counter()
                        prediction=influence_columns(previous['q'],previous['k'],np.r_[s,shuffled],weights,model.device,
                            tile=CONFIG['query_tile'])
                        torch.cuda.synchronize();seconds=time.perf_counter()-start
                        denominator=previous['out'][0].float().norm(dim=-1).clamp_min(1e-12)
                        prediction=prediction/denominator[:,None]
                        scores={name:prediction[:,index].numpy() for index,name in enumerate(
                            ('embedding_source','hidden_source_oracle','value_source_oracle','shuffled_sources'))}
                        scores.update(uniform=np.ones(n),old_hidden_norm=previous['h'][0].float().norm(dim=-1).numpy())
                        metrics={name:frontier_metrics(relative.numpy(),score,absolute.numpy()**2,changed,
                            fraction=CONFIG['frontier_budget'],threshold=CONFIG['significant_relative_drift']) for name,score in scores.items()}
                        record['layers'].append(dict(layer=layer+1,drift=describe_drift(absolute,relative,exact,regions),
                            frontier_metrics=metrics,shadow_attention_seconds=seconds,
                            relative_hidden_drift=relative.tolist(),delta_squared_norm=(absolute**2).tolist()))
                    entry['edges'].append(record);write_json(args.output/'diagnostic.json',report)
                    print('EDGE',task,str(ident),i,'changed',len(s),'last-layer drift p95',record['layers'][-1]['drift']['all']['relative_p95'],flush=True)
                    del old,new,clean,canvases;gc.collect();torch.cuda.empty_cache()
        report['backend']=backend.report();assert report['backend']['torch_sdpa_calls']==0 and report['backend']['flash_calls']>0
    assert source=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'Frozen source changed'
    report['complete_edges']=sum(len(p['edges']) for p in report['prompts']);assert report['complete_edges']==18
    write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()

"""Exact-shape batch verification execution-cost control, not an online solver.

Teacher states are paid offline oracles. Graph creation, geometry preparation,
owned cache import and copies are explicitly measured. No E2E speed claim.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import time

import torch

from .batch_verify_probe import decision, pack_states, validate_window, save
from .native_row_ops import NativeRowOps
from .static_dual_geometry import StaticDualGeometry


class Engine:
    def __init__(self,model,window,width,graph,backend):
        self.model,self.width,self.graph_mode,self.backend=model,width,graph,backend
        self.geometry=StaticDualGeometry(window['mask'].expand(width,-1))
        self.ids,self.cache=pack_states(window['states'][:width],window['past'],owned=True)
        self.cuda_graph=None
        self.graph_output=None
        self.replayed_forwards=0
        self.capture_flash_calls=0

    def forward(self):
        return self.model(self.ids,past_key_values=self.cache,use_cache=True,
            replace_position=self.geometry.mask)

    def prepare(self):
        if not self.graph_mode:return
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):self.forward()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        before=self.backend.stats.flash_calls
        sdpa=self.backend.stats.torch_sdpa_calls
        self.cuda_graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.cuda_graph,stream=stream):
            self.graph_output=self.forward()
        torch.cuda.synchronize()
        self.capture_flash_calls=self.backend.stats.flash_calls-before
        if self.capture_flash_calls!=len(self.cache)*self.width or self.backend.stats.torch_sdpa_calls!=sdpa:
            raise AssertionError('Capture did not execute the expected native-row Flash operators')

    def import_window(self,window):
        self.geometry.set_block(window['start'])
        for owned,original in zip(self.cache,window['past']):
            for dest,source in zip(owned,original):
                if dest.data_ptr()==source.data_ptr():
                    raise AssertionError('The authoritative cache cannot be graph scratch storage')
                dest.copy_(source.expand_as(dest))

    def call(self,ids):
        self.geometry.validate()
        if ids.shape!=self.ids.shape or ids.dtype!=self.ids.dtype or ids.device!=self.ids.device:
            raise ValueError('Replay input violates the fixed graph shape/type/device contract')
        self.ids.copy_(ids)
        if self.graph_mode:
            self.cuda_graph.replay()
            self.replayed_forwards+=1
            output=self.graph_output
        else:output=self.forward()
        # Serial graph replays share one output buffer. Apply the same copying
        # cost to eager comparators, so previous results cannot be overwritten.
        return output.logits.clone()

    def close(self):
        torch.cuda.synchronize()
        if self.cuda_graph is not None:self.cuda_graph.reset()
        self.graph_output=None


def median(values):return sorted(values)[len(values)//2]


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from .run import load_model
    from .competitors import select_samples, generation_prompt
    from ..llada_common import MODEL_ID, REVISION, MASK_ID, prompt_ids
    from ..llada_decode import generate_dual_cache
    from ..llada_backend import LLaDAAttentionBackend

    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--repeats',type=int,default=9)
    args=parser.parse_args()
    if args.repeats<5:raise ValueError('At least five stable trials required')
    args.output.mkdir(exist_ok=False)
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    implementation={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    report=dict(model=MODEL_ID,revision=REVISION,gpu_binding=binding,
        implementation=implementation,dataset_sha256=digest(args.dataset),records=[],
        configuration=dict(task='humaneval',sample_seed=51713,offset=0,examples=1,length=256,
            block=32,threshold=.90,widths=[1,4],repeats=args.repeats,cache='dual',
            bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
        scope='One reused development prompt, all its blocks and first four ordinary states/block. '
            'A common static-geometry engineering control. Oracle state inputs are not a drafter. '
            'No online releases, new algorithm, universal exactness certificate or E2E speed claim.',
        timing='Initial packing/cache ownership, geometry preparation, capture/actual-workload warmup '
            'and each later block full cache import are measured separately. Timed windows include '
            'state packing/input copies, full model/head, output clones and native decisions. '
            'Parallel windows additionally compare complete-state edges. Serial windows do not. '
            'Graph buffers are reused across later blocks with fixed addresses. Python Flash counters '
            'do not count graph replay; captured operators and replay-derived counts are explicitly separated.')
    save(args.output/'diagnostic.json',report)
    model,tokenizer=load_model('cuda:0');torch.manual_seed(1234)
    sample=select_samples(args.dataset,1,0)[0]
    ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
    prompt=torch.tensor([ids],device=model.device)
    windows=[];pending={}
    try:
        with LLaDAAttentionBackend(model,'flash') as backend:
            clean=generate_dual_cache(model,prompt,gen_length=256)

            def before(_model,call_args,kwargs):
                past=kwargs.get('past_key_values')
                if past is None:
                    pending['warm']=True
                    return
                window=windows[-1]
                if [t.data_ptr() for pair in past for t in pair]!=window['pointers']:
                    raise AssertionError('Unexpected formal cache inside a block')
                window['ordinary_total']+=1
                if len(window['states'])<4:
                    state=call_args[0] if call_args else kwargs['input_ids']
                    pending['state']=state.detach().clone()

            def after(_model,_args,_kwargs,out):
                if pending.pop('warm',False):
                    start=len(ids)+len(windows)*32
                    mask=torch.zeros(1,len(ids)+256,dtype=torch.bool,device=model.device)
                    mask[:,start:start+32]=True
                    windows.append(dict(start=start,mask=mask,past=out.past_key_values,
                        pointers=[t.data_ptr() for pair in out.past_key_values for t in pair],
                        states=[],logits=[],active_cache=[],ordinary_total=0))
                elif 'state' in pending:
                    window=windows[-1];start=window['start']
                    window['states'].append(pending.pop('state'))
                    window['logits'].append(out.logits.detach().clone())
                    window['active_cache'].append([tuple(t[:,:,start:start+32].detach().clone() for t in pair)
                        for pair in out.past_key_values])

            handles=[model.register_forward_pre_hook(before,with_kwargs=True),
                     model.register_forward_hook(after,with_kwargs=True)]
            try:observed=generate_dual_cache(model,prompt,gen_length=256)
            finally:
                for handle in handles:handle.remove()
            assert torch.equal(clean.output,observed.output) and clean.nfe==observed.nfe
            assert len(windows)==8
            report['teacher']=dict(id=sample.get('id',sample.get('task_id')),tokens_match=True,nfe=clean.nfe,
                output_sha256=hashlib.sha256(json.dumps(clean.output[0,len(ids):].tolist()).encode()).hexdigest(),
                block_ordinary_calls=[w['ordinary_total'] for w in windows])
            snapshots=[[t.detach().cpu().clone() for pair in w['past'] for t in pair] for w in windows]
            versions=[[t._version for pair in w['past'] for t in pair] for w in windows]
            save(args.output/'diagnostic.json',report)
            for width in (1,4):
                eligible=[w for w in windows if len(w['states'])>=width]
                if not eligible:
                    report['records'].append(dict(width=width,status='No prespecified block has enough states'))
                    continue
                for graph in (False,True):
                    torch.cuda.synchronize();started=time.perf_counter()
                    engine=Engine(model,eligible[0],width,graph,backend)
                    record=dict(width=width,graph=graph,blocks=[],initialization_seconds=None,
                        geometry_addresses=engine.geometry.addresses)
                    report['records'].append(record)
                    with engine.geometry.scope(model), (NativeRowOps(model) if width==4 else nullcontext()) as row_ops:
                        try:
                            engine.prepare()
                            torch.cuda.synchronize()
                            record['initialization_seconds']=time.perf_counter()-started
                            save(args.output/'diagnostic.json',report)
                            for window in windows:
                                if len(window['states'])<width:
                                    record['blocks'].append(dict(block=(window['start']-len(ids))//32,
                                        status='Unavailable state window, not replaced',ordinary_calls=window['ordinary_total']))
                                    continue
                                rows=window['states'][:width];validate_window(rows,32,MASK_ID)
                                torch.cuda.synchronize();before_import=time.perf_counter()
                                engine.import_window(window)
                                torch.cuda.synchronize();import_seconds=time.perf_counter()-before_import
                                addresses=engine.geometry.addresses
                                assert addresses==record['geometry_addresses']

                                def run():
                                    output=(engine.call(torch.cat(rows,0)) if width==4 else
                                        torch.cat([engine.call(row) for row in window['states'][:4]],0))
                                    used_rows=rows if width==4 else window['states'][:4]
                                    actions=[decision(output[i],row[0],32,MASK_ID,.90) for i,row in enumerate(used_rows)]
                                    edges=([torch.equal(actions[i][3],used_rows[i+1][0]) for i in range(len(used_rows)-1)]
                                        if width==4 else [])
                                    return output,actions,edges,used_rows

                                torch.cuda.synchronize();before_warmup=time.perf_counter()
                                run()
                                torch.cuda.synchronize();warmup_seconds=time.perf_counter()-before_warmup
                                elapsed=[];torch.cuda.reset_peak_memory_stats()
                                for _ in range(args.repeats):
                                    output=actions=edges=used=None
                                    torch.cuda.synchronize();started=time.perf_counter()
                                    output,actions,edges,used=run()
                                    torch.cuda.synchronize();elapsed.append(time.perf_counter()-started)
                                reference=torch.cat(window['logits'][:len(used)],0)
                                max_error=float((output.float()-reference.float()).abs().max())
                                reference_actions=[decision(reference[i],row[0],32,MASK_ID,.90) for i,row in enumerate(used)]
                                matches=[all(torch.equal(a[k],b[k]) for k in (0,1)) for a,b in zip(actions,reference_actions)]
                                conf_error=max(float((a[2]-b[2]).abs().max()) for a,b in zip(actions,reference_actions))
                                complete_edges=[torch.equal(actions[i][3],used[i+1][0]) for i in range(len(used)-1)]
                                source=window['past'];outside=~window['mask'][0]
                                outside_same=all(torch.equal(value[:,:,outside],source[layer][kind][:,:,outside].expand_as(value[:,:,outside]))
                                    for layer,pair in enumerate(engine.cache) for kind,value in enumerate(pair))
                                indices=range(width) if width==4 else (len(used)-1,)
                                active_matches=[]
                                for i in indices:
                                    cache_row=i if width==4 else 0
                                    active_matches.append(all(torch.equal(
                                        value[cache_row:cache_row+1,:,window['start']:window['start']+32],
                                        window['active_cache'][i][layer][kind])
                                        for layer,pair in enumerate(engine.cache) for kind,value in enumerate(pair)))
                                block_record=dict(block=(window['start']-len(ids))//32,ordinary_calls=window['ordinary_total'],
                                    observed_states=len(used),seconds=elapsed,median_seconds=median(elapsed),
                                    geometry_and_warm_cache_import_seconds=import_seconds,
                                    benchmark_warmup_seconds=warmup_seconds,
                                    max_logit_error=max_error,max_confidence_error=conf_error,
                                    action_match=matches,complete_state_edges=complete_edges,
                                    active_cache_reference_matches=active_matches,outside_cache_same=outside_same,
                                    peak_gib=torch.cuda.max_memory_allocated()/2**30,
                                    graph_geometry_addresses_unchanged=True)
                                record['blocks'].append(block_record)
                                save(args.output/'diagnostic.json',report)
                                print('SUMMARY',width,graph,json.dumps(block_record),flush=True)
                                if max_error or conf_error or not all(matches+complete_edges+active_matches) or not outside_same:
                                    raise AssertionError('Exact native control failed; no silent numeric tolerance or retry')
                            record['captured_flash_calls']=engine.capture_flash_calls
                            record['graph_replayed_forwards']=engine.replayed_forwards
                            record['graph_replay_flash_calls_inferred_from_capture']=engine.replayed_forwards*engine.capture_flash_calls
                            record['row_operator_python_stats']=row_ops.stats.copy() if row_ops else None
                        finally:engine.close()
                    del engine;torch.cuda.empty_cache()
            for snapshot,version,window in zip(snapshots,versions,windows):
                values=[t for pair in window['past'] for t in pair]
                assert version==[t._version for t in values]
                assert all(torch.equal(a,b.detach().cpu()) for a,b in zip(snapshot,values))
            report['formal_teacher_caches_unchanged']=True
            report['backend_python_calls_only']=backend.report()
            assert backend.stats.torch_sdpa_calls==0
            report['capture_reused_across_blocks']=True
            report['comparisons']=[];report['setup_inclusive_accounting']=[]
            for graph in (False,True):
                serial=next(r for r in report['records'] if r['width']==1 and r.get('graph')==graph)
                batch=next(r for r in report['records'] if r['width']==4 and r.get('graph')==graph)
                common=[]
                for a,b in zip(serial['blocks'],batch['blocks']):
                    assert a['block']==b['block']
                    if a.get('observed_states')==4 and b.get('observed_states')==4:
                        common.append((a,b))
                        report['comparisons'].append(dict(graph=graph,block=a['block'],
                            zero_draft_fixed_window_ratio=a['median_seconds']/b['median_seconds'],
                            import_included_ratio=(a['median_seconds']+a['geometry_and_warm_cache_import_seconds'])/
                                (b['median_seconds']+b['geometry_and_warm_cache_import_seconds']),
                            scope='No draft/acceptance or generation timing; setup remains separately charged'))
                if common:
                    totals=[r['initialization_seconds']+sum(
                        pair[i]['median_seconds']+pair[i]['geometry_and_warm_cache_import_seconds']
                        for pair in common) for i,r in enumerate((serial,batch))]
                    warm_totals=[totals[i]+sum(pair[i]['benchmark_warmup_seconds'] for pair in common)
                        for i in (0,1)]
                    report['setup_inclusive_accounting'].append(dict(graph=graph,
                        blocks=[a['block'] for a,b in common],serial_seconds=totals[0],parallel_seconds=totals[1],
                        ratio=totals[0]/totals[1],benchmark_warmups_included_ratio=warm_totals[0]/warm_totals[1],
                        scope='Optimistic offline accounting of one four-state window per eligible block, '
                            'including one graph initialization/request and paid block imports. '
                            'No draft, acceptance distribution or actual online request time. '
                            'Additional reference calls outside these windows are not included.'))
    except Exception as error:
        report['failure']=dict(type=type(error).__name__,message=str(error))
        save(args.output/'diagnostic.json',report)
        raise
    assert implementation=={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    save(args.output/'diagnostic.json',report)
    (args.output/'complete').write_text('OK\n')


if __name__=='__main__':main()

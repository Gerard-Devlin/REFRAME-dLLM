"""Finite layer-16 stream-DAG experiment; not a persistent runtime or decoder."""
import argparse
from contextlib import contextmanager
import gc
import json
from pathlib import Path
import statistics
import time

import torch

from .delta_frontier import select_edges
from .delta_frontier_probe import check_action
from .temporal_batch import reconstruct_states
from .versioned_dataflow import LayerFlow, GraphCall

CONFIG=dict(length=256,block=32,threshold=.9,sample_seed=51713,prompts_per_task=2,
    layer=16,edges='Predeclared middle within-block edge from the same native trajectory',
    tiles=[128,256],repeats=21,inner_replays=10,warmups=3,minimum_layer_speedup=1.05,
    gate='Pipeline must be bitwise equal to both phased tile execution and native layer, '
        'preserve native logits/actions on both real states, and beat native CUDA Graph at matched copy cost. '
        'No multi-layer or generation followup unless all six inputs pass.')


def compare(reference,actual):
    if reference.shape!=actual.shape:raise ValueError('Shape changed')
    delta=reference.float()-actual.float()
    return dict(bitwise_equal=bool(torch.equal(reference,actual)),
        changed_fraction=float((reference!=actual).float().mean()),
        max_absolute=float(delta.abs().max()),
        relative_rms=float(delta.square().sum().sqrt()/reference.float().square().sum().sqrt().clamp_min(1e-30)))


@contextmanager
def replacement(block, owned, call):
    original=block.forward;existed='forward' in block.__dict__
    def forward(x,attention_bias=None,layer_past=None,use_cache=False,replace_position=None):
        if attention_bias is not None or layer_past is not None or use_cache or replace_position is not None:
            raise ValueError('Uncached native layer only')
        owned.copy_(x)
        return call(),None
    block.forward=forward
    try:yield
    finally:
        if existed:block.forward=original
        else:delattr(block,'forward')


def measure(functions,owned,hidden):
    """Balanced order, common resident and input-copy/output-clone comparators."""
    values={mode:{name:[] for name in functions} for mode in ('resident','request_copies')}
    for name,fun in functions.items():
        owned[name].copy_(hidden)
        for _ in range(CONFIG['warmups']):fun()
    torch.cuda.synchronize()
    names=list(functions)
    for mode in values:
        for trial in range(CONFIG['repeats']):
            order=names[trial%len(names):]+names[:trial%len(names)]
            if trial%2:order=list(reversed(order))
            for name in order:
                fun=functions[name]
                torch.cuda.synchronize();start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                began=time.perf_counter();start.record()
                result=None
                for _ in range(CONFIG['inner_replays']):
                    if mode=='request_copies':owned[name].copy_(hidden)
                    result=fun()
                    if mode=='request_copies':result=result.clone()
                end.record();end.synchronize()
                values[mode][name].append(dict(stream_seconds=start.elapsed_time(end)/1000/CONFIG['inner_replays'],
                    wall_seconds=(time.perf_counter()-began)/CONFIG['inner_replays']))
                del result
    return {mode:{name:dict(raw=times,median_stream_seconds=statistics.median(t['stream_seconds'] for t in times),
        median_wall_seconds=statistics.median(t['wall_seconds'] for t in times)) for name,times in row.items()}
        for mode,row in values.items()}


def parse_profile_trace(trace):
    """Use complete CPU replay scopes, never narrower duplicate GPU annotations.

    Kineto emits one gpu_user_annotation per stream with the same profile name.
    Treating those as CPU scopes overwrites the full multi-stream statistics.
    The CPU record includes synchronize(), so it encloses the entire replay.
    """
    events=trace['traceEvents']
    kernels=[e for e in events if e.get('ph')=='X' and e.get('cat')=='kernel' and e.get('dur',0)>0]
    result={};accounted=set()
    regions=[e for e in events if e.get('ph')=='X' and e.get('cat')=='user_annotation'
             and e.get('name','').startswith('profile_')]
    for region in regions:
        if region['name'] in result:raise ValueError('Duplicate CPU profile scope')
        indices=[i for i,e in enumerate(kernels) if region['ts']<=e['ts']
                 and e['ts']+e['dur']<=region['ts']+region['dur']]
        selected=[kernels[i] for i in indices];accounted.update(indices)
        # Relative timestamps avoid subtracting large absolute clock values.
        spans=sorted((e['ts']-region['ts'],e['ts']-region['ts']+e['dur']) for e in selected)
        union=0.;end=-float('inf')
        for a,b in spans:
            union+=max(0,b-max(a,end));end=max(end,b)
        summed=sum(e['dur'] for e in selected)
        result[region['name']]=dict(kernels=len(selected),summed_kernel_us=summed,union_kernel_us=union,
            overlap_kernel_us=max(0.,summed-union),
            note='Complete CPU replay scope; kernel timestamp overlap, not SM occupancy or performance attribution.')
    if not regions:raise ValueError('Missing CPU profile scopes')
    return dict(regions=result,trace_kernels=len(kernels),accounted_kernels=len(accounted),
                unassigned_kernels=len(kernels)-len(accounted))


def profile_graphs(functions,path):
    """Paid CUDA trace, not timing samples or an occupancy/bandwidth estimate."""
    try:
        from torch.profiler import profile,ProfilerActivity,record_function
        with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as prof:
            for name,fun in functions.items():
                if name=='native_eager':continue
                torch.cuda.synchronize()
                with record_function(f'profile_{name}'):
                    fun();torch.cuda.synchronize()
        prof.export_chrome_trace(str(path))
        return dict(path=path.name,**parse_profile_trace(json.loads(path.read_text())))
    except Exception as error:
        return dict(unavailable=f'{type(error).__name__}: {error}',scope='Timing/numerical results retained; no overlap claim without CUDA trace.')


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    from .competitors import select_samples,generation_prompt
    from ..common import sha256,write_json
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from ..llada_evaluate import load_model
    from ..llada_backend import LLaDAAttentionBackend
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    binding=check_binding(required=True)
    source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    refs={(p['task'],str(p['id'])):p for p in json.loads(args.reference.read_text())['prompts']}
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    report=dict(configuration=CONFIG,model=MODEL_ID,revision=REVISION,binding=binding,implementation=source,
        reference_sha256=sha256(args.reference),datasets={k:sha256(p) for k,p in data.items()},records=[],
        scope='One native LLaDA layer at six reused development prompts, two real adjacent states per prompt. '
            'No speculation, pruning, training, stale/new KV guessing, next-step overlap or generator. '
            'Shadow full-model evaluations use legal prompts and earlier actual commits, never gold/tests.',
        implementation_scope='A row-grain CUDA stream/event DAG, captured in CUDA Graph. NOT an SM-level persistent '
            'kernel, MPK adapter, automatic compiler or zero-HBM fusion. All full current K/V must be ready before '
            'native FlashAttention. Only the global attention-to-post/MLP phase barrier is removed. '
            'Changing GEMM query-row geometry can change BF16 arithmetic and is independently audited against native.',
        timing_scope='Same mathematical FLOPs, full attention connectivity and precision flags. Common graph engineering '
            'controls, resident body and matched input-copy/output-clone cost, plus setup/capture and extra memory. '
            'No single-layer speed is called full-generation acceleration. Profile timings are excluded.',
        sources={'MPK':'https://arxiv.org/abs/2512.22219','floating_point':'https://arxiv.org/abs/2609.11356',
                 'stream_capture':'https://docs.pytorch.org/docs/2.7/notes/cuda.html#usage-with-multiple-streams'})
    write_json(args.output/'diagnostic.json',report)
    model,tokenizer=load_model('cuda:0');model.eval();torch.set_num_threads(1)
    precision_flags=dict(bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        tf32=torch.backends.cuda.matmul.allow_tf32)
    report['precision_flags']=precision_flags
    block=model.model.transformer.blocks[CONFIG['layer']-1]
    versions=[p._version for p in block.parameters()]
    with LLaDAAttentionBackend(model,'flash') as backend:
        for task,path in data.items():
            for sample in select_samples(path,2,0,seed=CONFIG['sample_seed']):
                ident=str(sample.get('id',sample.get('task_id')));saved=refs[(task,ident)]
                ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                states,targets=reconstruct_states(ids,saved['trace'],saved['generated_token_ids'])
                edge=select_edges(saved['trace'])[1];inputs=[];outputs=[];native_logits=[]
                for j in (edge,edge+1):
                    captured={}
                    def pre(module,args):captured['input']=args[0].detach().clone()
                    def post(module,args,value):captured['output']=value[0].detach().clone()
                    hooks=[block.register_forward_pre_hook(pre),block.register_forward_hook(post)]
                    canvas=torch.tensor([states[j]],device=model.device)
                    try:out=model(canvas)
                    finally:
                        for hook in hooks:hook.remove()
                    target=torch.tensor(targets[j],device=model.device)
                    logits=out.logits.index_select(1,target).cpu()
                    got,expected=check_action(logits.to(model.device),canvas,target,saved['trace'][j])
                    assert got==[(p+len(ids),v) for p,v in expected]
                    inputs.append(captured['input']);outputs.append(captured['output']);native_logits.append(logits)
                    del out
                record=dict(task=task,id=ident,call_before=edge,call_after=edge+1,length=inputs[0].shape[1],
                    native_actions_match_original=True,tiles={},setup={})
                native_input=inputs[0].clone()
                native_eager=lambda:block(native_input)[0]
                native_graph=GraphCall(native_eager)
                began=time.perf_counter();native_graph.prepare();record['setup']['native_graph_seconds']=time.perf_counter()-began
                functions=dict(native_eager=native_eager,native_graph=native_graph)
                owned={name:native_input for name in functions};flows={};graphs=[]
                baseline_allocated=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
                for tile in CONFIG['tiles']:
                    began=time.perf_counter();flow=LayerFlow(block,inputs[0],tile);flows[tile]=flow
                    phase=GraphCall(flow.phased);pipe=GraphCall(flow.pipeline)
                    phase.prepare();pipe.prepare();graphs.extend([phase,pipe])
                    record['setup'][f'tile{tile}_allocation_capture_seconds']=time.perf_counter()-began
                    functions[f'phase_{tile}']=phase;functions[f'flow_{tile}']=pipe
                    owned[f'phase_{tile}']=owned[f'flow_{tile}']=flow.x
                    tile_record=dict(parts=flow.parts,audits=[])
                    record['tiles'][str(tile)]=tile_record
                    # A -> B -> A verifies that captured buffers never silently retain old values.
                    for state_index in (0,1,0):
                        hidden=inputs[state_index]
                        native_input.copy_(hidden);native=native_graph().clone()
                        flow.x.copy_(hidden);phased=phase().clone();pipelined=pipe().clone();torch.cuda.synchronize()
                        row=dict(state=state_index,native_graph=compare(outputs[state_index],native),
                            phased_vs_native=compare(outputs[state_index],phased),
                            pipeline_vs_native=compare(outputs[state_index],pipelined),
                            pipeline_vs_phased=compare(phased,pipelined))
                        # One-layer substitution checks downstream amplification and native commit actions.
                        canvas=torch.tensor([states[edge+state_index]],device=model.device)
                        with replacement(block,flow.x,pipe):shadow=model(canvas)
                        target=torch.tensor(targets[edge+state_index],device=model.device)
                        shadow_logits=shadow.logits.index_select(1,target).cpu();del shadow
                        got,expected=check_action(shadow_logits.to(model.device),canvas,target,saved['trace'][edge+state_index])
                        row.update(final_logits=compare(native_logits[state_index],shadow_logits),
                            final_commit_actions_equal=got==[(p+len(ids),v) for p,v in expected])
                        tile_record['audits'].append(row)
                    print('AUDIT',task,ident,tile,'native-bitwise',all(a['pipeline_vs_native']['bitwise_equal'] for a in tile_record['audits']),
                        'phased-bitwise',all(a['pipeline_vs_phased']['bitwise_equal'] for a in tile_record['audits']),flush=True)
                record['timings']=measure(functions,owned,inputs[0])
                record['extra_peak_gib']=(torch.cuda.max_memory_allocated()-baseline_allocated)/2**30
                record['profile']=profile_graphs(functions,args.output/f'profile_{len(report["records"])}.json')
                report['records'].append(record);write_json(args.output/'diagnostic.json',report)
                print('CASE COMPLETE',task,ident,flush=True)
                for g in graphs+[native_graph]:g.close()
                for flow in flows.values():flow.close()
                del inputs,outputs,native_logits,functions,owned,flows,graphs,phase,pipe,flow,captured,native_input
                del native,phased,pipelined,hidden,native_eager
                gc.collect();torch.cuda.empty_cache()
        report['backend']=backend.report()
        report['backend_counter_scope']='Python counts eager and captured operators only; graph replays do not increment counters. Captures use Flash, SDPA0.'
        assert report['backend']['torch_sdpa_calls']==0 and report['backend']['flash_calls']>0
    assert versions==[p._version for p in block.parameters()],'Weight mutation'
    assert precision_flags==dict(bf16_reduced_precision_reduction=torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction,
        tf32=torch.backends.cuda.matmul.allow_tf32),'Precision policy mutation'
    assert source=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    assert len(report['records'])==6
    report['gates']={}
    for tile in CONFIG['tiles']:
        audits=[a for r in report['records'] for a in r['tiles'][str(tile)]['audits']]
        speeds=[r['timings']['request_copies']['native_graph']['median_wall_seconds']/
                r['timings']['request_copies'][f'flow_{tile}']['median_wall_seconds'] for r in report['records']]
        exact=all(a['pipeline_vs_native']['bitwise_equal'] and a['native_graph']['bitwise_equal'] and
                  a['pipeline_vs_phased']['bitwise_equal'] and a['final_logits']['bitwise_equal'] and
                  a['final_commit_actions_equal'] for a in audits)
        faster=all(s>=CONFIG['minimum_layer_speedup'] for s in speeds)
        report['gates'][str(tile)]=dict(exact_native_and_phased=exact,all_lengths_faster=faster,
            native_graph_over_pipeline_speedups=speeds,eligible_for_multilayer_test=exact and faster)
    report['complete']=True
    write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()

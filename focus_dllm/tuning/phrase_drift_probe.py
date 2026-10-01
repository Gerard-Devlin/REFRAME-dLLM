"""Observe normal native trajectories, never move or commit any phrase.

Three separately paid trajectories per prompt: clean timing, state-chain audit,
and full 32-token draft observation. No extra Transformer is used to find or
score phrases. The teacher's completed output is only an offline reference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import torch

from .phrase_drift import CONFIG, analyze_prompt, aggregate

MASK_ID=126336


def fingerprint(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def read_block(logits, canvas, mask_id=MASK_ID):
    """Fresh probabilities only. Full vocabulary is ephemeral, not archived."""
    if logits.ndim!=2 or len(canvas)!=len(logits):
        raise ValueError('Aligned local logits and canvas required')
    n=len(canvas);size=CONFIG['phrase_length'];distance=CONFIG['max_shift']
    active=[token==mask_id for token in canvas]
    top=logits.argmax(-1)
    logp=logits.double().log_softmax(-1)
    confidence=torch.softmax(logits.double(),-1).gather(-1,top[:,None]).squeeze(-1).tolist()
    top_list=top.tolist()
    draft=[int(top_list[i]) if on else int(canvas[i]) for i,on in enumerate(active)]
    # Score the SAME literal argmax phrase at nearby positions from CURRENT
    # probabilities. Pure unary score cannot beat the current argmax placement.
    scores=[[None]*(2*distance+1) for _ in range(n)]
    destinations=[];rows=[];columns=[]
    for b in range(n-size+1):
        for delta in range(-distance,distance+1):
            a=b+delta
            if a<0 or a+size>n or not all(active[a:a+size]):continue
            destinations.append((b,delta+distance))
            rows.extend(range(a,a+size));columns.extend(draft[b:b+size])
    if destinations:
        r=torch.tensor(rows,device=logits.device);c=torch.tensor(columns,device=logits.device)
        values=logp[r,c].reshape(-1,size).sum(-1).tolist()
        for (b,offset),value in zip(destinations,values):scores[b][offset]=value
    return dict(active=active,draft=draft,confidence=confidence,placement_scores=scores)


def finish_trace(records, final, block_size=32, mask_id=MASK_ID):
    """Infer the real native releases; fail on changed context/rollback."""
    for i,row in enumerate(records):
        before=row.pop('full_canvas')
        after=records[i+1]['full_canvas'] if i+1<len(records) else final
        if len(before)!=len(after):raise ValueError('Canvas length changed')
        positions=[];values=[]
        for p,(old,new) in enumerate(zip(before,after)):
            if old!=mask_id and new!=old:raise ValueError('Committed teacher context changed')
            if old==mask_id and new!=old:
                if p//block_size!=row['block']:raise ValueError('Native release crossed block boundary')
                local=p%block_size
                if new!=row['draft'][local]:raise ValueError('Native release differs from captured argmax')
                positions.append(p);values.append(new)
        if not positions:raise ValueError('No native progress')
        row['commit_positions']=positions;row['commit_values']=values
        # Independently check threshold semantics, including argmax fallback.
        active=[j for j,on in enumerate(row['active']) if on]
        chosen=[j for j in active if row['confidence'][j]>=CONFIG['threshold']]
        best=max(active,key=lambda j:row['confidence'][j])
        if best not in chosen:chosen.append(best)
        expected={row['block']*block_size+j for j in chosen}
        if expected!=set(positions):raise ValueError('Threshold/fallback action changed')
    return records


class Observer:
    def __init__(self, model, prompt_length, capture=False):
        self.model=model;self.prompt_length=prompt_length;self.capture=capture
        self.chain=[];self.records=[]

    def __call__(self, canvas):
        before=canvas.detach().clone()
        self.chain.append(fingerprint(before))
        local=before[0,self.prompt_length:].tolist()
        active=[i for i,t in enumerate(local) if t==MASK_ID]
        if not active:raise ValueError('Normal model called with completed canvas')
        block=active[0]//CONFIG['block'];start=block*CONFIG['block']
        if self.capture:torch.cuda.synchronize();started=time.perf_counter()
        result=self.model(canvas)
        if self.capture:
            torch.cuda.synchronize();seconds=time.perf_counter()-started
            read_started=time.perf_counter()
            logits=result.logits[0,self.prompt_length+start:self.prompt_length+start+CONFIG['block']]
            fields=read_block(logits,local[start:start+CONFIG['block']])
            torch.cuda.synchronize();read_seconds=time.perf_counter()-read_started
            self.records.append(dict(call=len(self.records),block=block,canvas=local[start:start+CONFIG['block']],
                full_canvas=local,forward_seconds=seconds,observation_seconds=read_seconds,**fields))
        assert torch.equal(canvas,before),'Observation mutated native input'
        return result


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding=check_binding(required=True)
    from ..common import sha256,write_json
    from ..llada_common import MODEL_ID,REVISION,prompt_ids
    from ..llada_evaluate import load_model
    from ..llada_decode import generate
    from ..llada_backend import LLaDAAttentionBackend
    from .competitors import select_samples,generation_prompt
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    implementation={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    old=json.loads(args.reference.read_text())
    references={(p['task'],str(p['id'])):p for p in old['prompts']}
    report=dict(model=MODEL_ID,revision=REVISION,binding=binding,configuration=CONFIG,
        implementation=implementation,datasets={k:sha256(p) for k,p in data.items()},
        prior_reference_sha256=sha256(args.reference),prompts=[],warmup_model_calls=0,
        scope='Normal full original LLaDA, BF16 / FlashAttention, no cache/pruning/Flash-Verify/new decoder. '
            'Read-only literal phrase drift diagnostic on six reused development prompts. '
            'Only legitimate paper_prompt enters the model; gold/tests/solutions/final output never enter candidate construction. '
            'Final teacher text is not objective task correctness. No quality, losslessness, novelty or speedup claim.',
        cost_scope='Clean original request timing is reported separately. Observed model-call intervals are synchronized, '
            'not pure kernel time. CPU matching/readout is extra diagnostic overhead, not baseline latency. '
            'Oracle-covered teacher calls do not imply they can be skipped after an online change.')
    write_json(args.output/'diagnostic.json',report)
    model,tokenizer=load_model('cuda:0');torch.set_num_threads(1);torch.manual_seed(CONFIG['seed'])
    forbidden=set(tokenizer.all_special_ids)|{MASK_ID,126081}
    with LLaDAAttentionBackend(model,'flash') as backend:
        for task,path in data.items():
            for sample in select_samples(path,CONFIG['prompts_per_task'],CONFIG['offset'],seed=CONFIG['sample_seed']):
                ident=sample.get('id',sample.get('task_id'))
                ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                prompt=torch.tensor([ids],device=model.device)
                # Paid per-shape warmup, no draft or reference construction.
                warm=torch.full((1,len(ids)+CONFIG['length']),MASK_ID,device=model.device,dtype=torch.long)
                warm[0,:len(ids)]=prompt
                model(warm);torch.cuda.synchronize();report['warmup_model_calls']+=1
                kwargs=dict(gen_length=CONFIG['length'],block_length=CONFIG['block'],threshold=CONFIG['threshold'])
                clean=generate(model,prompt,**kwargs)
                audit=Observer(model,len(ids),False);audited=generate(audit,prompt,**kwargs)
                observer=Observer(model,len(ids),True);observed=generate(observer,prompt,**kwargs)
                assert clean.nfe==audited.nfe==observed.nfe
                assert torch.equal(clean.output,audited.output) and torch.equal(clean.output,observed.output)
                assert audit.chain==observer.chain,'All native input states must match under observation'
                final=clean.output[0,len(ids):].tolist()
                previous=references[(task,str(ident))]
                assert previous['clean_nfe']==clean.nfe and previous['generated_token_ids']==final,'Native trajectory changed since preceding probe'
                records=finish_trace(observer.records,final)
                started=time.perf_counter();analysis=analyze_prompt(records,final,forbidden);cpu_seconds=time.perf_counter()-started
                p=dict(task=task,id=ident,prompt_tokens=len(ids),clean_nfe=clean.nfe,audited_nfe=audited.nfe,
                    observed_nfe=observed.nfe,clean_seconds=clean.seconds,audited_seconds=audited.seconds,
                    observed_seconds=observed.seconds,teacher_canvas_chain_match=True,teacher_tokens_nfe_match=True,
                    original_threshold_actions_match=True,prior_native_tokens_nfe_match=True,
                    extra_diagnostic_transformer_calls=2*clean.nfe+1,generated_token_ids=final,
                    generated_text=tokenizer.decode(final,skip_special_tokens=False),forbidden=sorted(forbidden),
                    cpu_analysis_seconds=cpu_seconds,observation_readout_seconds=sum(r['observation_seconds'] for r in records),
                    trace=records,analysis=analysis)
                report['prompts'].append(p);write_json(args.output/'diagnostic.json',report)
                print('PROMPT_COMPLETE',task,ident,json.dumps(analysis['summary']),flush=True)
        report['backend']=backend.report()
        assert report['backend']['torch_sdpa_calls']==0 and report['backend']['flash_calls']>0
    assert implementation=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'Frozen source changed'
    report['summary']=aggregate(report['prompts'])
    report['outcome']='Initial mechanism signal only; online cost/quality testing still unproven' if report['summary']['initial_gate_pass'] else 'No initial mechanism gate; no generator or few-step experiment launched'
    write_json(args.output/'diagnostic.json',report)
    (args.output/'complete').write_text('OK\n');print('COMPLETE',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()

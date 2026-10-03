"""Separate final-normalization and BF16 head-shape errors, without extra layers.

All alternatives project the same current teacher hidden states. They never
affect the sampler. Padding is a numerical control, not a deployed speedup.
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from ..llada_backend import LLaDAAttentionBackend
from ..llada_common import MASK_ID, MODEL_ID, REVISION, prompt_ids
from ..llada_decode import generate_prefix_cache
from .competitors import select_samples, generation_prompt, digest, write
from .focus_v2_probe import compare
from .run import load_model
from .gpu_contract import check_binding


def project(model,hidden):
    core,tr = model.model,model.model.transformer
    result = F.linear(hidden,tr.wte.weight) if core.config.weight_tying else tr.ff_out(hidden)
    if core.config.scale_logits:
        result = result * (1/math.sqrt(core.config.d_model))
    return result


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    binding=check_binding(required=True)
    model,tokenizer=load_model('cuda:0')
    report=dict(model=MODEL_ID,revision=REVISION,records=[],prompts=[],gpu_binding=binding,
        source_sha256=digest(__file__),dataset_sha256=digest(args.dataset),
        scope='Only head projections of two reused dev trajectories; no speed or quality claim')
    for sample in select_samples(args.dataset,2,0):
        ids=prompt_ids(tokenizer,generation_prompt(sample),'humaneval',preformatted=True)
        memo=dict(block=-1,refine=0,shadow=False)
        norm=model.model.transformer.ln_f
        def before(_module,inputs):
            if not memo['shadow']:memo['before']=inputs[0]
        def after(_module,_inputs,value):
            if not memo['shadow']:memo['after']=value
        def observe(_model,inputs,kwargs,value):
            canvas=inputs[0]
            if kwargs.get('past_key_values') is None:
                memo['block']+=1;memo['refine']=0
                start=len(ids)+memo['block']*32
            else:
                memo['refine']+=1;start=0
            targets=(canvas[0,start:start+32]==MASK_ID).nonzero().flatten()+start
            hidden,normalized=memo['before'],memo['after']
            baseline=value.logits.index_select(1,targets)
            selected=normalized.index_select(1,targets)
            memo['shadow']=True
            try:
                local_norm=norm(hidden.index_select(1,targets))
                alternatives=dict(full_reproject=project(model,normalized).index_select(1,targets),
                    after_norm=project(model,selected),before_norm=project(model,local_norm))
                for length in (32,64,128,256):
                    padded=F.pad(selected,(0,0,0,max(0,length-selected.shape[1])))
                    alternatives['pad'+str(length)]=project(model,padded)[:,:targets.numel()]
                scattered=torch.zeros_like(normalized)
                scattered.index_copy_(1,targets,selected)
                alternatives['full_scatter']=project(model,scattered).index_select(1,targets)
                row=dict(id=sample.get('id',sample.get('task_id')),block=memo['block'],
                    refine=memo['refine'],full_rows=canvas.shape[1],active_rows=targets.numel(),
                    norm_max_error=float((selected.float()-local_norm.float()).abs().max()),
                    methods={name:compare(baseline,other,targets) for name,other in alternatives.items()})
                assert row['methods']['full_reproject']['max_logit_error']==0.,'Head reconstruction changed'
                report['records'].append(row)
            finally:
                memo['shadow']=False
        handles=[norm.register_forward_pre_hook(before),norm.register_forward_hook(after),
                 model.register_forward_hook(observe,with_kwargs=True)]
        try:
            with LLaDAAttentionBackend(model,'flash') as backend:
                result=generate_prefix_cache(model,torch.tensor([ids],device=model.device),gen_length=128)
        finally:
            for handle in handles:handle.remove()
        assert backend.report()['flash_calls']==result.nfe*32 and backend.report()['torch_sdpa_calls']==0
        report['prompts'].append(dict(id=sample.get('id',sample.get('task_id')),nfe=result.nfe,
            tokens=result.output[0,len(ids):].tolist(),backend=backend.report()))
        write(args.output/'diagnostic.json',report)
        print('Head shape observed '+str(sample.get('id')),flush=True)
    summary={}
    for name in report['records'][0]['methods']:
        rows=[r['methods'][name] for r in report['records']]
        summary[name]=dict(states=len(rows),logits_exact=sum(r['max_logit_error']==0 for r in rows),
            action_matches=sum(r['action_match'] for r in rows),max_logit_error=max(r['max_logit_error'] for r in rows))
    report['summary']=summary
    report['norm_exact_states']=sum(r['norm_max_error']==0 for r in report['records'])
    write(args.output/'diagnostic.json',report)
    (args.output/'complete').write_text('OK\n')
    print(str(summary),flush=True)


if __name__=='__main__':main()

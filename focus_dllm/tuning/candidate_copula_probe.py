"""Paid same-state candidate-copula falsification on six real development prompts.

Independent / shuffled / aligned sampling have identical original marginals,
temperature and pre-draw reveal positions. Conditional teacher calls diagnose
pair interactions only; none are used in generation. No new decoder or quality
claim. This first gate does not launch 2/4/8-step generation automatically.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from .candidate_copula import (factors, shuffled_factors, category_probabilities,
    draw_tokens, pair_table, interaction, choose_pair)


CONFIG = dict(prompts_per_task=2, sample_seed=51713, offset=0,
              length=256, block=32, teacher_threshold=.9, observed_blocks=[0, 2, 4, 6],
              k=4, temperature=1., gamma=.8, regularization=.001,
              monte_carlo_draws=32768, shuffled_controls=8, seed=1234,
              gate_minimum_nats=.02, bootstrap_samples=10000)


def fingerprint(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def conditional_canvas(canvas, own_position, other_position, other_token, mask_id=126336):
    if own_position == other_position or canvas.shape[0] != 1:
        raise ValueError('Two distinct positions in a batch-one state required')
    if canvas[0, own_position] != mask_id or canvas[0, other_position] != mask_id:
        raise ValueError('Both query positions must initially be MASK')
    value = canvas.clone()
    value[0, other_position] = other_token
    assert value[0, own_position] == mask_id
    return value


def aggregate(records):
    import numpy as np
    groups = defaultdict(list)
    for row in records:
        groups[row['task']+':'+str(row['id'])].append(row)
    if not groups:
        return dict(states=0, mechanism_gate_pass=False, scope='No eligible pairs')
    prompt_means = [np.mean([[r['aligned_interaction_nats']-r['independent_interaction_nats'],
                              r['aligned_interaction_nats']-r['shuffled_interaction_nats']]
                             for r in values], axis=0) for values in groups.values()]
    prompt_means = np.asarray(prompt_means)
    rng = np.random.default_rng(CONFIG['seed'])
    indices = rng.integers(0, len(groups), (CONFIG['bootstrap_samples'], len(groups)))
    low, high = np.percentile(prompt_means[indices].mean(1), [2.5, 97.5], axis=0)
    delta = prompt_means.mean(0)
    by_task = {task:sum(r['aligned_interaction_nats']-r['independent_interaction_nats'] for r in records if r['task']==task) /
                    sum(r['task']==task for r in records) for task in sorted({r['task'] for r in records})}
    # An initial research gate only. Require a practically visible, consistent
    # interaction improvement over BOTH independent and shuffled controls.
    gate = (len(groups)==6 and len(records)==24 and bool((low>0).all()) and
            bool((delta>=CONFIG['gate_minimum_nats']).all()) and min(by_task.values())>0)
    return dict(states=len(records), prompts=len(groups), mean_aligned_minus_independent_nats=float(delta[0]),
        aligned_minus_independent_95ci_nats=[float(low[0]),float(high[0])],
        mean_aligned_minus_shuffled_nats=float(delta[1]),
        aligned_minus_shuffled_95ci_nats=[float(low[1]),float(high[1])],
        task_aligned_minus_independent_nats=by_task, mechanism_gate_pass=gate,
        bootstrap_unit='Prompt, not sampled tokens or repeated draws',
        bootstrap_samples=CONFIG['bootstrap_samples'],
        largest_marginal_mc_error=max(r['largest_marginal_mc_error'] for r in records),
        mean_topk_pair_mass=sum(r['topk_pair_mass_independent'] for r in records)/len(records),
        scope='Conditional model-consistency proxy on reused development states; not correctness, '
              'an independent holdout, saved NFE, or evidence of online few-step speedup. '
              'TAIL interaction is neutral/unmeasured, explicitly included in joint statistics.')


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    binding = check_binding(required=True)
    from ..common import sha256, write_json
    from ..llada_common import MODEL_ID, REVISION, prompt_ids, MASK_ID
    from ..llada_evaluate import load_model
    from ..llada_decode import generate
    from ..llada_backend import LLaDAAttentionBackend
    from .competitors import select_samples, generation_prompt
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', type=Path, nargs=3, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False)
    data = dict(zip(('humaneval','mbpp','math'),args.datasets))
    implementation = {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    report = dict(model=MODEL_ID, revision=REVISION, binding=binding, configuration=CONFIG,
        implementation=implementation, datasets={k:sha256(p) for k,p in data.items()},
        prompts=[], records=[], skipped=[],
        scope='Frozen same-weight LLaDA full-sequence BF16 FlashAttention. '
              'Only legitimate paper_prompt enters teacher generation. Own-masked conditionals '
              'use proposed top-K tokens privately, never gold/tests/solutions or future teacher outputs. '
              'No cache/pruning/readout/Flash-Verify route is used or modified. '
              'Original per-position p = softmax(logits / temperature); floating-point CDF clipping '
              'is reported as an approximation, not universal lossless sampling.',
        probability_precision='FP64 reference copula construction and complete-vocabulary normalization',
        cost_scope='Logged generation is not a speed baseline. All conditional and sampling costs are paid diagnostics.')
    write_json(args.output/'diagnostic.json',report)
    model, tokenizer = load_model('cuda:0')
    torch.set_num_threads(1); torch.manual_seed(CONFIG['seed'])
    tr = model.model.transformer
    weight = tr.wte.weight if model.model.config.weight_tying else tr.ff_out.weight
    bias = None if model.model.config.weight_tying else tr.ff_out.bias
    assert bias is None or not torch.count_nonzero(bias), 'Nonzero output bias requires explicit feature design'
    forbidden = set(tokenizer.all_special_ids) | {MASK_ID,126081}
    scale = 1/math.sqrt(model.model.config.d_model) if model.model.config.scale_logits else 1.
    with LLaDAAttentionBackend(model,'flash') as backend:
        for task,path in data.items():
            for sample in select_samples(path,CONFIG['prompts_per_task'],CONFIG['offset']):
                ident = sample.get('id',sample.get('task_id'))
                ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                prompt = torch.tensor([ids],device=model.device)
                flags = dict(observe=False,busy=False,hidden=None,current=None,blocks=set(),shadow_calls=0)
                chains = []
                def pre(_model,inputs,kwargs):
                    if flags['busy']:
                        flags['shadow_calls'] += 1
                        assert not kwargs.get('use_cache',False) and kwargs.get('past_key_values') is None
                        return
                    canvas = inputs[0]
                    chains.append(fingerprint(canvas))
                    if not flags['observe']: return
                    active = (canvas[0,len(ids):] == MASK_ID).nonzero().flatten()
                    if not len(active): return
                    block = int(active[0])//CONFIG['block']
                    if block in CONFIG['observed_blocks'] and block not in flags['blocks']:
                        flags['blocks'].add(block)
                        flags['current'] = (canvas.clone(),block)
                def norm_hook(_norm,_inputs,value):
                    if flags['observe'] and not flags['busy'] and flags['current'] is not None:
                        flags['hidden'] = value.detach().clone()
                def post(_model,inputs,kwargs,output):
                    if flags['busy'] or flags['current'] is None: return
                    canvas,block = flags['current'];flags['current']=None
                    hidden = flags['hidden'];flags['hidden']=None
                    start = len(ids)+block*CONFIG['block'];end=start+CONFIG['block']
                    active = (canvas[0,start:end] == MASK_ID).nonzero().flatten()+start
                    all_logits = output.logits[0].index_select(0,active)
                    pair = choose_pair(all_logits,active,forbidden)
                    if pair is None:
                        report['skipped'].append(dict(task=task,id=ident,block=block,reason='Fewer than two non-special-top1 active positions'))
                        return
                    positions = active.index_select(0,pair)
                    logits = output.logits[0].index_select(0,positions)
                    full,top,mass_gpu = category_probabilities(logits,CONFIG['k'],CONFIG['temperature'])
                    h = hidden[0].index_select(0,positions)
                    w = weight.index_select(0,top.flatten()).reshape(2,CONFIG['k'],-1)
                    evidence_gpu = h[:,None,:].double()*w.double()*scale
                    reconstruction = (evidence_gpu.sum(-1)-logits.double().gather(1,top)).abs().max()
                    # Benchmarks of literal reference construction/sampling,
                    # including no Transformer. Never substitute MC table cost.
                    timings=[]
                    for repeat in range(4):
                        torch.cuda.synchronize();started=time.perf_counter()
                        draw_tokens(logits,evidence_gpu,1,torch.Generator(device='cuda').manual_seed(9000+repeat),
                                    gamma=CONFIG['gamma'],regularization=CONFIG['regularization'],temperature=CONFIG['temperature'])
                        torch.cuda.synchronize()
                        if repeat:timings.append(time.perf_counter()-started)
                    value = factors(evidence_gpu.cpu(),CONFIG['gamma'],CONFIG['regularization'])
                    mass = mass_gpu.cpu()
                    conditional = torch.zeros(2,CONFIG['k'],CONFIG['k'],dtype=torch.float64)
                    teacher_started = time.perf_counter(); flags['busy']=True
                    original = fingerprint(inputs[0])
                    try:
                        for own in range(2):
                            other = 1-own
                            for candidate in range(CONFIG['k']):
                                query = conditional_canvas(canvas,int(positions[own]),int(positions[other]),int(top[other,candidate]))
                                z = model(query,use_cache=False).logits[0,int(positions[own])].double()/CONFIG['temperature']
                                conditional[own,:,candidate] = z.log_softmax(-1).gather(0,top[own]).cpu()
                                assert query[0,int(positions[own])] == MASK_ID
                        assert original==fingerprint(inputs[0])==fingerprint(canvas),'Private conditional changed real canvas'
                    finally:flags['busy']=False
                    torch.cuda.synchronize();teacher_seconds=time.perf_counter()-teacher_started
                    log_base = full.gather(1,top).log().cpu()
                    raw = torch.zeros(CONFIG['k']+1,CONFIG['k']+1,dtype=torch.float64)
                    raw[:-1,:-1] = .5*(conditional[0]-log_base[0,:,None] + conditional[1].T-log_base[1,None,:])
                    centered = interaction(raw,mass)
                    independent_exact = mass[0,:,None]*mass[1,None,:]
                    state_seed = CONFIG['seed']+len(report['records'])*100
                    independent = pair_table(factors(evidence_gpu.cpu(),gamma=0.,regularization=CONFIG['regularization']),
                                             mass,CONFIG['monte_carlo_draws'],state_seed)
                    aligned = pair_table(value,mass,CONFIG['monte_carlo_draws'],state_seed)
                    shuffle_tables=[];permutations=[]
                    rng=torch.Generator().manual_seed(state_seed+37)
                    for control in range(CONFIG['shuffled_controls']):
                        perm=torch.stack([torch.randperm(CONFIG['k'],generator=rng) for _ in range(2)])
                        shuffled=shuffled_factors(value,perm)
                        shuffle_tables.append(pair_table(shuffled,mass,CONFIG['monte_carlo_draws'],state_seed+control+1))
                        permutations.append(perm.tolist())
                    shuffled=torch.stack(shuffle_tables).mean(0)
                    marginal_error=max(float((table.sum(1)-mass[0]).abs().max()) for table in [independent,aligned,*shuffle_tables])
                    marginal_error=max(marginal_error,max(float((table.sum(0)-mass[1]).abs().max()) for table in [independent,aligned,*shuffle_tables]))
                    record=dict(task=task,id=ident,block=block,positions=positions.tolist(),input_sha256=original,
                        canvas_token_ids=canvas[0].tolist(),candidate_ids=top.tolist(),
                        candidate_text=[[tokenizer.decode([int(t)]) for t in row] for row in top],
                        mass_with_tail=mass.tolist(),raw_conditional_interaction_nats=raw.tolist(),
                        centered_interaction_nats=centered.tolist(),conditional_log_probabilities=conditional.tolist(),
                        joint_tables=dict(independent=independent.tolist(),aligned=aligned.tolist(),shuffled=shuffled.tolist()),
                        shared_cross_covariance=(value.shared[0]@value.shared[1].T).tolist(),
                        shuffle_permutations=permutations,independent_interaction_nats=float((independent*centered).sum()),
                        analytic_independent_interaction_nats=float((independent_exact*centered).sum()),
                        aligned_interaction_nats=float((aligned*centered).sum()),
                        shuffled_interaction_nats=float((shuffled*centered).sum()),largest_marginal_mc_error=marginal_error,
                        topk_pair_mass_independent=float(independent_exact[:-1,:-1].sum()),
                        topk_pair_mass_aligned=float(aligned[:-1,:-1].sum()),
                        topk_pair_mass_shuffled=float(shuffled[:-1,:-1].sum()),
                        candidate_dot_vs_bf16_logit_max_error=float(reconstruction),
                        teacher_conditional_calls=2*CONFIG['k'],teacher_conditional_seconds=teacher_seconds,
                        literal_single_joint_draw_seconds=timings,
                        scope='Candidate labels enter private other-position context only; own queried slot stays MASK. '
                              'Full-vocabulary TAIL is sampled independently; TAIL interaction unmeasured/neutral. '
                              'MC covariance projection is diagnostic algebra, not online measured implementation.')
                    report['records'].append(record);report['summary']=aggregate(report['records'])
                    write_json(args.output/'diagnostic.json',report)
                    print('STATE',json.dumps({k:record[k] for k in ['task','id','block','aligned_interaction_nats','shuffled_interaction_nats','teacher_conditional_seconds']}),flush=True)
                handles=[model.register_forward_pre_hook(pre,with_kwargs=True),
                         model.register_forward_hook(post,with_kwargs=True),tr.ln_f.register_forward_hook(norm_hook)]
                try:
                    clean=generate(model,prompt,gen_length=CONFIG['length'],block_length=CONFIG['block'],threshold=CONFIG['teacher_threshold'])
                    clean_chain=list(chains);chains.clear();flags['observe']=True
                    observed=generate(model,prompt,gen_length=CONFIG['length'],block_length=CONFIG['block'],threshold=CONFIG['teacher_threshold'])
                    assert clean.nfe==observed.nfe and torch.equal(clean.output,observed.output)
                    assert clean_chain==chains,'Input state sequence changed under observation'
                    assert flags['current'] is None and flags['hidden'] is None and not flags['busy']
                    report['prompts'].append(dict(task=task,id=ident,prompt_tokens=len(ids),clean_nfe=clean.nfe,
                        observed_nfe=observed.nfe,private_conditional_calls=flags['shadow_calls'],
                        generated_token_ids=clean.output[0,len(ids):].tolist(),
                        teacher_canvas_chain_match=True,teacher_tokens_nfe_match=True,
                        logged_clean_seconds=clean.seconds,logged_observed_seconds=observed.seconds,
                        observed_blocks=sorted(flags['blocks'])))
                    write_json(args.output/'diagnostic.json',report)
                    print('PROMPT_COMPLETE',task,ident,len(report['records']),flush=True)
                finally:
                    for handle in handles:handle.remove()
        report['backend']=backend.report()
        assert report['backend']['torch_sdpa_calls']==0 and report['backend']['flash_calls']>0
    assert implementation=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},'Running source changed'
    report['summary']=aggregate(report['records'])
    report['outcome']='Mechanism gate passed; quality/speed validation still required' if report['summary']['mechanism_gate_pass'] else 'No initial mechanism gate; do not expand to few-step generation'
    write_json(args.output/'diagnostic.json',report)
    (args.output/'complete').write_text('OK\n')
    print('COMPLETE',json.dumps(report['summary']),flush=True)


if __name__=='__main__':main()

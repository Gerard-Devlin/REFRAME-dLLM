"""Six reused dev prompts: actual output scoring and complete latency."""
import argparse
import ast
import hashlib
import inspect
import json
from pathlib import Path
import statistics
import time
from types import SimpleNamespace

import torch

from focus_dllm.tuning.flash_readout import ModelReadout, generator, suppress_official_prints
from focus_v6.audit_probe import forbid_sdpa
from .generation import generate
from .observe import instrument


def fixed_horizon(function, final_canvas):
    """ONLY disable discovered-EOS shortening; preserve proposals/commits."""
    source = inspect.unwrap(function)
    tree = ast.parse(inspect.getsource(source))
    tree.body[0].decorator_list = []
    calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
             and isinstance(n.func,ast.Name) and n.func.id=='model']
    calls.sort(key=lambda n:n.lineno)
    if len(calls)!=2:
        raise ValueError('pinned model call sites changed')
    for call,expression in zip(calls,('(0, block_m)','(seqlen_keep[0] + num_verify, num_verify)')):
        call.keywords.append(ast.keyword(arg='focus_head_rows',value=ast.parse(expression,mode='eval').body))
    count = [0,0]
    class Change(ast.NodeTransformer):
        def visit_Assign(self, node):
            if ast.unparse(node) == 'predicted_length[j] = decoded_eos_pos[j] + 1':
                count[0] += 1
                node.value = ast.parse('prompt_lengths[i] + gen_length + j * max_length',mode='eval').body
            if ast.unparse(node).startswith('generated_answer_ids = x['):
                count[1]+=1
                callback=ast.parse('_fixed_final(x, prompt_lengths[i], max_length, gen_length)').body[0]
                return [ast.copy_location(callback,node),node]
            return self.generic_visit(node)
    tree = Change().visit(tree)
    if count != [2,1]:
        raise ValueError('unknown pinned EOS horizon sites')
    ast.fix_missing_locations(tree)
    scope = dict(source.__globals__,_fixed_final=final_canvas)
    exec(compile(tree, source.__code__.co_filename+':fixed_work', 'exec'),scope)
    return scope[source.__name__]


def score_outputs(task, sample, outputs):
    # Called only after both complete outputs have been persisted. No reference
    # answer/test/solution is ever passed into the generation API.
    if task == 'humaneval':
        from dllm_eval.score_humaneval import clean_completion, check
        return {name:dict(correct=check(clean_completion(sample['prompt'],r['text'],sample['entry_point']),
                        sample['test'],sample['entry_point'],6)) for name,r in outputs.items()}
    if task == 'mbpp':
        from dllm_eval.score_mbpp import clean_completion, check
        return {name:dict(correct=check(clean_completion(r['text']),sample['test_list'],6))
                for name,r in outputs.items()}
    from dllm_eval.score_answers import assess, MathComparison
    from dllm_eval.score_math import load_metric, installed_utils
    metric = load_metric(installed_utils())
    gold = metric['remove_boxed'](metric['last_boxed_only_string'](sample['solution']))
    comparison = MathComparison()
    return {name:assess(r['text'],gold,'math',comparison) for name,r in outputs.items()}


@torch.no_grad()
def main():
    from focus_dllm.common import sha256, write_json
    from focus_dllm.llada_common import MODEL_ID, REVISION, prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt, load_external, load_model, select_samples
    from focus_dllm.tuning.gpu_contract import check_binding
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--third-party',type=Path,required=True)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(); args.output.mkdir(exist_ok=False)
    cls,external,adaptation = load_external(args.third_party,'flash_verify')
    raw,tokenizer = load_model(SimpleNamespace(method='flash_verify'),cls)
    compact = ModelReadout(raw,compact=True,minimum=32)
    source = Path(inspect.getsourcefile(inspect.unwrap(external)))
    frozen = {p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    scorer = Path(__file__).parents[1]/'focus_dllm/dllm-eval/dllm_eval'
    scorer_hashes = {p.name:sha256(p) for p in scorer.glob('score*.py')}
    from importlib.metadata import version
    assert version('math-verify')=='0.1.0'
    report = dict(model=MODEL_ID,revision=REVISION,binding=check_binding(required=True),
        implementation=frozen,scorers=scorer_hashes,math_verify=version('math-verify'),
        official_sha256=sha256(source),adaptation=adaptation,
        scope='Six reused development prompts; separate task-scored diagnosis after failed token-agreement gate',
        configuration=dict(length=256,k=16,threshold=.9,admission='argmax-match+one correction',
            proposal_refill='dated normalized bootstrap hidden, paid projection',
            mandatory_repair='all changed identities in next clean query',seed=51713,offset=0),
        warmups=[],records=[])
    reference = {(r['task'],str(r['id'])):r for r in json.loads(args.reference.read_text())['prompts']}
    calls,query_rows = [0],[0]
    def count(_model,inputs):
        calls[0] += 1; query_rows[0] += int(inputs[0].shape[1])
    handle = raw.register_forward_pre_hook(count)

    def run_baseline(ids,fixed):
        final = [None]
        def capture(canvas,prompt_length,maximum,length):
            final[0] = canvas[prompt_length:prompt_length+length].detach().cpu().tolist()
        official = fixed_horizon(external,capture) if fixed else instrument(
            external,lambda *_:None,lambda *_:None,capture)
        response,steps = [None],[0]
        official(compact,[torch.tensor(ids,device=raw.device)],[len(ids)],1,response,steps,
                 gen_length=256,block_length=32,threshold=.9,gamma=.8,track_num=4,mask_num=4,
                 verify=True,tokenizer=tokenizer,stop_tokens=[])
        raw_ids = final[0]
        if raw_ids is None:
            raise AssertionError('baseline final canvas not captured')
        eos = next((i for i,v in enumerate(raw_ids) if v==126081),None)
        committed = sum(v!=126336 for v in raw_ids)
        return dict(text=response[0],raw_token_ids=raw_ids,nfe=2*steps[0],cycles=steps[0],
                    committed_count=committed,first_eos=eos,truncated=eos is None,
                    fixed_work=fixed,sdpa_calls=0)

    def run(ids,method,fixed=False):
        calls[0]=0; query_rows[0]=0
        torch.cuda.synchronize(); started=time.perf_counter()
        with forbid_sdpa(),suppress_official_prints():
            value = run_baseline(ids,fixed) if method=='flash_verify' else generate(
                compact,tokenizer,external,ids,length=256,fixed_work=fixed)
        torch.cuda.synchronize()
        value.update(seconds=time.perf_counter()-started,model_calls=calls[0],query_rows=query_rows[0])
        if value['nfe']!=calls[0]:
            raise AssertionError('NFE/model call mismatch')
        return value

    try:
        samples = [(task,s) for task,path in zip(('humaneval','mbpp','math'),args.datasets)
                   for s in select_samples(path,2,0)]
        warm_ids = prompt_ids(tokenizer,generation_prompt(samples[0][1]),'humaneval',preformatted=True)
        for method in ('flash_verify','focus_v7'):
            warm = run(warm_ids,method)
            report['warmups'].append(dict(method=method,seconds=warm['seconds'],nfe=warm['nfe']))
            print('WARMUP',method,warm['seconds'],flush=True)
        for index,(task,sample) in enumerate(samples):
            ident = str(sample.get('id',sample.get('task_id')))
            ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            row = dict(task=task,id=ident,outputs={})
            methods = ('flash_verify','focus_v7') if index%2==0 else ('focus_v7','flash_verify')
            for method in methods:
                value = run(ids,method)
                if method=='flash_verify' and hashlib.sha256(value['text'].encode()).hexdigest()!=reference[(task,ident)]['text_sha256']:
                    raise AssertionError('shared engineering changed original Flash output')
                row['outputs'][method] = value
                write_json(args.output/f'output_{index}_{method}.json',dict(task=task,id=ident,**value))
                print('OUTPUT',task,ident,method,'seconds',value['seconds'],'nfe',value['nfe'],flush=True)
            row['scores'] = score_outputs(task,sample,row['outputs'])
            report['records'].append(row)
            write_json(args.output/'summary.json',report)
            print('SCORE',task,ident,{n:s['correct'] for n,s in row['scores'].items()},flush=True)
        report['fixed_work_controls'] = []
        for task in ('humaneval','mbpp','math'):
            sample = next(s for t,s in samples if t==task)
            ids = prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
            row = dict(task=task,id=str(sample.get('id',sample.get('task_id'))),outputs={})
            for method in ('focus_v7','flash_verify'):
                row['outputs'][method] = run(ids,method,True)
                if row['outputs'][method]['committed_count']!=256:
                    raise AssertionError('fixed work did not fill all256positions')
                print('FIXED_WORK',task,method,row['outputs'][method]['seconds'],flush=True)
            row['latency_ratio'] = row['outputs']['flash_verify']['seconds']/row['outputs']['focus_v7']['seconds']
            report['fixed_work_controls'].append(row)
            write_json(args.output/'summary.json',report)
        report['task_summary'] = {}
        for task in ('humaneval','mbpp','math'):
            group = [r for r in report['records'] if r['task']==task]
            task_result = {}
            for method in ('flash_verify','focus_v7'):
                task_result[method] = dict(correct=sum(r['scores'][method]['correct'] for r in group),
                    total=len(group),mean_seconds=statistics.mean(r['outputs'][method]['seconds'] for r in group),
                    mean_nfe=statistics.mean(r['outputs'][method]['nfe'] for r in group),
                    mean_committed=statistics.mean(r['outputs'][method]['committed_count'] for r in group),
                    truncations=sum(r['outputs'][method]['truncated'] for r in group))
            task_result['latency_ratio'] = task_result['flash_verify']['mean_seconds']/task_result['focus_v7']['mean_seconds']
            report['task_summary'][task] = task_result
        assert frozen=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
        assert scorer_hashes=={p.name:sha256(p) for p in scorer.glob('score*.py')}
        assert report['official_sha256']==sha256(source)
        write_json(args.output/'summary.json',report)
        (args.output/'complete').write_text('OK\n')
        print(json.dumps(report['task_summary'],indent=2),flush=True)
    finally:
        handle.remove()


if __name__=='__main__':
    main()

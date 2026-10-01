"""Same 18 native state pairs: full FP64 spectra of hidden/attention/MLP deltas."""
import argparse
import gc
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from .delta_frontier import select_edges
from .delta_frontier_probe import check_action
from .delta_rank import (CaptureComponents, audit_legacy_row_energy, full_spectrum,
                        remove_energetic_rows, residual_audit)
from .temporal_batch import reconstruct_states

CONFIG = dict(length=256, block=32, threshold=.9, sample_seed=51713,
    prompts_per_task=2, pairs_per_prompt=3, tail_removed_fraction=.25,
    spectral_method='full smaller Gram matrix, FP64 eigvalsh; all singular energies saved',
    direct_svd_controls='First prompt, first edge, full hidden deltas at layers 1 and 32',
    rank_energy_percentages=[90,95,99], fixed_rank_budgets=[1,2,4,8,16,32,64])


@torch.no_grad()
def observed(model, canvas):
    with CaptureComponents(model) as observer:
        output=model(canvas)
    assert len(observer.rows)==32
    assert all(set(r)=={'input','hidden','attention','mlp'} for r in observer.rows)
    for previous,current in zip(observer.rows,observer.rows[1:]):
        assert torch.equal(previous['hidden'],current['input'])
    return output,observer.rows


@torch.no_grad()
def main():
    from .gpu_contract import check_binding
    from .competitors import select_samples, generation_prompt
    from ..common import sha256, write_json
    from ..llada_common import MODEL_ID, REVISION, prompt_ids
    from ..llada_evaluate import load_model
    from ..llada_backend import LLaDAAttentionBackend
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets',type=Path,nargs=3,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--frontier-reference',type=Path,required=True)
    parser.add_argument('--resume-report',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();args.output.mkdir(exist_ok=False)
    binding=check_binding(required=True)
    source={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    reference=json.loads(args.reference.read_text())
    previous=json.loads(args.frontier_reference.read_text())
    assert previous['complete_edges']==18
    refs={(p['task'],str(p['id'])):p for p in reference['prompts']}
    frontier={(p['task'],str(p['id'])):p for p in previous['prompts']}
    data=dict(zip(('humaneval','mbpp','math'),args.datasets))
    report=dict(configuration=CONFIG,model=MODEL_ID,revision=REVISION,binding=binding,
        implementation=source,datasets={k:sha256(p) for k,p in data.items()},
        reference_sha256=sha256(args.reference),frontier_reference_sha256=sha256(args.frontier_reference),
        component_definitions=dict(hidden='Native block output, after both BF16 residual additions',
            attention='Native attn_out output after multi-head attention and output projection, before residual addition',
            mlp='Native ff_out output after SwiGLU, before residual addition',
            tail='Hidden delta after removing ceil(25% N) rows with highest observed FP64 squared row norm; oracle selection'),
        scope='Same six reused development prompts and exactly the same 18 native within-block state edges. '
            'Only lawful prompt plus actual earlier commits construct inputs. No gold, decoder, cache approximation, '
            'new sampler, task score, trajectory-losslessness claim or online speedup.',
        numerical_scope='Exact BF16 activations are converted before subtraction. FP64 full Gram eigenvalues '
            'give squared singular values. Negative roundoff mass and trace conservation are checked. '
            'r90/r95/r99 are energy approximation ranks, not exact algebraic rank. Full direct FP64 SVD '
            'on two predeclared real matrices checks the Gram result. No random/truncated spectrum.',
        interpretation='Low rank of a score perturbation does not imply low rank after row softmax, '
            'multi-head mixing, RMSNorm, SwiGLU or BF16 residual rounding. Neither observed low rank nor '
            'paid SVD factors are an available cheap online updater. Tail energy share is reported.',
        cost_scope='Clean B1 forward, observed B1 forward with CPU copies, full spectral decomposition '
            'and direct SVD control cost are separate. This is a paid mechanism diagnostic.',
        prompts=[],direct_svd_controls=[])
    write_json(args.output/'diagnostic.json',report)
    resumed={}
    if args.resume_report:
        old_report=json.loads(args.resume_report.read_text())
        for key in ('configuration','datasets','reference_sha256','frontier_reference_sha256'):
            assert old_report[key]==report[key],f'Resume mismatch: {key}'
        resumed={(p['task'],str(p['id'])):p for p in old_report['prompts'] if p['edges']}
        report['resume_provenance']=dict(report_path=str(args.resume_report),
            report_sha256=sha256(args.resume_report),implementation=old_report['implementation'],
            complete_edges=sum(len(p['edges']) for p in resumed.values()),
            reason='Only legacy row-energy comparison repaired; completed full FP64 spectra remain unchanged.')
        report['direct_svd_controls']=old_report['direct_svd_controls']
        for p in resumed.values():
            for e in p['edges']:
                path=args.resume_report.parent/e['spectrum_file']
                assert sha256(path)==e['spectrum_sha256']
                shutil.copy2(path,args.output/path.name)
    model,tokenizer=load_model('cuda:0');model.eval()
    torch.set_num_threads(1);torch.manual_seed(1234)
    with LLaDAAttentionBackend(model,'flash') as backend:
        for task,path in data.items():
            for sample in select_samples(path,2,0,seed=CONFIG['sample_seed']):
                ident=str(sample.get('id',sample.get('task_id')))
                saved=refs[(task,ident)];prior=frontier[(task,ident)]
                ids=prompt_ids(tokenizer,generation_prompt(sample),task,preformatted=True)
                states,targets=reconstruct_states(ids,saved['trace'],saved['generated_token_ids'])
                edges=select_edges(saved['trace'])
                assert edges==[r['call_before'] for r in prior['edges']]
                entry=resumed.get((task,ident),dict(task=task,id=ident,prompt_tokens=len(ids),teacher_nfe=saved['clean_nfe'],edges=[]))
                report['prompts'].append(entry)
                reused={e['call_before'] for e in entry['edges']}
                assert reused<=set(edges)
                if reused==set(edges):
                    print('RETAINED COMPLETE SPECTRA',task,ident,flush=True)
                    continue
                model(torch.tensor([states[edges[0]]],device=model.device));torch.cuda.synchronize()
                for pair_number,i in enumerate(edges):
                    if i in reused:continue
                    start_pair=time.perf_counter()
                    changed=np.flatnonzero(np.asarray(states[i])!=np.asarray(states[i+1]))
                    old_record=prior['edges'][pair_number]
                    assert changed.tolist()==old_record['changed_positions']
                    clean=[];clean_times=[];captured=[];capture_times=[]
                    canvases=[torch.tensor([states[j]],device=model.device) for j in (i,i+1)]
                    for j,canvas in zip((i,i+1),canvases):
                        torch.cuda.synchronize();began=time.perf_counter();out=model(canvas)
                        torch.cuda.synchronize();clean_times.append(time.perf_counter()-began)
                        target=torch.tensor(targets[j],device=model.device)
                        logits=out.logits.index_select(1,target).detach().cpu()
                        got,expected=check_action(logits.to(model.device),canvas,target,saved['trace'][j])
                        assert got==[(p+len(ids),v) for p,v in expected]
                        clean.append(logits);del out
                    for index,(j,canvas) in enumerate(zip((i,i+1),canvases)):
                        began=time.perf_counter();out,rows=observed(model,canvas);torch.cuda.synchronize()
                        capture_times.append(time.perf_counter()-began)
                        target=torch.tensor(targets[j],device=model.device)
                        assert torch.equal(out.logits.index_select(1,target).cpu(),clean[index])
                        assert canvas.tolist()==[states[j]],'Observer changed canvas'
                        captured.append(rows);del out
                    old,new=captured
                    embedding=new[0]['input'][0].double()-old[0]['input'][0].double()
                    unchanged=np.ones(len(states[i]),bool);unchanged[changed]=False
                    assert torch.count_nonzero(embedding[unchanged])==0
                    embed_summary,_=full_spectrum(embedding,model.device)
                    assert embed_summary['r99']<=len(changed)
                    record=dict(call_before=i,call_after=i+1,block=saved['trace'][i]['block'],
                        changed_positions=changed.tolist(),input_changed_count=len(changed),
                        clean_call_seconds=clean_times,observed_call_seconds=capture_times,
                        native_release_actions_equal=True,observer_logits_bitwise_equal=True,
                        embedding=embed_summary,layers=[])
                    spectra={name:[] for name in ('hidden','attention','mlp','tail')}
                    for layer,(left,right) in enumerate(zip(old,new),start=1):
                        audit=residual_audit(left,right)
                        matrices={k:right[k][0].double()-left[k][0].double() for k in ('hidden','attention','mlp')}
                        legacy_audit=audit_legacy_row_energy(left['hidden'][0],right['hidden'][0],
                            old_record['layers'][layer-1]['delta_squared_norm'])
                        tail,selection=remove_energetic_rows(matrices['hidden'],CONFIG['tail_removed_fraction'])
                        matrices['tail']=tail
                        row=dict(layer=layer,residual_audit=audit,tail_selection=selection,legacy_row_energy_audit=legacy_audit)
                        for name,matrix in matrices.items():
                            summary,energy=full_spectrum(matrix,model.device)
                            row[name]=summary;spectra[name].append(energy)
                            if len(report['prompts'])==1 and pair_number==0 and layer in (1,32) and name=='hidden':
                                torch.cuda.synchronize();began=time.perf_counter()
                                sv=torch.linalg.svdvals(matrix.to(device=model.device,dtype=torch.float64),driver='gesvd').square().cpu().numpy()
                                seconds=time.perf_counter()-began
                                error=float(np.abs(sv-energy).sum()/max(float(energy.sum()),1e-300))
                                assert error<1e-8,'Gram spectrum differs from full direct SVD'
                                report['direct_svd_controls'].append(dict(task=task,id=ident,call=i,layer=layer,
                                    energy_l1_relative_error=error,seconds=seconds,shape=list(matrix.shape)))
                        row['sparse_plus_rank_energy']={str(r):selection['removed_energy_fraction']+
                            selection['tail_energy_fraction']*row['tail']['energy_at_rank'][str(r)]
                            for r in CONFIG['fixed_rank_budgets']}
                        record['layers'].append(row)
                        if layer%8==0:print('LAYER',task,ident,i,layer,'r99',row['hidden']['r99'],
                            'tail_r99',row['tail']['r99'],flush=True)
                    filename=f'spectrum_{len(report["prompts"])-1}_{pair_number}.npz'
                    np.savez_compressed(args.output/filename,**{k:np.stack(v) for k,v in spectra.items()})
                    record.update(spectrum_file=filename,spectrum_sha256=sha256(args.output/filename),
                                  diagnostic_pair_seconds=time.perf_counter()-start_pair)
                    entry['edges'].append(record);write_json(args.output/'diagnostic.json',report)
                    print('EDGE COMPLETE',task,ident,i,flush=True)
                    del captured,old,new,clean,canvases,matrices,embedding,spectra,rows,left,right,tail,matrix
                    gc.collect();torch.cuda.empty_cache()
        report['backend']=backend.report()
        report['backend_counter_scope']='Current process only. Retained edges keep their original native Flash replay/observer controls and source provenance.'
        assert report['backend']['torch_sdpa_calls']==0 and report['backend']['flash_calls']>0
    assert source=={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')}
    report['complete_edges']=sum(len(p['edges']) for p in report['prompts'])
    assert report['complete_edges']==18 and len(report['direct_svd_controls'])==2
    write_json(args.output/'diagnostic.json',report);(args.output/'complete').write_text('OK\n')
    print('COMPLETE',flush=True)


if __name__=='__main__':main()

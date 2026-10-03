"""Private real-model perturbation controls, separate from online work counts."""
import dataclasses
import torch
from .engine import probabilities


@torch.no_grad()
def control(runtime,ready,dag,tokenizer):
    old_nfe=runtime.nfe;old_reference=runtime.reference;old_attention=runtime.attention
    from focus_dllm.tuning.firebreak.attention import streaming
    runtime.attention=streaming
    # Hash-equivalent bit comparison of immutable public cache after controls.
    snapshot=[(k.clone(),v.clone()) for k,v in runtime.cache]
    label=next(t for t in tokenizer.encode('0 1',add_special_tokens=False)
               if t not in tokenizer.all_special_ids and t!=int(ready.ids[0]))
    changed=dataclasses.replace(ready,ids=ready.ids.clone());changed.ids[0]=label
    count=0
    try:
        runtime.reference=False
        base=runtime.run(ready);count+=1
        altered=runtime.run(changed);count+=1
        ancestors=dag.ancestors();unreachable=[i for i,a in enumerate(ancestors) if not a&1]
        error=float((base['logits'][unreachable].float()-altered['logits'][unreachable].float()).abs().max())
        clean_error=max(float((a.float()-b.float()).abs().max())
                        for ua,ub in zip(base['updates'],altered['updates']) for a,b in zip(ua,ub))
        assert error==0 and clean_error==0,'Forbidden draft label reached verification/shared context'
        runtime.reference=True
        dense=runtime.run(ready);count+=1
        candidate=ready.ids[:len(dag.parents)]
        def accept(result):
            probability=probabilities(result['logits']).gather(1,candidate[:,None]).flatten()
            return dag.closed_accept(((result['logits'].argmax(-1)==candidate)&(probability>=.8)).cpu().tolist())
        assert all(torch.equal(k,sk) and torch.equal(v,sv) for (k,v),(sk,sv) in zip(runtime.cache,snapshot))
        return dict(private_forwards=count,forbidden_rows=len(unreachable),forbidden_logit_error=error,
                    clean_kv_error=clean_error,public_cache_unchanged=True,
                    dense_max_logit_error=float((base['logits'].float()-dense['logits'].float()).abs().max()),
                    dense_acceptance_equal=accept(base)==accept(dense),
                    reference='Independent FP32 dense attention; not bitwise BF16 equivalence',
                    scope='Private diagnostic calls/time excluded from clean latency and online NFE')
    finally:
        runtime.nfe=old_nfe;runtime.reference=old_reference;runtime.attention=old_attention


@torch.no_grad()
def kernel_controls(device):
    from .engine import prepare
    from .graph import DAG
    from focus_dllm.tuning.firebreak.attention import dense_reference,streaming
    checks=[];torch.manual_seed(28481)
    for count,n in ((1,13),(5,129),(16,257)):
        dag=DAG(tuple(() if i%4==0 else (i-1,) for i in range(count)))
        canvas=[126336]*count+[11]*(n-count)
        ready=prepare(canvas,tuple(range(count))+tuple(range(count,min(count+8,n))),None,device,
                      candidate_positions=range(count),candidate_tokens=range(20,20+count),dag=dag)
        m,b=len(ready.ids),len(ready.private_rows)
        q=(torch.randn(m,2,128,device=device)*.2).bfloat16()
        k=torch.randn(n,2,128,device=device).bfloat16()*.2;v=torch.randn_like(k)
        pk=torch.randn(b,2,128,device=device).bfloat16()*.2;pv=torch.randn_like(pk)
        fast=streaming(q,k,v,pk,pv,ready.mapping,ready.choices)
        reference=dense_reference(q,k,v,pk,pv,ready.mapping,ready.choices)
        error=float((fast.float()-reference).abs().max());assert error<.025 and torch.isfinite(fast).all()
        checks.append(dict(candidates=count,cache=n,max_error=error))
    return checks

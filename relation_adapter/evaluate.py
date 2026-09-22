"""Sharded free generation and held-out reconstruction for frozen charts."""
import json
import math
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from relation_block.boundary import reconstruction
from relation_block.common import digest, snapshot, write_json
from relation_block.evaluate import answer, generate, prompt_ids
from relation_block.model import clean_mask
from relation_block.resident_eval import merge_records, merge_reconstruction
from .model import evaluation_copy


def base_parity(base, chart, tokenizer, block):
    """Zero-initialized token chart must reproduce frozen v2 logits exactly."""
    ids = prompt_ids(tokenizer, 'What is 1 plus 1?')
    x = torch.tensor([ids], device=next(base.parameters()).device)
    pos = torch.arange(len(ids), device=x.device)[None]
    mask = clean_mask(len(ids), block, x.device)
    with torch.no_grad():
        original = base(x, pos, mask)[0]
        candidate = chart(x, pos, mask)[0]
    if not torch.equal(original, candidate):
        raise AssertionError('Zero chart changed original logits')


@torch.no_grad()
def evaluate(base_bf16, chart, codec, data, config, output, rank, world,
             limit=256, rounds=(4, 8, 16), max_new_tokens=512, heldout_limit=32):
    from transformers import AutoTokenizer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((Path(data)/'gsm8k_dev_full.json').read_text())[:limit]
    heldout = json.loads((Path(data)/'heldout.json').read_text())[:heldout_limit]
    if len(rows) != limit or len(heldout) != heldout_limit:
        raise ValueError('Requested evaluation sample count unavailable')
    cpu_rng, gpu_rng, py_rng = torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
    started = time.perf_counter()
    model = None
    try:
        model = evaluation_copy(base_bf16, chart) if chart is not None else base_bf16
        tokenizer = AutoTokenizer.from_pretrained(snapshot(),local_files_only=True)
        results = {}
        for budget in rounds:
            generate(model, prompt_ids(tokenizer, 'What is 1 plus 1?'), codec,
                     config['mask_id'], config['eos_id'], budget, max_new_tokens)
            local = []
            for sample in rows[rank::world]:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                tick = time.perf_counter()
                generated, calls = generate(model, prompt_ids(tokenizer, sample['question']), codec,
                    config['mask_id'], config['eos_id'], budget, max_new_tokens)
                torch.cuda.synchronize()
                seconds = time.perf_counter()-tick
                prediction = tokenizer.decode(generated, skip_special_tokens=True)
                extracted = answer(prediction)
                target = answer(sample['answer'], gold=True)
                local.append(dict(id=sample['id'], prediction=prediction, correct=extracted is not None and extracted==target,
                                  extracted=extracted, target=target, tokens=len(generated),
                                  length_capped=len(generated)>=max_new_tokens, calls=calls,
                                  seconds=seconds,peak_gib=torch.cuda.max_memory_allocated()/2**30))
                print(f'eval rank={rank} rounds={budget} {len(local)}/{len(rows[rank::world])}',flush=True)
            shards = [None]*world
            dist.all_gather_object(shards,local)
            merged = merge_records(shards,[x['id'] for x in rows])
            times = sorted(x['seconds'] for x in merged)
            results[str(budget)] = dict(accuracy=sum(x['correct'] for x in merged)/limit,
                truncation_rate=sum(x['length_capped'] for x in merged)/limit,
                mean_generated_tokens=sum(x['tokens'] for x in merged)/limit,
                mean_calls=sum(sum(x['calls'].values()) for x in merged)/limit,
                mean_seconds=sum(times)/limit,
                p95_seconds=times[math.ceil(.95*limit)-1],
                tokens_per_second=sum(x['tokens'] for x in merged)/sum(times),
                peak_gib=max(x['peak_gib'] for x in merged))
            if rank==0:
                (output/f'samples_{budget}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in merged),encoding='utf-8')
        local_probes = [reconstruction(model,codec,[heldout[i]],config,'clean',seed=1234+i)
                        for i in range(rank,len(heldout),world)]
        shard = merge_reconstruction(local_probes) if local_probes else {}
        shards = [None]*world
        dist.all_gather_object(shards,shard)
        diagnostic = merge_reconstruction(shards)
        report = dict(results=results,reconstruction=diagnostic,examples=limit,
                      heldout_examples=heldout_limit,ids=[x['id'] for x in rows],
                      evaluation_seconds=time.perf_counter()-started,
                      dev_hash=digest(Path(data)/'gsm8k_dev_full.json'),
                      heldout_hash=digest(Path(data)/'heldout.json'),
                      note='Accuracy is sharded across GPUs. Per-request latency includes adapter and inverse codec, with concurrent requests on different GPUs; evaluation wall time is separate. Reconstruction CE differs across transformed target alphabets.')
        if rank==0:
            write_json(output/'summary.json',report)
        return report
    finally:
        del model
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(gpu_rng)
        random.setstate(py_rng)

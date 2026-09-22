"""Evaluate disjoint shards while all training ranks retain their state."""
import json
from pathlib import Path
import random
import time
import torch
import torch.distributed as dist
from .common import digest, manifest, snapshot, write_json
from .codec import Codec
from .model import Model
from .evaluate import answer, generate, prompt_ids
from .diagnose import reconstruction


def bf16_copy(model):
    # Same tensor rounding as checkpoint export; do not mutate FP32 masters or DDP.
    with torch.device('meta'):
        result = Model(model.config)
    from .adaptation import merged_state
    result.load_state_dict(merged_state(model), assign=True)
    if model.config.get('tie_word_embeddings', False):
        result.lm_head.weight = result.model.embed_tokens.weight
    return result.requires_grad_(False).eval()


def merge_records(shards, ids):
    records = [r for shard in shards for r in shard]
    indexed = {r['id']: r for r in records}
    if len(indexed) != len(records) or set(indexed) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError('Evaluation shards contain missing/duplicate/unexpected IDs')
    return [indexed[i] for i in ids]


def merge_reconstruction(shards):
    result = {}
    for probability in ('.25', '.5', '.75'):
        key = str(float(probability))
        parts = [s[key] for s in shards if s]
        total = sum(s['categories']['all']['total'] for s in parts)
        if not total:
            raise ValueError('Empty reconstruction shards')
        categories = {}
        names = ['all', 'boundary', 'ordinary', 'eos']
        if all('block_first' in s['categories'] for s in parts):
            names.append('block_first')
        for name in names:
            count = sum(s['categories'][name]['total'] for s in parts)
            correct = sum(s['categories'][name]['correct'] for s in parts)
            categories[name] = dict(correct=correct, total=count, accuracy=correct/count if count else None)
            if all('ce_sum' in s['categories'][name] for s in parts):
                ce_sum = sum(s['categories'][name]['ce_sum'] for s in parts)
                categories[name].update(ce_sum=ce_sum, cross_entropy=ce_sum/count if count else None,
                                        loss_contribution=ce_sum/total)
        result[key] = dict(cross_entropy=sum(s['cross_entropy'] * s['categories']['all']['total'] for s in parts)/total,
                           categories=categories)
    return result


@torch.no_grad()
def evaluate_resident(master, data, output, arm, rank, world, limit, rounds, cap, reconstruction_limit,
                      boundary_probes=False):
    from transformers import AutoTokenizer
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    cfg = manifest(data)
    rows = json.loads((data / 'gsm8k_dev_full.json').read_text())
    if not 0 < limit <= len(rows):
        raise ValueError('Invalid evaluation limit')
    rows = rows[:limit]
    cpu_rng, cuda_rng, py_rng = torch.get_rng_state(), torch.cuda.get_rng_state(), random.getstate()
    tick = time.perf_counter()
    model = None
    try:
        model = bf16_copy(master)
        tok = AutoTokenizer.from_pretrained(snapshot(), local_files_only=True)
        codec = Codec(json.loads((data / 'codec.json').read_text()), identity=arm == 'token').to(next(master.parameters()).device)
        results = {}
        for budget in map(int, rounds.split(',')):
            generate(model, prompt_ids(tok, 'What is 1 plus 1?'), codec, cfg['mask_id'], cfg['eos_id'], budget, cap)
            records = []
            for sample in rows[rank::world]:
                generated, calls = generate(model, prompt_ids(tok, sample['question']), codec,
                                            cfg['mask_id'], cfg['eos_id'], budget, cap)
                text = tok.decode(generated, skip_special_tokens=True)
                pred, gold = answer(text), answer(sample['answer'], gold=True)
                records.append(dict(id=sample['id'], prediction=text, extracted=pred, target=gold,
                                    correct=pred is not None and pred == gold, tokens=len(generated),
                                    calls=calls, length_capped=len(generated) >= cap, rank=rank))
                print(f'eval rank={rank} rounds={budget} {len(records)}/{len(rows[rank::world])}', flush=True)
            shards = [None] * world
            if world > 1:
                dist.all_gather_object(shards, records)
            else:
                shards[0] = records
            merged = merge_records(shards, [r['id'] for r in rows])
            results[str(budget)] = dict(accuracy=sum(r['correct'] for r in merged)/limit,
                truncation_rate=sum(r['length_capped'] for r in merged)/limit,
                mean_generated_tokens=sum(r['tokens'] for r in merged)/limit,
                mean_calls=sum(sum(r['calls'].values()) for r in merged)/limit)
            if rank == 0:
                (output / f'samples_{budget}.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in merged), encoding='utf-8')
        heldout = json.loads((data / 'heldout.json').read_text())[:reconstruction_limit]
        # Keep original index-based seeds, independent of world size/sharding.
        probes = {}
        for mode in (('clean', 'masked') if boundary_probes else ('legacy',)):
            if mode == 'legacy':
                local_probes = [reconstruction(model, codec, [heldout[i]], cfg, seed=1234+i)
                                for i in range(rank, len(heldout), world)]
            else:
                from .boundary import reconstruction as boundary_reconstruction
                local_probes = [boundary_reconstruction(model, codec, [heldout[i]], cfg, mode, seed=1234+i)
                                for i in range(rank, len(heldout), world)]
            local = merge_reconstruction(local_probes) if local_probes else {}
            gathered = [None] * world
            if world > 1:
                dist.all_gather_object(gathered, local)
            else:
                gathered[0] = local
            probes[mode] = merge_reconstruction(gathered)
        if rank == 0:
            write_json(output / 'summary.json', dict(results=results, reconstruction=probes.get('legacy'),
                reconstruction_by_objective=probes if boundary_probes else None,
                reconstruction_partition='boundary non-EOS + interior non-EOS + EOS' if boundary_probes else 'legacy',
                examples=limit, ids=[r['id'] for r in rows], world_size=world,
                evaluation_seconds=time.perf_counter()-tick, evaluator_sha256=digest(Path(__file__)),
                scope='Resident distributed accuracy evaluation. Wall time is not single-GPU inference latency.'))
    finally:
        del model
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state(cuda_rng)
        random.setstate(py_rng)

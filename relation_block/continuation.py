"""Matched short token continuations; reuse the unchanged training implementation."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .common import digest, manifest, write_json
from .full_state import epoch_batches, resolve_checkpoint
from .preflight import implementation_hashes, require_gate


def milestones(rows, batches, length, budgets=(500000, 1000000, 2000000)):
    """Stop at the first complete optimizer batch reaching each token budget."""
    result, seen = [], 0
    for step, indices in enumerate(batches, 1):
        seen += sum(min(len(rows[i]['ids']), length) for i in indices)
        while len(result) < len(budgets) and seen >= budgets[len(result)]:
            result.append(dict(requested_tokens=budgets[len(result)], original_tokens=seen, step=step))
        if len(result) == len(budgets):
            if len({r['step'] for r in result}) != len(result):
                raise ValueError('Milestones must fall in different optimizer batches')
            return result
    raise ValueError('Dataset is smaller than the requested token budget')


def training_command(args, destination, stop, resume=None):
    cmd = [sys.executable, '-u', '-m', 'torch.distributed.run', '--standalone', '--nnodes=1',
           f'--nproc_per_node={args.world_size}', '-m', 'relation_block.train',
           '--data', str(args.data), '--output', str(destination), '--arm', 'token',
           '--steps', '0', '--stop-after', str(stop), '--lr', str(args.lr),
           '--global-batch', str(args.global_batch), '--micro-batch', '1',
           '--seed', str(args.seed), '--max-seconds', '0', '--eval-every', '0',
           '--save-every', '1000000000']
    if resume is not None:
        cmd += ['--resume', str(resume)]
    return cmd


def probe(args):
    import torch
    from .codec import Codec
    from .common import snapshot
    from .diagnose import reconstruction
    from .full_state import load_full_model
    from .model import Model
    model = load_full_model(args.checkpoint)[0] if args.checkpoint else Model.load(snapshot()).eval()
    codec = Codec(json.loads((args.data / 'codec.json').read_text()), identity=True).cuda()
    rows = json.loads((args.data / 'heldout.json').read_text())[:args.reconstruction_limit]
    if not rows:
        raise ValueError('No held-out reconstruction samples')
    with torch.no_grad():
        result = reconstruction(model, codec, rows, manifest(args.data))
    write_json(args.output, dict(results=result, examples=len(rows),
        heldout_hash=digest(args.data / 'heldout.json'), seed=1234))


def campaign(args):
    require_gate(args.data)
    cfg = manifest(args.data)
    if args.output.exists():
        raise ValueError('Use a new campaign output directory')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if len(visible) != args.world_size or not all(visible) or args.global_batch % args.world_size:
        raise ValueError('Visible GPUs/world size/global batch mismatch')
    gate = json.loads((args.data / 'full_smoke.json').read_text())
    expected = dict(implementation=implementation_hashes(), data_hash=digest(args.data / 'manifest.json'),
                    world_size=args.world_size, global_batch=args.global_batch, micro_batch=1)
    if not gate.get('pass') or any(gate.get(k) != v for k, v in expected.items()):
        raise ValueError('Full optimizer smoke gate must match this GPU count/batch/data/implementation')
    rows = json.loads((args.data / 'train.json').read_text())
    batches = epoch_batches(len(rows), args.global_batch, args.seed)
    points = milestones(rows, batches, cfg['length'])
    args.output.mkdir(parents=True)
    plan = dict(milestones=points, epoch_steps=len(batches), seed=args.seed,
                learning_rates=[2e-5, 4e-6], schedule='Original full-epoch 3% warmup/cosine; pause at token milestones',
                global_batch=args.global_batch, world_size=args.world_size,
                implementation=implementation_hashes(), driver_sha256=digest(Path(__file__)),
                data_hash=digest(args.data / 'manifest.json'),
                dev_hash=digest(args.data / 'gsm8k_dev_full.json'),
                note='Token only. No trajectory supervision or causal/AR equivalence claim. Token budget includes prompt.')
    write_json(args.output / 'plan.json', plan)
    print(json.dumps(plan), flush=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=visible[0])
    for key in ('RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT'):
        env.pop(key, None)
    results = {}

    def evaluate(label, checkpoint=None):
        dest = args.output / label
        cmd = [sys.executable, '-u', '-m', 'relation_block.evaluate', '--data', str(args.data),
               '--output', str(dest / 'eval'), '--limit', str(args.limit), '--rounds', args.rounds,
               '--max-new-tokens', '512']
        if checkpoint:
            cmd += ['--checkpoint', str(checkpoint)]
        subprocess.run(cmd, env=env, check=True)
        cmd = [sys.executable, '-u', '-m', 'relation_block.continuation', 'probe', '--data', str(args.data),
               '--output', str(dest / 'reconstruction.json'), '--reconstruction-limit', str(args.reconstruction_limit)]
        if checkpoint:
            cmd += ['--checkpoint', str(checkpoint)]
        subprocess.run(cmd, env=env, check=True)
        results[label] = dict(evaluation=json.loads((dest / 'eval/summary.json').read_text()),
                             reconstruction=json.loads((dest / 'reconstruction.json').read_text()))
        if checkpoint:
            results[label]['training'] = json.loads((checkpoint / 'metadata.json').read_text())
        write_json(args.output / 'summary.json', dict(complete=False, results=results))

    evaluate('original')
    for name, lr in (('lr_2e-5', 2e-5), ('lr_4e-6', 4e-6)):
        args.lr = lr
        destination = args.output / name / 'train'
        previous = None
        for point in points:
            subprocess.run(training_command(args, destination, point['step'], previous), check=True)
            previous = resolve_checkpoint(destination)
            meta = json.loads((previous / 'metadata.json').read_text())
            if meta['original_tokens'] != point['original_tokens'] or meta['completed_steps'] != point['step']:
                raise AssertionError('Training cursor differs from planned token budget')
            evaluate(f"{name}/tokens_{point['requested_tokens']}", previous)
    write_json(args.output / 'summary.json', dict(complete=True, results=results))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('mode', choices=['campaign', 'probe'])
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--world-size', type=int, default=6)
    p.add_argument('--global-batch', type=int, default=12)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--limit', type=int, default=256)
    p.add_argument('--rounds', default='8,16')
    p.add_argument('--reconstruction-limit', type=int, default=32)
    p.add_argument('--checkpoint', type=Path)
    args = p.parse_args()
    if min(args.world_size, args.global_batch, args.limit, args.reconstruction_limit) < 1:
        p.error('Counts must be positive')
    (probe if args.mode == 'probe' else campaign)(args)


if __name__ == '__main__':
    main()

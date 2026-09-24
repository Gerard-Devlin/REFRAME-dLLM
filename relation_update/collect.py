"""Collect prompt-disjoint pairs from the unchanged official v2 decoder."""

import argparse
import json
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist

from .data import load_prompts, save_torch, sha256, write_json
from .teacher import NativeObserver


MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
REVISION = "da5608172d2b74380e4e780baa19c71645e4f981"
CODE_HASH = "d363ee4a4d4bf52958645d5c715712c5b027525bb90611a52178e58695e09b50"


def distributed():
    rank, world, local = (int(os.getenv(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    return rank, world, device


def barrier():
    if dist.is_initialized():
        dist.barrier()


def parity(model, ids, options, observer_options):
    inputs = []
    original = model.forward

    def capture(*args, **kwargs):
        value = kwargs.get("input_ids", args[0] if args else None)
        inputs.append((value.detach().cpu().tolist(), kwargs.get("update_past_key_values"),
                       kwargs.get("use_block_cache"), kwargs.get("replace_position")))
        return original(*args, **kwargs)

    short = {**options, "max_new_tokens": 64}
    torch.cuda.synchronize()
    start = time.perf_counter()
    model.forward = capture
    try:
        expected = model.generate(ids.clone(), **short)
    finally:
        model.forward = original
    torch.cuda.synchronize()
    plain_seconds = time.perf_counter() - start
    start = time.perf_counter()
    with NativeObserver(model, **observer_options) as observer:
        observer.capture_trace = True
        actual = model.generate(ids.clone(), **short)
    torch.cuda.synchronize()
    observed_seconds = time.perf_counter() - start
    if not torch.equal(expected, actual) or inputs != observer.trace:
        raise AssertionError("Observing the teacher changed native tokens/forward inputs; stop")
    return dict(pass_=True, forward_calls=len(inputs), same_tokens=True, same_forward_inputs=True,
                plain_seconds=plain_seconds, instrumented_seconds=observed_seconds,
                note="Short instrumentation diagnostic, not a speedup benchmark")


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--train-prompts", type=int, default=256)
    p.add_argument("--heldout-prompts", type=int, default=64)
    p.add_argument("--max-pairs-per-prompt", type=int, default=32)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--feature-size", type=int, default=64)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if min(args.train_prompts, args.heldout_prompts, args.max_pairs_per_prompt, args.feature_size) < 1:
        p.error("Positive collection sizes required")
    if args.top_k < 2 or args.max_new_tokens < 64 or args.max_new_tokens % 32 or not 0 < args.threshold <= 1:
        p.error("top-k >=2, generation a multiple of 32 >=64, and threshold in (0,1] required")
    source_manifest = json.loads((args.data / "manifest.json").read_text())
    if source_manifest["revision"] != REVISION:
        raise ValueError("Prepared prompt tokenizer revision differs from pinned teacher")
    prompts = load_prompts(args.data, args.train_prompts, args.heldout_prompts, args.seed)
    jobs = [(split, row) for split in ("train", "heldout") for row in prompts[split]]
    rank, world, device = distributed()
    if world > len(jobs):
        raise ValueError("Use at most one GPU per prompt")
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Trace output must be new/empty; partial collections are never accepted for training")
        args.output.mkdir(parents=True, exist_ok=True)
    barrier()
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    if sha256(snapshot / "modeling.py") != CODE_HASH:
        raise ValueError("Pinned official model source mismatch")
    model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True,
                local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    if any(parameter.requires_grad for parameter in model.parameters()):
        raise AssertionError("Teacher must be frozen")
    embeddings = model.get_input_embeddings().weight.detach()
    if args.top_k >= len(embeddings):
        raise ValueError("top-k must be smaller than vocabulary")
    generator = torch.Generator().manual_seed(args.seed)
    projection = torch.randn(embeddings.shape[1], args.feature_size, generator=generator) / args.feature_size ** 0.5
    if rank == 0:
        table = (embeddings @ projection.to(device, embeddings.dtype)).cpu()
        save_torch(args.output / "features.pt", dict(table=table, projection=projection))
        del table
    options = dict(max_new_tokens=args.max_new_tokens, block_size=32, small_block_size=8,
                   threshold=args.threshold, temperature=0, use_block_cache=False)
    observer_options = dict(top_k=args.top_k, max_pairs=args.max_pairs_per_prompt,
                            threshold=args.threshold, seed=args.seed)
    check = parity(model, torch.tensor([jobs[rank][1]["ids"]], device=device), options, observer_options)
    write_json(args.output / f"parity_rank_{rank}.json", check)
    print(f"rank={rank} native observation parity passed: {check['forward_calls']} calls", flush=True)
    shard_info = []
    started = time.perf_counter()
    for index in range(rank, len(jobs), world):
        split, row = jobs[index]
        ids = torch.tensor([row["ids"]], device=device)
        observer_options["seed"] = args.seed + index
        with NativeObserver(model, **observer_options) as observer:
            output = model.generate(ids, **options)
        filename = f"{split}/prompt_{row['id']}.pt"
        save_torch(args.output / filename, dict(prompt_id=row["id"], records=observer.records))
        info = dict(file=filename, split=split, prompt_id=row["id"], pairs=len(observer.records),
                    seen_pairs=observer.seen, forward_calls=observer.calls,
                    generated_tokens=output.shape[-1] - ids.shape[-1],
                    sha256=sha256(args.output / filename))
        shard_info.append(info)
        print(f"rank={rank} prompt={index+1}/{len(jobs)} split={split} "
              f"pairs={len(observer.records)}/{observer.seen} calls={observer.calls}", flush=True)
    write_json(args.output / f"rank_{rank}.json", shard_info)
    barrier()
    if rank == 0:
        shards = [item for worker in range(world)
                  for item in json.loads((args.output / f"rank_{worker}.json").read_text())]
        if len(shards) != len(jobs) or len({item["prompt_id"] for item in shards}) != len(jobs):
            raise AssertionError("Missing/duplicate prompts across collection ranks")
        report = dict(format_version=1, status="complete", model=MODEL_ID, revision=REVISION,
                      hidden_size=embeddings.shape[1], feature_size=args.feature_size,
                      top_k=args.top_k, block_size=32, small_block_size=8,
                      threshold=args.threshold, max_new_tokens=args.max_new_tokens, seed=args.seed,
                      feature_sha256=sha256(args.output / "features.pt"), shards=shards,
                      source=str(args.data), source_sha256={name: sha256(args.data / name)
                          for name in ("train.json", "heldout.json", "manifest.json")},
                      implementation_sha256={name: sha256(Path(__file__).parent / name)
                          for name in ("collect.py", "teacher.py", "data.py")},
                      world_size=world, collection_seconds=time.perf_counter() - started,
                      note="Natural native transitions only. Original answers ignored. No extra counterfactual forward. "
                           "Offline top-K+OTHER diagnosis, not full-vocabulary KL or end-to-end acceleration.")
        write_json(args.output / "manifest.json", report)
        print(json.dumps({split: sum(s['pairs'] for s in shards if s['split'] == split)
                          for split in ("train", "heldout")}), flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

"""Bounded single-GPU / torchrun DDP training with fixed global batch size."""
import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .neural import Denoiser, ModelConfig, masked_loss, update_batch
from .prepare import digest
from .torch_codec import TorchCodec


def setup(device):
    world = int(os.environ.get("WORLD_SIZE", "1"))
    rank, local = int(os.environ.get("RANK", "0")), int(os.environ.get("LOCAL_RANK", "0"))
    if device == "cuda":
        torch.cuda.set_device(local)
        dev = torch.device("cuda", local)
    else:
        dev = torch.device("cpu")
        torch.set_num_threads(1)
    if world > 1:
        dist.init_process_group("nccl" if device == "cuda" else "gloo")
    return world, rank, dev


def amp(device, dtype):
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16) if dtype == "bfloat16" else nullcontext()


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(path, model, optimizer, step, meta, elapsed):
    temp = path.with_suffix(".tmp")
    torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                    metadata=meta, train_seconds=elapsed), temp)
    os.replace(temp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--codec", choices=["identity", "rename", "random", "relation1", "relation2"], default="identity")
    p.add_argument("--objective", choices=["diffusion", "one-step"], default="diffusion")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--max-seconds", type=float, default=600, help="Per invocation budget, checked after each optimizer update")
    p.add_argument("--global-batch", type=int, default=48)
    p.add_argument("--micro-batch", type=int, default=8)
    p.add_argument("--width", type=int, default=384)
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.steps, args.global_batch, args.micro_batch, args.log_every, args.save_every) < 1 or args.max_seconds <= 0:
        p.error("Budgets and intervals must be positive")
    if args.lr <= 0:
        p.error("Learning rate must be positive")
    world, rank, device = setup(args.device)
    try:
        if args.global_batch % (world * args.micro_batch):
            raise ValueError("global-batch must be divisible by world_size * micro-batch; do not silently change training budget")
        accumulation = args.global_batch // (world * args.micro_batch)
        manifest = json.loads((args.data / "manifest.json").read_text(encoding="utf-8"))
        spec = json.loads((args.data / f"{args.codec}.json").read_text(encoding="utf-8"))
        if digest(args.data / f"{args.codec}.json") != manifest["codecs"][args.codec]:
            raise ValueError("Codec hash mismatch")
        if digest(args.data / "train.npy") != manifest["train_sha256"]:
            raise ValueError("Training data hash mismatch")
        data = np.load(args.data / "train.npy", mmap_mode="r")
        cfg = ModelConfig(length=manifest["length"], width=args.width, layers=args.layers, heads=args.heads)
        torch.manual_seed(args.seed)
        model = Denoiser(cfg).to(device)
        codec = TorchCodec(spec).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        meta = dict(model=asdict(cfg), codec=args.codec, codec_spec=spec, objective=args.objective,
                    data_manifest=manifest, seed=args.seed, global_batch=args.global_batch,
                    planned_steps=args.steps, lr=args.lr, dtype=args.dtype,
                    parameter_count=sum(x.numel() for x in model.parameters()),
                    training_note="from scratch; dropout=0; deterministic global batches; no KV cache",
                    torch_version=torch.__version__)
        start_step, previous_seconds = 0, 0.0
        checkpoint = args.output / "checkpoint.pt"
        if args.resume:
            # Only load checkpoints produced by this trusted training program.
            saved = torch.load(checkpoint, map_location=device, weights_only=False)
            for key in ("model", "codec_spec", "objective", "data_manifest", "seed", "global_batch", "planned_steps", "lr", "dtype"):
                if saved["metadata"][key] != meta[key]:
                    raise ValueError(f"Resume configuration changed: {key}")
            model.load_state_dict(saved["model"])
            optimizer.load_state_dict(saved["optimizer"])
            start_step, previous_seconds = saved["step"], saved["train_seconds"]
        elif rank == 0 and args.output.exists():
            raise ValueError("Output exists; use --resume with identical training settings or choose a new directory")
        if rank == 0:
            args.output.mkdir(parents=True, exist_ok=args.resume)
            (args.output / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            print(json.dumps(dict(parameters=meta["parameter_count"], world_size=world,
                                  global_batch=args.global_batch, micro_batch=args.micro_batch,
                                  accumulation=accumulation, planned_steps=args.steps), indent=2), flush=True)
        if world > 1:
            dist.barrier()
            wrapped = DDP(model, device_ids=[device.index] if device.type == "cuda" else None,
                          broadcast_buffers=False)
        else:
            wrapped = model
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
        started = time.perf_counter()
        completed, timed_out = start_step, False
        warm_times = []
        for step in range(start_step, args.steps):
            tick = time.perf_counter()
            ids, masks, probability = update_batch(len(data), args.global_batch, cfg.length,
                                                    manifest["prefix"], args.seed, step, args.objective)
            warmup = min(100, max(1, args.steps // 10))
            rate = args.lr * min(1, (step + 1) / warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * step / args.steps)))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            total_loss = torch.zeros((), device=device)
            # Contiguous rank shards of the same global examples and masks.
            rank_start = rank * (args.global_batch // world)
            for micro in range(accumulation):
                lo = rank_start + micro * args.micro_batch
                hi = lo + args.micro_batch
                clean = torch.from_numpy(np.asarray(data[ids[lo:hi].numpy()], dtype=np.int64)).to(device)
                mask, prob = masks[lo:hi].to(device), probability[lo:hi].to(device)
                codes = codec.encode(clean)
                noisy = codes.masked_fill(mask, cfg.vocab_size)
                sync = wrapped.no_sync() if world > 1 and micro < accumulation - 1 else nullcontext()
                with sync:
                    with amp(device, args.dtype):
                        loss = masked_loss(wrapped(noisy), codes, mask, prob, manifest["prefix"])
                        scaled = loss / accumulation
                    scaled.backward()
                total_loss += loss.detach() / accumulation
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(total_loss)
                total_loss /= world
            synchronize(device)
            completed = step + 1
            warm_times.append(time.perf_counter() - tick)
            elapsed = time.perf_counter() - started
            stop = torch.tensor(int(elapsed >= args.max_seconds), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            timed_out = bool(stop.item())
            if rank == 0 and (completed % args.log_every == 0 or completed == args.steps or timed_out):
                steady = warm_times[min(5, len(warm_times) - 1):]
                update_seconds = sum(steady) / len(steady)
                row = dict(step=completed, loss=float(total_loss), lr=rate, world_size=world,
                           elapsed_seconds=elapsed, cumulative_train_seconds=previous_seconds + elapsed,
                           seen_symbols=completed * args.global_batch * cfg.length,
                           update_seconds=update_seconds,
                           symbols_per_second=args.global_batch * cfg.length / update_seconds,
                           eta_seconds=(args.steps - completed) * update_seconds,
                           max_memory_allocated_gib=torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None)
                with (args.output / "train.jsonl").open("a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
            if rank == 0 and (completed % args.save_every == 0 or completed == args.steps or timed_out):
                save_checkpoint(checkpoint, model, optimizer, completed, meta,
                                previous_seconds + time.perf_counter() - started)
            if timed_out:
                break
        if rank == 0:
            status = dict(status="complete" if completed == args.steps else "budget_exhausted",
                          completed_steps=completed, planned_steps=args.steps,
                          world_size=world, max_seconds=args.max_seconds,
                          seconds=previous_seconds + time.perf_counter() - started,
                          note="A time-limited checkpoint must be matched by actual completed updates when comparing codecs")
            (args.output / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
            print(json.dumps(status), flush=True)
        if world > 1:
            dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()

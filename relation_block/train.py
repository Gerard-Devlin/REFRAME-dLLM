"""Matched LoRA adaptation; single GPU or deterministic global-batch DDP.

No implicit training launch and no automatic budget extension. FP32 adapter
optimizer states; BF16 frozen backbone. Last checkpoint is resumable only with
the same data, world-independent global batch and planned schedule.
"""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.checkpoint import checkpoint
from .codec import Codec
from .common import digest, manifest, snapshot, write_json
from .model import Model, add_lora, adapter_state, training_mask


def batch(rows, indices, length, pad, device):
    ids = torch.full((len(indices), length), pad, dtype=torch.long)
    response = torch.zeros_like(ids, dtype=torch.bool)
    prefix = []
    for j, i in enumerate(indices):
        row = rows[i]
        n, p = len(row["ids"]), row["prefix"]
        ids[j, :n] = torch.tensor(row["ids"])
        response[j, p:n] = True
        prefix.append(p)
    return ids.to(device), response.to(device), torch.tensor(prefix, device=device)


def loss(model, raw, response, prefix, codec, mask_random, probabilities, mask_id, block_size):
    b, length = raw.shape
    z = codec(raw, prefix)
    # Block position 0 is generated from the preceding clean block.
    boundary = torch.arange(length, device=raw.device) % block_size == 0
    eligible = response & ~boundary[None, :]
    choose = mask_random < probabilities.repeat_interleave(block_size, dim=1)
    mask = training_mask(length, block_size, raw.device)
    valid = response | (torch.arange(length, device=raw.device)[None, :] < prefix[:, None])
    mask = mask & valid.repeat(1, 2)[:, None, None, :]
    positions = torch.arange(length, device=raw.device).repeat(2)[None, :].expand(b, -1)
    base = model.module if isinstance(model, DDP) else model
    # Complementary masks: each eligible target is scored once, keeping the
    # same mask draws and denominators across token and relation arms.
    losses = []
    targets = []
    selections = []
    inputs = []
    for chosen in (choose, ~choose):
        masked = eligible & chosen
        noisy = torch.where(masked, mask_id, z)
        # Shifted head i-1 predicts target i; all selected indices >=1.
        row, col = masked.nonzero(as_tuple=True)
        selections.append((row, col - 1))
        targets.append(z[row, col])
        inputs.append(torch.cat((noisy, raw), 1))
    # Score block-first raw tokens from the PREVIOUS clean block. Its mask
    # cannot attend to this target block; no clean-answer leakage.
    first = response & boundary[None, :]
    first[:, 0] = False
    row, col = first.nonzero(as_tuple=True)
    selections[0] = (torch.cat((selections[0][0], row)),
                     torch.cat((selections[0][1], length + col - 1)))
    targets[0] = torch.cat((targets[0], raw[row, col]))
    # One DDP forward for both complementary paths (not two forwards before
    # backward). Chunk heads are recomputed during backward for memory.
    all_ids = torch.cat(inputs, 0)
    sel = (torch.cat((selections[0][0], selections[1][0] + b)),
           torch.cat((selections[0][1], selections[1][1])))
    hidden = model(all_ids, positions.repeat(2, 1), mask.repeat(2, 1, 1, 1), select=sel)
    target = torch.cat(targets)
    for start in range(0, len(target), 32):
        h, t = hidden[start:start + 32], target[start:start + 32]
        losses.append(checkpoint(lambda h, t: F.cross_entropy(base.lm_head(h).float(), t, reduction="sum"),
                                 h, t, use_reentrant=False))
    if not losses:
        raise ValueError("Batch has no response tokens")
    return sum(losses) / response.sum().clamp_min(1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=["token", "relation"], required=True)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--global-batch", type=int, default=12)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-seconds", type=float, default=3600)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--resume", type=Path)
    args = p.parse_args()
    world, rank, local = int(os.getenv("WORLD_SIZE", 1)), int(os.getenv("RANK", 0)), int(os.getenv("LOCAL_RANK", 0))
    if args.global_batch % (world * args.micro_batch):
        raise ValueError("global batch must be divisible by GPU count * micro batch")
    if min(args.steps, args.global_batch, args.micro_batch, args.rank, args.save_every) <= 0:
        raise ValueError("Budgets must be positive")
    device = torch.device("cuda", local)
    torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl", device_id=device)
    m = manifest(args.data)
    from .preflight import require_gate, implementation_hashes
    require_gate(args.data)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise ValueError("Output is not empty; use a new run or explicit --resume")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    model = Model.load(snapshot(), device)
    add_lora(model, args.rank)
    model.gradient_checkpointing = True
    codec = Codec(json.loads((args.data / "codec.json").read_text()), identity=args.arm == "token").to(device)
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad], lr=args.lr)
    meta = dict(arm=args.arm, steps=args.steps, global_batch=args.global_batch, micro_batch=args.micro_batch,
                rank=args.rank, lr=args.lr, seed=args.seed, data_hash=digest(args.data / "manifest.json"),
                model_revision=m["revision"], length=m["length"], block_size=m["block_size"],
                dtype="bf16-base/fp32-lora", objective="complementary response CE plus clean boundary CE",
                adaptation="LoRA all attention and MLP projections; embeddings/head frozen",
                implementation=implementation_hashes())
    start_step = 0
    elapsed_before = 0.
    seen_tokens = 0
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        if {k: ck["meta"][k] for k in meta} != meta:
            raise ValueError("Resume metadata differs (including planned schedule)")
        params = dict(model.named_parameters())
        with torch.no_grad():
            for n, v in ck["adapter"].items():
                params[n].copy_(v)
        optimizer.load_state_dict(ck["optimizer"])
        start_step, elapsed_before = ck["meta"]["completed_steps"], ck["meta"]["training_seconds"]
        seen_tokens = ck["meta"]["original_tokens"]
        if start_step >= args.steps:
            raise ValueError("Checkpoint already completed this schedule; evaluate it directly")
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False) if world > 1 else model
    wrapped.train()
    rows = json.loads((args.data / "train.json").read_text())
    accum = args.global_batch // (world * args.micro_batch)
    started = time.perf_counter()
    completed = start_step
    status = "complete" if start_step >= args.steps else "running"
    for step in range(start_step, args.steps):
        rng = torch.Generator().manual_seed(args.seed + 1000003 * step)
        indices = torch.randint(len(rows), (args.global_batch,), generator=rng).tolist()
        seen_tokens += sum(len(rows[i]["ids"]) for i in indices)
        random_mask = torch.rand(args.global_batch, m["length"], generator=rng)
        probabilities = .001 + .999 * torch.rand(args.global_batch, m["length"] // m["block_size"], generator=rng)
        warmup = max(1, min(20, args.steps // 10))
        scale = min((step + 1) / warmup, 1.) * (.5 + .5 * math.cos(math.pi * max(0, step - warmup) / max(1, args.steps - warmup)))
        for group in optimizer.param_groups:
            group["lr"] = args.lr * scale
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=device)
        for j in range(accum):
            lo = (j * world + rank) * args.micro_batch
            hi = lo + args.micro_batch
            raw, response, prefix = batch(rows, indices[lo:hi], m["length"], m["pad_id"], device)
            context = wrapped.no_sync() if world > 1 and j < accum - 1 else nullcontext()
            with context:
                value = loss(wrapped, raw, response, prefix, codec, random_mask[lo:hi].to(device),
                             probabilities[lo:hi].to(device), m["mask_id"], m["block_size"])
                # Match global token-normalized objective across microbatch / world sizes.
                global_den = sum(min(len(rows[i]["ids"]), m["length"]) - rows[i]["prefix"] for i in indices)
                weighted = value * response.sum() * world / global_den
                if not torch.isfinite(weighted):
                    raise FloatingPointError("Non-finite loss; stop this run")
                weighted.backward()
            total += value.detach() / accum
        torch.nn.utils.clip_grad_norm_([x for x in model.parameters() if x.requires_grad], 1., error_if_nonfinite=True)
        optimizer.step()
        completed = step + 1
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        stop = torch.tensor(int(elapsed >= args.max_seconds), device=device)
        if world > 1:
            dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            dist.all_reduce(total)
            total /= world
        status = "complete" if completed == args.steps else ("budget_exhausted" if stop.item() else "running")
        if rank == 0:
            record = dict(arm=args.arm, step=completed, loss=total.item(), lr=optimizer.param_groups[0]["lr"],
                          seconds=elapsed, training_seconds=elapsed_before + elapsed,
                          original_tokens=seen_tokens,
                          padded_tokens=completed * args.global_batch * m["length"], world_size=world,
                          peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            print(json.dumps(record), flush=True)
            with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
            if completed % args.save_every == 0 or status != "running":
                current = dict(meta, completed_steps=completed, training_seconds=elapsed_before + elapsed,
                               original_tokens=seen_tokens,
                               world_size=world, status=status)
                tmp = args.output / "checkpoint.tmp"
                torch.save(dict(adapter=adapter_state(model), optimizer=optimizer.state_dict(), meta=current), tmp)
                tmp.replace(args.output / "checkpoint.pt")
                write_json(args.output / "status.json", current)
        if world > 1:
            dist.barrier(device_ids=[local])
        if status != "running":
            break
    if world > 1:
        dist.destroy_process_group()
    if status == "budget_exhausted":
        raise SystemExit("Time budget reached; checkpoint saved. No automatic continuation.")


if __name__ == "__main__":
    main()

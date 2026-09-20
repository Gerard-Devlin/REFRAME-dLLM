"""Full-parameter matched one-epoch training with DDP and sharded AdamW."""
import argparse
from contextlib import nullcontext
import json
import math
import os
from pathlib import Path
import time
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .codec import Codec
from .common import digest, manifest, snapshot, write_json
from .model import Model, training_mask


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
    # Complementary masks: each eligible target is scored once, keeping the
    # same mask draws and denominators across token and relation arms.
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
    numerator = model(all_ids, positions.repeat(2, 1), mask.repeat(2, 1, 1, 1),
                      select=sel, targets=torch.cat(targets))
    return numerator / response.sum().clamp_min(1)


def main():
    import datetime
    import statistics
    import subprocess
    import sys
    from .full_state import epoch_batches, lr_scale, save_checkpoint, restore_checkpoint, barrier
    from .preflight import require_gate, implementation_hashes
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=["token", "relation"], required=True)
    p.add_argument("--steps", type=int, default=0, help="0 = one complete epoch; positive = explicit short budget")
    p.add_argument("--global-batch", type=int, default=12)
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-seconds", type=float, default=0, help="0 = no time limit")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-every", type=int, default=1000)
    p.add_argument("--eval-limit", type=int, default=32)
    p.add_argument("--final-eval-limit", type=int, default=256)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--rounds", default="2,4,8,16")
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int, default=0, help="Save a resumable deliberate pause (smoke only)")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    world, rank, local = (int(os.getenv(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    if min(args.global_batch, args.micro_batch, args.save_every) < 1 or args.steps < 0 or args.eval_every < 0:
        raise ValueError("Invalid training budget")
    if args.global_batch % (world * args.micro_batch):
        raise ValueError("Global batch must be divisible by GPU count * microbatch")
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl", device_id=device, timeout=datetime.timedelta(hours=6))
    m = manifest(args.data)
    require_gate(args.data)
    hashes = implementation_hashes()
    if not args.smoke:
        gate = json.loads((args.data / "full_smoke.json").read_text())
        expected_gate = dict(implementation=hashes, data_hash=digest(args.data / "manifest.json"),
                             world_size=world, global_batch=args.global_batch, micro_batch=args.micro_batch)
        if not gate.get("pass") or any(gate.get(k) != v for k, v in expected_gate.items()):
            raise ValueError("Run full smoke with this implementation, data and GPU/batch configuration first")
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()) and not args.resume:
            raise ValueError("Output exists; use a new run or --resume")
        args.output.mkdir(parents=True, exist_ok=True)
    barrier()
    torch.manual_seed(args.seed)
    model = Model.load(snapshot(), device, dtype=torch.float32)
    model.requires_grad_(True)
    model.gradient_checkpointing = True
    total_params = sum(x.numel() for x in model.parameters())
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    assert trainable == total_params
    if rank == 0:
        print(f"FULL training: total={total_params:,}, trainable={trainable:,}, world_size={world}", flush=True)
    wrapped = DDP(model, device_ids=[local], broadcast_buffers=False, gradient_as_bucket_view=True) if world > 1 else model
    if world > 1:
        from torch.distributed.optim import ZeroRedundancyOptimizer
        optimizer = ZeroRedundancyOptimizer(model.parameters(), optimizer_class=torch.optim.AdamW,
                                            lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    rows = json.loads((args.data / "train.json").read_text())
    batches = epoch_batches(len(rows), args.global_batch, args.seed)
    if args.steps > len(batches):
        raise ValueError("Explicit steps exceed one epoch; use 0 for full epoch")
    steps = args.steps or len(batches)
    meta = dict(arm=args.arm, steps=steps, global_batch=args.global_batch, micro_batch=args.micro_batch,
                lr=args.lr, weight_decay=args.weight_decay, seed=args.seed, sampling="one-epoch-no-replacement",
                data_hash=digest(args.data / "manifest.json"), model_revision=m["revision"],
                length=m["length"], block_size=m["block_size"], dtype="fp32-master/bf16-autocast",
                objective="complementary response CE plus clean boundary CE", adaptation="full",
                implementation=hashes, trainable_parameters=trainable, total_parameters=total_params)
    start, seen, supervised, train_seconds, eval_seconds = 0, 0, 0, 0., 0.
    if args.resume:
        old = restore_checkpoint(args.resume, model, optimizer, rank, world, meta)
        start, seen, supervised = old["completed_steps"], old["original_tokens"], old["supervised_tokens"]
        train_seconds, eval_seconds = old["training_seconds"], old["evaluation_seconds"]
        if old["sampler_cursor"] != sum(len(b) for b in batches[:start]):
            raise ValueError("Sampler cursor mismatch")
        if start >= steps:
            raise ValueError("Training is already complete; evaluate the exported full checkpoint directly")
    codec = Codec(json.loads((args.data / "codec.json").read_text()), identity=args.arm == "token").to(device)
    writer = None
    if rank == 0:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(str(args.output / "tensorboard"), flush_secs=5, purge_step=start + 1 if args.resume else None)
    wrapped.train()
    recent_times = []
    torch.cuda.reset_peak_memory_stats()
    invocation_start = time.perf_counter()
    try:
        for step in range(start, steps):
            torch.cuda.synchronize()
            tick = time.perf_counter()
            indices = batches[step]
            den = sum(min(len(rows[i]["ids"]), m["length"]) - rows[i]["prefix"] for i in indices)
            if den <= 0:
                raise ValueError("Empty supervised batch")
            seen += sum(min(len(rows[i]["ids"]), m["length"]) for i in indices)
            supervised += den
            rng = torch.Generator().manual_seed(args.seed + 1000003 * step)
            slots = math.ceil(len(indices) / (world * args.micro_batch)) * world * args.micro_batch
            random_mask = torch.rand(slots, m["length"], generator=rng)
            probs = .001 + .999 * torch.rand(slots, m["length"] // m["block_size"], generator=rng)
            for group in optimizer.param_groups:
                group["lr"] = args.lr * lr_scale(step, steps)
            optimizer.zero_grad(set_to_none=True)
            numerator = torch.zeros((), device=device)
            accum = slots // (world * args.micro_batch)
            for j in range(accum):
                lo = (j * world + rank) * args.micro_batch
                selected = indices[lo:lo + args.micro_batch]
                active = bool(selected)
                # Empty ranks still participate in DDP using a zero-weight valid example.
                selected = selected or [indices[0]]
                raw, response, prefix = batch(rows, selected, m["length"], m["pad_id"], device)
                count = response.sum() if active else torch.zeros((), device=device)
                with wrapped.no_sync() if world > 1 and j < accum - 1 else nullcontext():
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        value = loss(wrapped, raw, response, prefix, codec,
                                     random_mask[lo:lo + len(selected)].to(device),
                                     probs[lo:lo + len(selected)].to(device), m["mask_id"], m["block_size"])
                        weighted = value * count * world / den
                    if not torch.isfinite(weighted):
                        raise FloatingPointError("Non-finite full-training loss")
                    weighted.backward()
                numerator += value.detach() * count
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            if world > 1:
                dist.all_reduce(numerator)
            torch.cuda.synchronize()
            step_seconds = time.perf_counter() - tick
            duration = torch.tensor(step_seconds, device=device)
            if world > 1:
                dist.all_reduce(duration, op=dist.ReduceOp.MAX)
            step_seconds = duration.item()
            recent_times.append(step_seconds)
            train_seconds += step_seconds
            completed = step + 1
            peaks = torch.tensor([torch.cuda.max_memory_allocated() / 2**30,
                                  torch.cuda.max_memory_reserved() / 2**30], device=device)
            all_peaks = [torch.zeros_like(peaks) for _ in range(world)]
            if world > 1:
                dist.all_gather(all_peaks, peaks)
            else:
                all_peaks[0] = peaks
            steady = statistics.median(recent_times[3:] or recent_times)
            stop = torch.tensor(int(args.max_seconds > 0 and time.perf_counter() - invocation_start >= args.max_seconds), device=device)
            if world > 1:
                dist.all_reduce(stop, op=dist.ReduceOp.MAX)
            status = "complete" if completed == steps else ("paused" if args.stop_after and completed >= args.stop_after else
                         ("budget_exhausted" if stop.item() else "running"))
            record = dict(arm=args.arm, step=completed, steps=steps, loss=numerator.item() / den,
                          lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm),
                          step_seconds=step_seconds, training_seconds=train_seconds, evaluation_seconds=eval_seconds,
                          original_tokens=seen, supervised_tokens=supervised, world_size=world,
                          tokens_per_second=sum(min(len(rows[i]["ids"]), m["length"]) for i in indices) / step_seconds,
                          eta_seconds=(steps - completed) * steady, steady_step_seconds=steady,
                          peak_gib_by_rank=[x[0].item() for x in all_peaks],
                          reserved_gib_by_rank=[x[1].item() for x in all_peaks])
            if rank == 0:
                print(json.dumps(record), flush=True)
                with (args.output / "metrics.jsonl").open("a") as f:
                    f.write(json.dumps(record) + "\n")
                for tag, key in (("train/loss", "loss"), ("train/learning_rate", "lr"), ("train/grad_norm", "grad_norm"),
                                 ("progress/original_tokens", "original_tokens"), ("progress/supervised_tokens", "supervised_tokens"),
                                 ("performance/step_seconds", "step_seconds"), ("performance/eta_seconds", "eta_seconds"),
                                 ("performance/original_tokens_per_second", "tokens_per_second")):
                    writer.add_scalar(tag, record[key], completed)
                for i, peak in enumerate(record["peak_gib_by_rank"]):
                    writer.add_scalar(f"memory/rank_{i}_peak_gib", peak, completed)
            evaluate = bool(args.eval_every and (completed % args.eval_every == 0 or status == "complete"))
            if completed % args.save_every == 0 or status != "running" or evaluate:
                current = dict(meta, completed_steps=completed, original_tokens=seen, supervised_tokens=supervised,
                               sampler_cursor=sum(len(b) for b in batches[:completed]), scheduler_step=completed,
                               training_seconds=train_seconds, evaluation_seconds=eval_seconds,
                               status=status, steady_step_seconds=steady, peak_gib_by_rank=record["peak_gib_by_rank"],
                               evaluation_data_hash=digest(args.data / "gsm8k_dev_full.json"))
                saved = save_checkpoint(args.output, model, optimizer, current, rank, world)
                if evaluate:
                    eval_tick = time.perf_counter()
                    error = [None]
                    if rank == 0:
                        # Eval lives in a separate process: BF16 full weights, no optimizer or training autocast state.
                        destination = args.output.parent / "eval" / ("final" if status == "complete" else f"step_{completed:08d}")
                        cmd = [sys.executable, "-u", "-m", "relation_block.evaluate", "--data", str(args.data),
                               "--checkpoint", str(saved), "--output", str(destination), "--limit",
                               str(args.final_eval_limit if status == "complete" else args.eval_limit),
                               "--rounds", args.rounds, "--max-new-tokens", str(args.max_new_tokens)]
                        optimizer.zero_grad(set_to_none=True)
                        torch.cuda.empty_cache()
                        env = dict(os.environ)
                        env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
                        for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                            env.pop(key, None)
                        try:
                            subprocess.run(cmd, env=env, check=True)
                        except Exception as exc:
                            error[0] = str(exc)
                    if world > 1:
                        dist.broadcast_object_list(error, src=0)
                    if error[0]:
                        raise RuntimeError(f"Evaluation failed; checkpoint saved: {error[0]}")
                    eval_seconds += time.perf_counter() - eval_tick
                    if rank == 0:
                        write_json(args.output / "evaluation_time.json", {"seconds": eval_seconds})
                        current["evaluation_seconds"] = eval_seconds
                        write_json(saved / "metadata.json", dict(current, world_size=world, format="full-v1"))
                        write_json(args.output / "status.json", dict(current, world_size=world, format="full-v1"))
            if status != "running":
                break
    finally:
        if writer is not None:
            writer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    if status == "budget_exhausted":
        raise SystemExit("Time limit reached. Full checkpoint saved; resume explicitly.")


if __name__ == "__main__":
    main()

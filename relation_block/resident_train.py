"""Resident multi-GPU continuation: all ranks train and evaluate without exiting."""
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
from .model import Model


from .train import batch, loss


def main():
    import datetime
    import statistics
    from .full_state import epoch_batches, lr_scale, save_checkpoint, restore_checkpoint, barrier
    from .preflight import require_gate, implementation_hashes
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--arm", choices=["token", "relation"], required=True)
    p.add_argument("--steps", type=int, default=0, help="0 = one complete epoch; positive = explicit short budget")
    p.add_argument("--global-batch", type=int, default=0, help="0 = twice the GPU count")
    p.add_argument("--micro-batch", type=int, default=1)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=.01)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--max-seconds", type=float, default=0, help="0 = no time limit")
    p.add_argument("--save-every", type=int, default=500)
    p.add_argument("--eval-limit", type=int, default=32)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--rounds", default="2,4,8,16")
    p.add_argument("--resume", type=Path)
    p.add_argument("--stop-after", type=int, default=0, help="Save a resumable deliberate pause (smoke only)")
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--token-budget", type=int, default=2000000)
    p.add_argument("--eval-token-budgets", default="500000,1000000,2000000")
    p.add_argument("--reconstruction-limit", type=int, default=32)
    p.add_argument("--boundary-mode", choices=['clean', 'masked'], default='clean')
    p.add_argument("--boundary-probes", action='store_true')
    p.add_argument("--eval-at-start", action='store_true')
    p.add_argument("--keep-checkpoints", action='store_true')
    args = p.parse_args()
    world, rank, local = (int(os.getenv(k, d)) for k, d in (("WORLD_SIZE", 1), ("RANK", 0), ("LOCAL_RANK", 0)))
    args.global_batch = args.global_batch or 2 * world
    if (args.boundary_probes or args.boundary_mode == 'masked') and args.arm != 'token':
        raise ValueError('Boundary A/B is token-only')
    from .boundary import masked_loss
    objective_loss = loss if args.boundary_mode == 'clean' else masked_loss
    if min(args.global_batch, args.micro_batch, args.save_every, args.eval_limit, args.reconstruction_limit) < 1 or args.steps < 0:
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
    from .continuation import milestones
    requested = sorted(set([int(x) for x in args.eval_token_budgets.split(",") if int(x) <= args.token_budget] + [args.token_budget]))
    if args.token_budget < 1 or min(requested) < 1:
        raise ValueError("Token budgets must be positive")
    points = milestones(rows, batches, m["length"], requested)
    evaluation_steps = {x["step"] for x in points}
    stop_step = points[-1]["step"]
    if stop_step > steps:
        raise ValueError("Token budget exceeds step schedule")
    meta = dict(arm=args.arm, steps=steps, global_batch=args.global_batch, micro_batch=args.micro_batch,
                lr=args.lr, weight_decay=args.weight_decay, seed=args.seed, sampling="one-epoch-no-replacement",
                data_hash=digest(args.data / "manifest.json"), model_revision=m["revision"],
                length=m["length"], block_size=m["block_size"], dtype="fp32-master/bf16-autocast",
                objective="complementary response CE plus clean boundary CE", adaptation="full",
                implementation=hashes, trainable_parameters=trainable, total_parameters=total_params)
    if args.boundary_probes or args.boundary_mode == 'masked':
        meta.update(boundary_mode=args.boundary_mode, boundary_sha256=digest(Path(__file__).with_name('boundary.py')),
                    objective=('complementary response CE plus clean boundary CE' if args.boundary_mode == 'clean'
                               else 'complementary response CE including boundaries; preceding noisy-position logits'))
    start, seen, supervised, train_seconds, eval_seconds = 0, 0, 0, 0., 0.
    if args.resume:
        old = restore_checkpoint(args.resume, model, optimizer, rank, world, meta)
        start, seen, supervised = old["completed_steps"], old["original_tokens"], old["supervised_tokens"]
        train_seconds, eval_seconds = old["training_seconds"], old["evaluation_seconds"]
        if old["sampler_cursor"] != sum(len(b) for b in batches[:start]):
            raise ValueError("Sampler cursor mismatch")
        if start >= steps:
            raise ValueError("Training is already complete; evaluate the exported full checkpoint directly")
    if start >= stop_step:
        raise ValueError("Checkpoint already reached this token budget")
    driver_hash = digest(Path(__file__))
    evaluator_hash = digest(Path(__file__).with_name('resident_eval.py'))
    if args.resume and old.get("resident_driver") not in (None, driver_hash):
        raise ValueError("Resident driver changed since checkpoint")
    if args.resume and old.get("resident_evaluator") not in (None, evaluator_hash):
        raise ValueError("Resident evaluator changed since checkpoint")
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
        if args.eval_at_start and not args.resume:
            from .resident_eval import evaluate_resident
            initial_tick = time.perf_counter()
            evaluate_resident(model, args.data, args.output.parent / 'eval/step_00000000', args.arm, rank, world,
                              args.eval_limit, args.rounds, args.max_new_tokens,
                              args.reconstruction_limit, boundary_probes=args.boundary_probes)
            eval_seconds += time.perf_counter() - initial_tick
        for step in range(start, stop_step):
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
                        value = objective_loss(wrapped, raw, response, prefix, codec,
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
            status = "complete" if completed == stop_step else ("paused" if args.stop_after and completed >= args.stop_after else
                         ("budget_exhausted" if stop.item() else "running"))
            record = dict(arm=args.arm, step=completed, steps=steps, loss=numerator.item() / den,
                          lr=optimizer.param_groups[0]["lr"], grad_norm=float(grad_norm),
                          step_seconds=step_seconds, training_seconds=train_seconds, evaluation_seconds=eval_seconds,
                          original_tokens=seen, supervised_tokens=supervised, world_size=world,
                          tokens_per_second=sum(min(len(rows[i]["ids"]), m["length"]) for i in indices) / step_seconds,
                          eta_seconds=(stop_step - completed) * steady, steady_step_seconds=steady,
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
            evaluate = completed in evaluation_steps
            if completed % args.save_every == 0 or status != "running" or evaluate:
                current = dict(meta, resident_driver=driver_hash, resident_evaluator=evaluator_hash,
                               completed_steps=completed, original_tokens=seen, supervised_tokens=supervised,
                               sampler_cursor=sum(len(b) for b in batches[:completed]), scheduler_step=completed,
                               training_seconds=train_seconds, evaluation_seconds=eval_seconds,
                               status=status, steady_step_seconds=steady, peak_gib_by_rank=record["peak_gib_by_rank"],
                               evaluation_data_hash=digest(args.data / "gsm8k_dev_full.json"))
                checkpoint_root = args.output / 'checkpoints' / f'budget_{completed:08d}' if args.keep_checkpoints else args.output
                saved = save_checkpoint(checkpoint_root, model, optimizer, current, rank, world)
                if args.keep_checkpoints and rank == 0:
                    write_json(args.output / 'latest.json', dict(checkpoint=str(saved.relative_to(args.output.resolve()))))
                    write_json(args.output / 'status.json', dict(current, world_size=world, format='full-v1'))
                if evaluate:
                    eval_tick = time.perf_counter()
                    # Every rank keeps model/optimizer resident and evaluates a disjoint shard.
                    from .resident_eval import evaluate_resident
                    optimizer.zero_grad(set_to_none=True)
                    destination = args.output.parent / "eval" / f"step_{completed:08d}"
                    evaluate_resident(model, args.data, destination, args.arm, rank, world,
                                      args.eval_limit, args.rounds, args.max_new_tokens,
                                      args.reconstruction_limit, boundary_probes=args.boundary_probes)
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

"""Matched small-head training on offline native transitions; backbone not loaded."""

import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from .collect import distributed, barrier
from .data import TraceDataset, build_inputs, save_torch, sha256, write_json
from .head import HeadConfig, ResidualHead, distillation_loss
from .metrics import metrics_batch, finalize


ARMS = ("calibration", "blind", "relation")


def epoch_indices(size, batch_per_gpu, world, seed, epoch):
    generator = torch.Generator().manual_seed(seed + epoch)
    order = torch.randperm(size, generator=generator).tolist()
    total = batch_per_gpu * world
    return [order[start:start+total] for start in range(0, size, total)]


def local_records(dataset, indices, rank, batch_per_gpu):
    selected = indices[rank * batch_per_gpu:(rank + 1) * batch_per_gpu]
    records = [dataset[index] for index in selected]
    # Last global batch is not dropped or duplicated in the objective. A rank
    # with no real samples still participates in exactly one DDP backward.
    if not records:
        padding = dict(dataset[indices[0]])
        padding["eligible"] = torch.zeros_like(padding["eligible"])
        records = [padding]
    return records


def merge_metrics(local, world):
    if world == 1:
        return local
    gathered = [None] * world
    dist.all_gather_object(gathered, local)
    result = {}
    for item in gathered:
        for key, value in item.items():
            result[key] = result.get(key, 0) + value
    return result


@torch.no_grad()
def evaluate(head, dataset, table, device, rank, world, batch_size, threshold):
    if head is not None:
        head.eval()
    local = {}
    indices = list(range(rank, len(dataset), world))
    for start in range(0, len(indices), batch_size):
        records = [dataset[index] for index in indices[start:start+batch_size]]
        batch = build_inputs(records, table, device)
        predicted = head(batch) if head is not None else batch["base_log_probs"]
        teacher_probs = torch.stack([r["teacher_probs"] for r in records]).to(device)
        teacher_top1 = torch.stack([r["teacher_top1"] for r in records]).to(device)
        teacher_selected = torch.stack([r["teacher_selected"] for r in records]).to(device)
        numbers = metrics_batch(predicted, batch, teacher_probs, teacher_top1, teacher_selected, threshold)
        for key, value in numbers.items():
            local[key] = local.get(key, 0) + float(value)
        local["teacher_forward_seconds_sum"] = local.get("teacher_forward_seconds_sum", 0) + sum(
            float(r["teacher_forward_seconds"]) for r in records)
    total = merge_metrics(local, world)
    report = finalize(total)
    report["mean_instrumented_teacher_forward_gpu_seconds"] = (
        total["teacher_forward_seconds_sum"] / total["transitions"])
    return report


@torch.no_grad()
def head_cost(head, record, table, device, repeats=100):
    """Isolated batch-one feature lookup + head GPU cost, NOT decoder latency."""
    batch = {key: record[key].unsqueeze(0).to(device) for key in
             ("hidden", "candidates", "base_log_probs", "commit_ids", "committed", "eligible")}
    def execute():
        batch["candidate_features"] = table[batch["candidates"]]
        batch["commit_features"] = table[batch["commit_ids"]] * batch["committed"].unsqueeze(-1)
        return head(batch)
    for _ in range(10):
        execute()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        execute()
    end.record()
    end.synchronize()
    return dict(mean_gpu_seconds=start.elapsed_time(end) / (1000 * repeats), repeats=repeats,
                batch_size=1, includes="frozen feature lookup and head only",
                excludes="candidate extraction, softmax, native sampler, cache updates and host transfer",
                note="CUDA event interval can include stream idle gaps from host dispatch. "
                     "Not a model speedup claim or end-to-end latency gate")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--traces", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-per-gpu", type=int, default=32)
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()
    if min(args.epochs, args.batch_per_gpu, args.width) < 1 or args.lr <= 0:
        p.error("Positive epochs/batch/width/lr required")
    manifest_path = args.traces / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("format_version") != 1:
        raise ValueError("A completed version-1 trace collection is required")
    train = TraceDataset(args.traces, manifest, "train")
    heldout = TraceDataset(args.traces, manifest, "heldout")
    train_prompts = {s["prompt_id"] for s in manifest["shards"] if s["split"] == "train"}
    heldout_prompts = {s["prompt_id"] for s in manifest["shards"] if s["split"] == "heldout"}
    if train_prompts & heldout_prompts:
        raise ValueError("Train/held-out prompt leakage")
    rank, world, device = distributed()
    settings = dict(manifest_sha256=sha256(manifest_path), epochs=args.epochs,
                    batch_per_gpu=args.batch_per_gpu, world=world, width=args.width,
                    lr=args.lr, seed=args.seed,
                    implementation_sha256={name: sha256(Path(__file__).parent / name)
                        for name in ("train.py", "head.py", "metrics.py", "data.py")})
    saved = None
    if args.resume:
        saved = torch.load(args.output / "checkpoint.pt", map_location="cpu", weights_only=True)
        if saved["settings"] != settings:
            raise ValueError("Resume configuration, traces, world size or implementation changed")
    if rank == 0:
        if not args.resume and args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Training output exists; use a new directory or --resume")
        args.output.mkdir(parents=True, exist_ok=True)
        if sha256(args.traces / "features.pt") != manifest["feature_sha256"]:
            raise ValueError("Frozen feature table changed")
        for shard in manifest["shards"]:
            if sha256(args.traces / shard["file"]) != shard["sha256"]:
                raise ValueError(f"Trace shard changed: {shard['file']}")
        if not args.resume:
            write_json(args.output / "plan.json", dict(**settings, arms=ARMS, train_pairs=len(train),
                        heldout_pairs=len(heldout), objective="global eligible-token-weighted coarse teacher KL",
                        note="No backbone weights loaded or trained. Offline teacher prediction only."))
    barrier()
    table = torch.load(args.traces / "features.pt", map_location="cpu", weights_only=True)["table"].to(device)
    writer = None
    if rank == 0:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(str(args.output / "tensorboard"), flush_secs=10)
        except ImportError:
            print("TensorBoard unavailable; JSONL metrics remain enabled", flush=True)
    baseline = evaluate(None, heldout, table, device, rank, world, args.batch_per_gpu, manifest["threshold"])
    results = {"reuse": baseline}
    for arm_index, arm in enumerate(ARMS):
        if saved is not None and arm_index < saved["arm_index"]:
            results[arm] = json.loads((args.output / arm / "summary.json").read_text())
            continue
        torch.manual_seed(args.seed)
        config = HeadConfig(manifest["hidden_size"], manifest["feature_size"], args.width, arm)
        head = ResidualHead(config).to(device)
        wrapped = DDP(head, device_ids=[device.index], broadcast_buffers=False) if world > 1 else head
        optimizer = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=0.01)
        start_epoch = 0
        if saved is not None and arm_index == saved["arm_index"]:
            head.load_state_dict(saved["head"])
            optimizer.load_state_dict(saved["optimizer"])
            start_epoch = saved["epoch_completed"]
        count = sum(p.numel() for p in head.parameters())
        if rank == 0:
            print(f"arm={arm} trainable_parameters={count} start_epoch={start_epoch}", flush=True)
        steps_per_epoch = math.ceil(len(train) / (args.batch_per_gpu * world))
        steps = steps_per_epoch * args.epochs
        trained_seconds = saved.get("training_seconds", 0) if saved is not None and arm_index == saved["arm_index"] else 0
        for epoch in range(start_epoch, args.epochs):
            head.train()
            batches = epoch_indices(len(train), args.batch_per_gpu, world, args.seed, epoch)
            epoch_started = time.perf_counter()
            running_kl = running_tokens = 0.0
            for offset, indices in enumerate(batches):
                step = epoch * steps_per_epoch + offset + 1
                warmup = max(1, math.ceil(0.03 * steps))
                factor = step / warmup if step <= warmup else 0.5 * (1 + math.cos(math.pi * (step-warmup) / max(1, steps-warmup)))
                for group in optimizer.param_groups:
                    group["lr"] = args.lr * factor
                records = local_records(train, indices, rank, args.batch_per_gpu)
                batch = build_inputs(records, table, device)
                targets = torch.stack([r["teacher_probs"] for r in records]).to(device)
                optimizer.zero_grad(set_to_none=True)
                log_probs = wrapped(batch)
                numerator, denominator = distillation_loss(log_probs, targets, batch["eligible"])
                global_count = denominator.detach().clone()
                if world > 1:
                    dist.all_reduce(global_count)
                if not int(global_count):
                    raise ValueError("Training batch has no eligible teacher targets")
                (numerator * world / global_count).backward()
                norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                combined = torch.stack((numerator.detach(), denominator.float()))
                if world > 1:
                    dist.all_reduce(combined)
                running_kl += float(combined[0])
                running_tokens += float(combined[1])
                if rank == 0 and (step % 10 == 0 or offset == len(batches)-1):
                    elapsed = time.perf_counter() - epoch_started
                    stats = dict(arm=arm, epoch=epoch+1, step=step, total_steps=steps,
                                 token_weighted_kl=running_kl/running_tokens, grad_norm=float(norm),
                                 supervised_positions=int(running_tokens),
                                 eta_seconds=elapsed/(offset+1)*(steps-step))
                    with (args.output / "metrics.jsonl").open("a", encoding="utf-8") as stream:
                        stream.write(json.dumps(stats) + "\n")
                    print(json.dumps(stats), flush=True)
                    if writer is not None:
                        for key in ("token_weighted_kl", "grad_norm", "eta_seconds"):
                            writer.add_scalar(f"{arm}/train/{key}", stats[key], step)
            torch.cuda.synchronize()
            trained_seconds += time.perf_counter() - epoch_started
            eval_started = time.perf_counter()
            metrics = evaluate(head, heldout, table, device, rank, world, args.batch_per_gpu, manifest["threshold"])
            if rank == 0:
                write_json(args.output / arm / f"epoch_{epoch+1}.json", metrics)
                if writer is not None:
                    for key, value in metrics.items():
                        if isinstance(value, (float, int)):
                            writer.add_scalar(f"{arm}/heldout/{key}", value, (epoch+1)*steps_per_epoch)
                    writer.flush()
                state = dict(settings=settings, arm_index=arm_index, epoch_completed=epoch+1,
                             head=head.state_dict(), optimizer=optimizer.state_dict(), training_seconds=trained_seconds)
                save_torch(args.output / "checkpoint.pt", state)
                print(f"arm={arm} epoch={epoch+1} evaluation_seconds={time.perf_counter()-eval_started:.2f} "
                      f"metrics={json.dumps(metrics)}", flush=True)
            barrier()
        final = evaluate(head, heldout, table, device, rank, world, args.batch_per_gpu, manifest["threshold"])
        if rank == 0:
            head.eval()
            cost = head_cost(head, heldout[0], table, device)
            final.update(trainable_parameters=count, config=asdict(config), training_seconds=trained_seconds,
                         isolated_head_cost=cost)
            save_torch(args.output / arm / "head.pt", dict(config=asdict(config), state=head.state_dict(),
                       manifest_sha256=settings["manifest_sha256"]))
            write_json(args.output / arm / "summary.json", final)
        barrier()
        if world > 1:
            shared = [final if rank == 0 else None]
            dist.broadcast_object_list(shared, src=0)
            final = shared[0]
        results[arm] = final
        del wrapped, head, optimizer
    if rank == 0:
        report = dict(status="complete", results=results, settings=settings,
                      task_accuracy_measured=False, end_to_end_speedup_measured=False,
                      note="Diagnose next-forward prediction, especially teacher-changed positions. "
                           "A passing offline result does not authorize a speedup or preserved-accuracy claim.")
        write_json(args.output / "summary.json", report)
        if writer is not None:
            writer.close()
        from .report import render
        print(render(report), flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

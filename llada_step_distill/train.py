from __future__ import annotations

import contextlib
import json
import itertools
import math
import os
import random
import shutil
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fastv_dllm.common import extract_answer, prompt_ids
from .collect import distributed
from .core import ExperimentConfig, SEED, atomic_json, digest, read_json, schedule_factor
from .data import AlignedShardDataset, build_acceleration_state, build_retention_state
from .decode import generate_fixed_quota
from .evaluate import fixed_split, load_gsm8k
from .model import (
    adapter_state_dict,
    detach_output_head,
    inject_lora,
    load_adapter,
    load_model,
    max_log_probs,
    selected_log_probs,
)

METRIC_KEYS = ("ce", "ranking", "anchor_kl", "sft", "supervised_tokens", "pair_release", "extra_error_release", "rank_margin", "invalid")


def _student_hidden(model, ids: list[int]) -> torch.Tensor:
    device = next(model.parameters()).device
    source = torch.tensor([ids], dtype=torch.long, device=device)
    return model(input_ids=source, use_cache=False).logits[0]


def acceleration_loss(model, head, prepared: dict, supervision: dict, config: ExperimentConfig):
    if not supervision.get("valid"):
        # Exercise the whole DDP graph so different ranks may safely encounter invalid records.
        ids = prepared["prompt_ids"] + [126336] * config.generation_length
        hidden = _student_hidden(model, ids)
        zero = hidden.sum() * 0.0
        return zero, {"ce": 0.0, "ranking": 0.0, "anchor_kl": 0.0, "pair_release": 0.0, "extra_error_release": 0.0, "rank_margin": 0.0, "invalid": 1.0}
    ids, _, absolute = build_acceleration_state(prepared, supervision["block"], supervision["reveal_mask"])
    hidden = _student_hidden(model, ids)
    targets = supervision["targets"]
    positions = torch.tensor([absolute + item["offset"] for item in targets], device=hidden.device)
    tokens = torch.tensor([item["token"] for item in targets], device=hidden.device)
    target_hidden = hidden.index_select(0, positions)
    target_lp = selected_log_probs(target_hidden, tokens, head)
    ce = -target_lp.mean()

    target_offsets = {item["offset"] for item in targets}
    remaining = [
        offset for offset in range(config.block_length)
        if not supervision["reveal_mask"] & (1 << offset) and offset not in target_offsets
    ]
    if remaining:
        remaining_hidden = hidden.index_select(
            0, torch.tensor([absolute + offset for offset in remaining], device=hidden.device)
        )
        other_values = max_log_probs(remaining_hidden, head)
        other_max = other_values.amax()
        ranking = F.relu(config.margin + other_max - target_lp.amin())
        rank_margin = target_lp.amin() - other_max
        extra_error = (other_values > target_lp.amin()).float().mean()
    else:
        ranking = ce * 0.0
        rank_margin = torch.tensor(0.0, device=ce.device)
        extra_error = ce * 0.0

    anchor_losses = []
    for anchor in supervision.get("anchors", []):
        anchor_hidden = hidden[absolute + anchor["offset"] : absolute + anchor["offset"] + 1]
        anchor_ids = torch.tensor([anchor["ids"]], device=hidden.device)
        student_lp = selected_log_probs(anchor_hidden, anchor_ids, head)[0]
        teacher_lp = torch.tensor(anchor["log_probs"], dtype=torch.float32, device=hidden.device)
        teacher_p = teacher_lp.exp()
        student_residual = torch.clamp(1.0 - student_lp.exp().sum(), min=1e-12).log()
        teacher_residual_lp = torch.tensor(anchor["residual_log_prob"], device=hidden.device)
        kl = (teacher_p * (teacher_lp - student_lp)).sum()
        kl = kl + teacher_residual_lp.exp() * (teacher_residual_lp - student_residual)
        anchor_losses.append(kl)
    anchor_kl = torch.stack(anchor_losses).mean() if anchor_losses else ce * 0.0
    loss = ce + ranking + anchor_kl
    pair_release = ((rank_margin > 0) | torch.tensor(not remaining, device=ce.device)).float()
    return loss, {
        "ce": float(ce.detach()), "ranking": float(ranking.detach()),
        "anchor_kl": float(anchor_kl.detach()), "pair_release": float(pair_release.detach()),
        "extra_error_release": float(extra_error.detach()), "rank_margin": float(rank_margin.detach()), "invalid": 0.0,
    }


def retention_loss(model, head, prepared: dict, supervision: dict):
    ids, reference, masked, p_mask = build_retention_state(prepared, int(supervision["mask_seed"]))
    hidden = _student_hidden(model, ids)
    prompt = len(prepared["prompt_ids"])
    positions = [prompt + index for index, selected in enumerate(masked) if selected]
    tokens = [reference[index] for index, selected in enumerate(masked) if selected]
    selected = hidden.index_select(0, torch.tensor(positions, device=hidden.device))
    labels = torch.tensor(tokens, device=hidden.device)
    log_probs = selected_log_probs(selected, labels, head)
    answer_length = min(len(prepared["response_ids"]), len(reference))
    loss = -log_probs.sum() / (p_mask * answer_length)
    return loss, {"sft": float(loss.detach()), "supervised_tokens": len(tokens)}


def _iter_rank(dataset: AlignedShardDataset, stage: str, rank: int, world: int, overfit_records: int | None = None):
    if overfit_records:
        selected = []
        for prepared, supervision in dataset.iter_epoch(0):
            if stage == "b" and not prepared.get("real_transition", False):
                continue
            selected.append((prepared, supervision))
            if len(selected) == overfit_records:
                break
        local = selected[rank::world]
        if not local:
            raise ValueError("overfit_records must be at least world_size")
        yield from itertools.cycle(local)
        return
    selected_index = 0
    for prepared, supervision in dataset.iter_epoch(0):
        if stage == "b" and not prepared.get("real_transition", False):
            continue
        if selected_index % world == rank:
            yield prepared, supervision
        selected_index += 1


def _optimizer(model, lr: float, weight_decay: float, world: int):
    params = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if world == 1:
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    from torch.distributed.optim import ZeroRedundancyOptimizer

    return ZeroRedundancyOptimizer(
        params, optimizer_class=torch.optim.AdamW, lr=lr, weight_decay=weight_decay,
        parameters_as_bucket_view=True, overlap_with_ddp=False,
    )


def _local_optimizer_state(optimizer):
    local = getattr(optimizer, "optim", optimizer)
    return local.state_dict()


def _load_local_optimizer_state(optimizer, state):
    local = getattr(optimizer, "optim", optimizer)
    local.load_state_dict(state)


def save_checkpoint(output: Path, model, optimizer, state: dict, rank: int, world: int, best: bool = False):
    checkpoint = output / f"checkpoint_{state['update']:08d}"
    checkpoint.mkdir(parents=True, exist_ok=True)
    rank_state = {
        "optimizer": _local_optimizer_state(optimizer), "state": state,
        "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }
    torch.save(rank_state, checkpoint / f"rank_{rank:03d}.pt")
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        torch.save({"adapter": adapter_state_dict(model.module if hasattr(model, "module") else model)}, checkpoint / "adapter.pt")
        atomic_json(checkpoint / "state.json", state)
        atomic_json(checkpoint / "complete.json", {"world_size": world, "update": state["update"]})
        if best:
            (output / "best.txt").write_text(checkpoint.name, encoding="utf-8")
        checkpoints = sorted(output.glob("checkpoint_*"))
        protected = {checkpoint.name, (output / "best.txt").read_text().strip() if (output / "best.txt").is_file() else ""}
        removable = [path for path in checkpoints[:-2] if path.name not in protected]
        for path in removable:
            shutil.rmtree(path)
    if world > 1:
        torch.distributed.barrier()
    return checkpoint


@torch.no_grad()
def development_evaluation(raw_model, head, dataset: Path, rank: int, world: int, device, limit: int = 128):
    from v1.llada.model.modeling_llada import ActivationCheckpointingStrategy
    from .model import load_tokenizer

    rows = fixed_split(load_gsm8k(dataset), "dev")[:limit]
    tokenizer = load_tokenizer()
    raw_model.model.transformer.ff_out = head
    raw_model.model.set_activation_checkpointing(None)
    raw_model.eval()
    correct = total = truncated = 0
    elapsed = nfe = 0.0
    try:
        for index in range(rank, len(rows), world):
            row = rows[index]
            ids = prompt_ids(tokenizer, row["question"], "gsm8k")
            result = generate_fixed_quota(raw_model, torch.tensor([ids], device=device), steps_per_block=16)
            generated = result.output[0, len(ids):].tolist()
            eos = generated.index(126081) if 126081 in generated else None
            text = tokenizer.decode(generated[:eos] if eos is not None else generated, skip_special_tokens=True)
            correct += int(extract_answer(text) == extract_answer(row["answer"], gold=True))
            truncated += int(eos is None)
            elapsed += result.seconds
            nfe += result.nfe
            total += 1
        values = torch.tensor([correct, total, truncated, elapsed, nfe], dtype=torch.float64, device=device)
        if world > 1:
            torch.distributed.all_reduce(values)
        return {
            "accuracy": float(values[0] / values[1]), "examples": int(values[1]),
            "truncation_rate": float(values[2] / values[1]), "mean_seconds": float(values[3] / values[1]),
            "mean_nfe": float(values[4] / values[1]),
        }
    finally:
        raw_model.model.transformer.ff_out = torch.nn.Identity()
        raw_model.model.set_activation_checkpointing(ActivationCheckpointingStrategy.whole_layer)
        raw_model.train()


def train(
    prepared: Path,
    supervision: Path,
    output: Path,
    *,
    stage: str,
    resume: Path | None = None,
    init_adapter: Path | None = None,
    dev_dataset: Path | None = None,
    smoke_updates: int | None = None,
    overfit_records: int | None = None,
    config: ExperimentConfig = ExperimentConfig(),
):
    if stage not in {"a", "b"}:
        raise ValueError("stage must be a or b")
    rank, world, local = distributed()
    device = torch.device("cuda", local)
    torch.cuda.set_device(device)
    random.seed(SEED + rank)
    torch.manual_seed(SEED + rank)
    model = load_model(device, training=True)
    names = inject_lora(model, config.rank, config.alpha, config.dropout)
    if init_adapter:
        init_adapter = Path(init_adapter)
        load_adapter(model, init_adapter / "adapter.pt" if init_adapter.is_dir() else init_adapter)
    head = detach_output_head(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[local], output_device=local, broadcast_buffers=False, find_unused_parameters=False
    ) if world > 1 else model
    raw_model = model.module if hasattr(model, "module") else model
    lr = config.stage_a_lr if stage == "a" else config.stage_b_lr
    optimizer = _optimizer(raw_model, lr, config.weight_decay, world)
    dataset = AlignedShardDataset(prepared, supervision, "train")
    p_manifest = read_json(Path(prepared) / "manifest.json")
    total_records = p_manifest["train_size"] if stage == "a" else p_manifest["real_transition_records"]
    global_batch = world * config.grad_accumulation
    total_updates = math.ceil(total_records / global_batch)
    if smoke_updates is not None:
        total_updates = min(total_updates, smoke_updates)
    state = {
        "stage": stage, "update": 0, "micro_step": 0, "total_updates": total_updates,
        "best_accuracy": -1.0,
        "world_size": world, "config_hash": digest(config.to_dict()),
        "prepared_hash": p_manifest["data_hash"], "supervision_hash": read_json(Path(supervision) / "manifest.json")["collection_hash"],
    }
    if resume:
        resume = Path(resume)
        saved = json.loads((resume / "state.json").read_text())
        if saved["world_size"] != world or saved["stage"] != stage or saved["config_hash"] != state["config_hash"]:
            raise ValueError("Resume topology/stage/config mismatch")
        load_adapter(raw_model, resume / "adapter.pt")
        rank_state = torch.load(resume / f"rank_{rank:03d}.pt", map_location="cpu", weights_only=False)
        _load_local_optimizer_state(optimizer, rank_state["optimizer"])
        state = rank_state["state"]
        random.setstate(rank_state["python_rng"])
        torch.set_rng_state(rank_state["torch_rng"])
        if rank_state["cuda_rng"] is not None:
            torch.cuda.set_rng_state(rank_state["cuda_rng"])
        # A smoke resume may deliberately extend the requested update count.
        # Formal runs recompute the same full-epoch value here.
        state["total_updates"] = total_updates
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(output / "config.json", {**config.to_dict(), "targets": names, "trainable": sum(p.numel() for p in raw_model.parameters() if p.requires_grad)})
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter(output / "tensorboard")
    else:
        writer = None
    if world > 1:
        torch.distributed.barrier()

    iterator = _iter_rank(dataset, stage, rank, world, overfit_records)
    local_skip = state["micro_step"] // world
    last = None
    for _ in range(local_skip):
        last = next(iterator)
    started = time.perf_counter()
    best_accuracy = float(state.get("best_accuracy", -1.0))
    optimizer.zero_grad(set_to_none=True)
    for update in range(state["update"], total_updates):
        sums: dict[str, float] = {key: 0.0 for key in METRIC_KEYS}
        supervised = 0
        for micro in range(config.grad_accumulation):
            try:
                pair = next(iterator)
                last = pair
                active = True
            except StopIteration:
                if last is None:
                    raise RuntimeError("Rank received no training rows")
                pair, active = last, False
            if world > 1:
                active_tensor = torch.tensor(float(active), device=device)
                torch.distributed.all_reduce(active_tensor)
                active_count = float(active_tensor)
            else:
                active_count = float(active)
            prepared_row, supervision_row = pair
            sync = contextlib.nullcontext() if micro == config.grad_accumulation - 1 or world == 1 else model.no_sync()
            with sync, torch.autocast("cuda", dtype=torch.bfloat16):
                if prepared_row["kind"] == "retention" and stage == "a":
                    loss, metrics = retention_loss(model, head, prepared_row, supervision_row)
                else:
                    loss, metrics = acceleration_loss(model, head, prepared_row, supervision_row, config)
                if not active:
                    loss = loss * 0.0
                elif 0 < active_count < world:
                    loss = loss * (world / active_count)
                (loss / config.grad_accumulation).backward()
            for key, value in metrics.items():
                sums[key] = sums.get(key, 0.0) + float(value) * active
            supervised += int(active)
            state["micro_step"] += world
        grad_norm = torch.nn.utils.clip_grad_norm_(list(raw_model.parameters()), config.grad_clip)
        factor = schedule_factor(update, total_updates, config.warmup_ratio)
        for group in optimizer.param_groups:
            group["lr"] = lr * factor
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        state["update"] = update + 1
        elapsed = time.perf_counter() - started
        if world > 1:
            packed = torch.tensor([sums[key] for key in METRIC_KEYS] + [supervised], dtype=torch.float64, device=device)
            torch.distributed.all_reduce(packed)
            sums = {key: float(packed[index]) for index, key in enumerate(METRIC_KEYS)}
            supervised = int(packed[-1])
        if writer and (update < 10 or (update + 1) % 10 == 0):
            for key, value in sums.items():
                writer.add_scalar(f"loss/{key}", value / max(1, supervised), update + 1)
            writer.add_scalar("train/grad_norm", float(grad_norm), update + 1)
            writer.add_scalar("train/lr", lr * factor, update + 1)
            writer.add_scalar("train/examples_per_second", (update + 1 - state.get("resume_update", 0)) * global_batch / elapsed, update + 1)
            writer.add_scalar("train/peak_gib", torch.cuda.max_memory_allocated(device) / 2**30, update + 1)
            writer.add_scalar("train/eta_hours", elapsed / (update + 1) * (total_updates - update - 1) / 3600, update + 1)
            writer.flush()
        interval = 25_000 if stage == "a" else 5_000
        if smoke_updates is not None:
            interval = max(1, smoke_updates)
        milestones = {math.ceil(total_updates * fraction) for fraction in (.25, .5, .75, 1.0)}
        evaluate_now = dev_dataset is not None and update + 1 in milestones and smoke_updates is None
        development = None
        if evaluate_now:
            development = development_evaluation(raw_model, head, dev_dataset, rank, world, device)
            if rank == 0:
                atomic_json(output / f"dev_{update+1:08d}.json", development)
                writer.add_scalar("dev/accuracy_16", development["accuracy"], update + 1)
                writer.add_scalar("dev/truncation_rate_16", development["truncation_rate"], update + 1)
        is_best = development is not None and development["accuracy"] > best_accuracy
        if development is not None:
            best_accuracy = max(best_accuracy, development["accuracy"])
            state["best_accuracy"] = best_accuracy
        if (update + 1) % interval == 0 or update + 1 == total_updates or evaluate_now:
            save_checkpoint(output, model, optimizer, state, rank, world, best=is_best)
        if rank == 0 and (update < 10 or (update + 1) % 50 == 0):
            print(f"stage={stage} update={update+1}/{total_updates} lr={lr*factor:.3g} peak={torch.cuda.max_memory_allocated(device)/2**30:.2f}GiB", flush=True)
    if writer:
        writer.close()
    return state

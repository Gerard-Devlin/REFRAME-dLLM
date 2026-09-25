from __future__ import annotations

import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from .core import BLOCK_LENGTH, EOS_ID, MASK_ID, SEED, atomic_json, digest, read_json, stable_u64
from .data import ShardWriter, build_acceleration_state, deterministic_state, load_shard
from .model import detach_output_head, hidden_states, load_model, topk_distribution


def distributed() -> tuple[int, int, int]:
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if world > 1 and not torch.distributed.is_initialized():
        torch.cuda.set_device(local)
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local))
    return rank, world, local


def _forward_positions(model, head, ids: list[int], positions: list[int]):
    source = torch.tensor([ids], dtype=torch.long, device=next(model.parameters()).device)
    hidden = hidden_states(model, source)[0].index_select(0, torch.tensor(positions, device=source.device))
    logits = F.linear(hidden, head.weight).float()
    probabilities = F.softmax(logits, dim=-1)
    return hidden, logits, probabilities


def _anchors(hidden, head, positions: list[int], excluded: set[int], limit: int = 8) -> list[dict]:
    candidates = [index for index in range(len(positions)) if index not in excluded]
    if not candidates:
        return []
    if len(candidates) > limit:
        chosen = torch.linspace(0, len(candidates) - 1, limit).round().long().tolist()
        candidates = [candidates[index] for index in chosen]
    selected = hidden.index_select(0, torch.tensor(candidates, device=hidden.device))
    ids, log_probs, residual = topk_distribution(selected, head, k=32)
    return [
        {
            "offset": int(positions[row]),
            "ids": ids[index].cpu().tolist(),
            "log_probs": log_probs[index].cpu().tolist(),
            "residual_log_prob": float(residual[index].cpu()),
        }
        for index, row in enumerate(candidates)
    ]


@torch.no_grad()
def supervise_acceleration(model, head, row: dict, *, attempts: int = 8) -> dict:
    special = {MASK_ID, EOS_ID}
    for attempt in range(attempts):
        block, reveal = deterministic_state(row, attempt)
        ids, reference, absolute = build_acceleration_state(row, block, reveal)
        offsets = [offset for offset in range(BLOCK_LENGTH) if not reveal & (1 << offset)]
        positions = [absolute + offset for offset in offsets]
        started = time.perf_counter()
        hidden, _, probabilities = _forward_positions(model, head, ids, positions)
        target_ids = torch.tensor([reference[block * BLOCK_LENGTH + offset] for offset in offsets], device=probabilities.device)
        top = probabilities.argmax(-1)
        confidence = probabilities.gather(1, target_ids[:, None]).squeeze(1)
        eligible = [i for i, token in enumerate(target_ids.tolist()) if top[i].item() == token and token not in special]
        if not eligible:
            continue
        first = max(eligible, key=lambda i: float(confidence[i]))
        if row.get("real_transition"):
            ids_second = list(ids)
            ids_second[positions[first]] = int(target_ids[first])
            remaining = [i for i in range(len(offsets)) if i != first]
            if not remaining:
                continue
            second_positions = [positions[i] for i in remaining]
            second_hidden, _, second_probabilities = _forward_positions(model, head, ids_second, second_positions)
            second_targets = torch.tensor([int(target_ids[i]) for i in remaining], device=probabilities.device)
            second_top = second_probabilities.argmax(-1)
            second_confidence = second_probabilities.gather(1, second_targets[:, None]).squeeze(1)
            eligible_second = [
                j for j, token in enumerate(second_targets.tolist())
                if second_top[j].item() == token and token not in special
            ]
            if not eligible_second:
                continue
            second_local = max(eligible_second, key=lambda j: float(second_confidence[j]))
            second = remaining[second_local]
            second_value = float(second_confidence[second_local])
        else:
            eligible_second = [i for i in eligible if i != first]
            if not eligible_second:
                continue
            second = max(eligible_second, key=lambda i: float(confidence[i]))
            second_value = float(confidence[second])
        targets = [
            {"offset": offsets[first], "token": int(target_ids[first]), "confidence": float(confidence[first]), "round": 1},
            {"offset": offsets[second], "token": int(target_ids[second]), "confidence": second_value, "round": 2},
        ]
        anchor_rows = _anchors(hidden, head, offsets, {first, second})
        return {
            "sample_id": row["sample_id"], "kind": "acceleration", "valid": True,
            "real_transition": bool(row.get("real_transition")), "attempt": attempt,
            "block": block, "reveal_mask": reveal, "targets": targets, "anchors": anchor_rows,
            "teacher_seconds": time.perf_counter() - started,
        }
    return {
        "sample_id": row["sample_id"], "kind": "acceleration", "valid": False,
        "real_transition": bool(row.get("real_transition")), "reason": "no_two_teacher_correct_targets",
    }


def collect(prepared: Path, output: Path, split: str = "all", attempts: int = 8) -> dict:
    prepared, output = Path(prepared), Path(output)
    manifest = read_json(prepared / "manifest.json")
    rank, world, local = distributed()
    device = torch.device("cuda", local) if torch.cuda.is_available() else torch.device("cpu")
    model = load_model(device)
    head = detach_output_head(model)
    splits = ("train", "validation") if split == "all" else (split,)
    stats = {"records": 0, "valid": 0, "invalid": 0, "retention": 0, "real": 0, "teacher_seconds": 0.0}
    for split_name in splits:
        shards = manifest["shards"][split_name]
        for shard_index in range(rank, len(shards), world):
            source_info = shards[shard_index]
            records = load_shard(prepared / source_info["file"])
            target = output / source_info["file"]
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.is_file():
                payload = torch.load(target, map_location="cpu", weights_only=False)
                if len(payload.get("records", [])) == len(records):
                    for item in payload["records"]:
                        stats["records"] += 1
                        stats["valid" if item.get("valid") else "invalid"] += 1
                        stats["retention"] += int(item.get("kind") == "retention")
                        stats["real"] += int(item.get("real_transition", False))
                        stats["teacher_seconds"] += item.get("teacher_seconds", 0.0)
                    print(f"rank={rank} skip complete {target}", flush=True)
                    continue
            supervised = []
            for row in records:
                if row["kind"] == "retention":
                    result = {
                        "sample_id": row["sample_id"], "kind": "retention", "valid": True,
                        "mask_seed": stable_u64("retention", SEED, row["sample_id"]),
                    }
                    stats["retention"] += 1
                else:
                    result = supervise_acceleration(model, head, row, attempts=attempts)
                    stats["real"] += int(row.get("real_transition", False))
                    stats["teacher_seconds"] += result.get("teacher_seconds", 0.0)
                stats["records"] += 1
                stats["valid" if result["valid"] else "invalid"] += 1
                supervised.append(result)
            tmp = target.with_suffix(".tmp")
            torch.save({"format_version": 1, "records": supervised}, tmp)
            os.replace(tmp, target)
            print(f"rank={rank} {split_name} {shard_index + 1}/{len(shards)} valid={stats['valid']}", flush=True)
    if world > 1:
        values = torch.tensor(list(stats.values()), dtype=torch.float64, device=device)
        torch.distributed.all_reduce(values)
        stats = {key: (int(value.item()) if key != "teacher_seconds" else value.item()) for key, value in zip(stats, values)}
        torch.distributed.barrier()
    if rank == 0:
        expected = [output / item["file"] for name in splits for item in manifest["shards"][name]]
        missing = [str(path) for path in expected if not path.is_file()]
        if missing:
            raise RuntimeError(f"Missing collected shards: {missing[:3]}")
        output_manifest = {
            "format_version": 1, "prepared_hash": manifest["data_hash"], "split": split,
            "shards": {name: manifest["shards"][name] for name in splits}, "stats": stats, "world_size": world,
        }
        output_manifest["collection_hash"] = digest(output_manifest)
        atomic_json(output / "manifest.json", output_manifest)
    return stats

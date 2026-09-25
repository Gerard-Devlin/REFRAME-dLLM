from __future__ import annotations

import json
import os
from pathlib import Path
import statistics
import time

import torch

from fastv_dllm.common import extract_answer, prompt_ids
from .collect import distributed
from .core import EOS_ID, REVISION, atomic_json, bootstrap_interval, file_sha256, stable_u64
from .decode import generate_fixed_quota
from .model import inject_lora, load_adapter, load_model, load_tokenizer


def load_gsm8k(path: Path) -> list[dict]:
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    if not all({"question", "answer"}.issubset(row) for row in rows):
        raise ValueError("GSM8K JSON rows require question and answer")
    normalized = []
    for index, row in enumerate(rows):
        normalized.append({**row, "id": str(row.get("id", f"test:{index}")), "source_index": index})
    return normalized


def fixed_split(rows: list[dict], split: str) -> list[dict]:
    ordered = sorted(rows, key=lambda row: (stable_u64("gsm8k-split", row["id"]), row["id"]))
    dev_ids = {row["id"] for row in ordered[:256]}
    if split == "dev":
        return [row for row in rows if row["id"] in dev_ids]
    if split == "holdout":
        return [row for row in rows if row["id"] not in dev_ids]
    if split == "full":
        return rows
    raise ValueError(split)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round((len(ordered) - 1) * q))]


def evaluate(
    dataset: Path,
    output: Path,
    *,
    split: str = "dev",
    steps: tuple[int, ...] = (8, 16, 32),
    adapter: Path | None = None,
    limit: int | None = None,
):
    rank, world, local = distributed()
    device = torch.device("cuda", local)
    model = load_model(device)
    tokenizer = load_tokenizer()
    model_name = "teacher"
    if adapter:
        inject_lora(model)
        load_adapter(model, Path(adapter) / "adapter.pt" if Path(adapter).is_dir() else adapter)
        model_name = Path(adapter).name
    rows = fixed_split(load_gsm8k(dataset), split)
    if limit:
        rows = rows[:limit]
    if world > len(rows):
        raise ValueError("More GPUs than evaluation prompts")
    output = Path(output)
    if rank == 0:
        output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()
    warmup = torch.tensor([prompt_ids(tokenizer, "What is 1+1?", "gsm8k")], device=device)
    generate_fixed_quota(model, warmup, steps_per_block=min(8, max(1, steps[0])), gen_length=32)
    fixed = torch.cat((warmup, torch.full((1, 512), 126336, dtype=torch.long, device=device)), dim=1)
    forward_times = []
    for _ in range(5):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        with torch.no_grad():
            logits = model(fixed, use_cache=False).logits
        torch.cuda.synchronize(device)
        forward_times.append(time.perf_counter() - started)
        del logits
    atomic_json(output / f"rank_{rank:03d}_forward.json", {"seconds": forward_times, "sequence_length": fixed.shape[1]})
    path = output / f"rank_{rank:03d}.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index in range(rank, len(rows), world):
            row = rows[index]
            ids = prompt_ids(tokenizer, row["question"], "gsm8k")
            prompt = torch.tensor([ids], dtype=torch.long, device=device)
            gold = extract_answer(row["answer"], gold=True)
            record = {"id": row["id"], "index": index, "gold": gold, "methods": {}}
            for count in steps:
                result = generate_fixed_quota(model, prompt, steps_per_block=count)
                generated = result.output[0, len(ids) :].tolist()
                eos = generated.index(EOS_ID) if EOS_ID in generated else None
                decoded = tokenizer.decode(generated[:eos] if eos is not None else generated, skip_special_tokens=True)
                answer = extract_answer(decoded)
                record["methods"][str(count)] = {
                    "answer": answer, "correct": answer == gold, "seconds": result.seconds,
                    "nfe": result.nfe, "calls_per_block": result.calls_per_block,
                    "eos": eos is not None, "truncated": eos is None, "tokens": eos if eos is not None else len(generated),
                    "peak_gib": result.peak_gib, "text": decoded,
                }
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"rank={rank} evaluated {index+1}/{len(rows)}", flush=True)
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        records = []
        for rank_id in range(world):
            records.extend(json.loads(line) for line in (output / f"rank_{rank_id:03d}.jsonl").read_text(encoding="utf-8").splitlines())
        records.sort(key=lambda row: row["index"])
        if len(records) != len(rows) or len({row["id"] for row in records}) != len(rows):
            raise AssertionError("Multi-GPU evaluation lost or duplicated prompts")
        summary = {}
        for count in steps:
            values = [row["methods"][str(count)] for row in records]
            seconds = [value["seconds"] for value in values]
            summary[str(count)] = {
                "examples": len(values), "accuracy": sum(value["correct"] for value in values) / len(values),
                "total_nfe": sum(value["nfe"] for value in values), "mean_nfe": statistics.mean(value["nfe"] for value in values),
                "mean_seconds": statistics.mean(seconds), "p50_seconds": _percentile(seconds, .5), "p95_seconds": _percentile(seconds, .95),
                "mean_tokens": statistics.mean(value["tokens"] for value in values),
                "eos_rate": sum(value["eos"] for value in values) / len(values),
                "truncation_rate": sum(value["truncated"] for value in values) / len(values),
            }
        report = {
            "model": model_name, "revision": REVISION, "split": split, "dataset_sha256": file_sha256(dataset),
            "world_size": world, "steps_per_block": list(steps), "summary": summary,
        }
        fixed_times = []
        for rank_id in range(world):
            fixed_times.extend(json.loads((output / f"rank_{rank_id:03d}_forward.json").read_text())["seconds"])
        report["fixed_workload_forward"] = {
            "sequence_length": fixed.shape[1], "mean_seconds": statistics.mean(fixed_times),
            "p50_seconds": _percentile(fixed_times, .5), "p95_seconds": _percentile(fixed_times, .95),
        }
        if "16" in summary and "32" in summary:
            report["speedup_32_to_16"] = summary["32"]["mean_seconds"] / summary["16"]["mean_seconds"]
            report["accuracy_delta_16_minus_32"] = summary["16"]["accuracy"] - summary["32"]["accuracy"]
        atomic_json(output / "summary.json", report)
        return report


def compare_runs(teacher: Path, stage_a: Path, stage_b: Path, output: Path) -> dict:
    def records(root: Path):
        values = []
        for path in sorted(Path(root).glob("rank_*.jsonl")):
            values.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines())
        return {row["id"]: row for row in values}

    runs = {"teacher": records(teacher), "stage_a": records(stage_a), "stage_b": records(stage_b)}
    shared = sorted(set.intersection(*(set(value) for value in runs.values())))
    if not shared:
        raise ValueError("No shared evaluation ids")
    report = {"examples": len(shared), "comparisons": {}}
    teacher_correct = [float(runs["teacher"][key]["methods"]["32"]["correct"]) for key in shared]
    teacher_time = [runs["teacher"][key]["methods"]["32"]["seconds"] for key in shared]
    for name in ("stage_a", "stage_b"):
        student = [float(runs[name][key]["methods"]["16"]["correct"]) for key in shared]
        differences = [a - b for a, b in zip(student, teacher_correct)]
        elapsed = [runs[name][key]["methods"]["16"]["seconds"] for key in shared]
        report["comparisons"][name] = {
            "accuracy_delta": statistics.mean(differences), "paired_bootstrap_95": bootstrap_interval(differences),
            "speedup": sum(teacher_time) / sum(elapsed),
        }
    atomic_json(Path(output), report)
    return report

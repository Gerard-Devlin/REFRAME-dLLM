"""Paired native/runner-up experiments on the pinned 1.5B v2 checkpoint."""

import argparse
from dataclasses import asdict
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import time

import torch

from .decode import generate
from .presets import PRESETS


MODEL_ID = "Efficient-Large-Model/Fast_dLLM_v2_1.5B"
REVISION = "da5608172d2b74380e4e780baa19c71645e4f981"
MODEL_CODE_SHA256 = "d363ee4a4d4bf52958645d5c715712c5b027525bb90611a52178e58695e09b50"


def extract_answer(text, gold=False):
    if gold:
        text = text.split("####")[-1]
    else:
        explicit = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
        boxed = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", text)
        values = explicit or boxed or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not values:
            return None
        text = values[-1]
    try:
        return str(Decimal(text.strip().replace(",", "")).normalize())
    except InvalidOperation:
        return None


def prompt_ids(tokenizer, question):
    return tokenizer.apply_chat_template([
        {"role": "user", "content": question + "\nExplain your reasoning and end with #### followed by the final number."}
    ], tokenize=True, add_generation_prompt=True)


def model_snapshot():
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    code = path / "modeling.py"
    actual = hashlib.sha256(code.read_bytes()).hexdigest()
    if actual != MODEL_CODE_SHA256:
        raise RuntimeError(f"Pinned decoder source mismatch: {actual}; expected {MODEL_CODE_SHA256}")
    return path


def count_forwards(model, capture=False):
    """Count the same outer model forwards for both native and modified runs."""
    state = {"calls": 0, "inputs": []}
    original = model.forward

    def wrapped(*args, **kwargs):
        state["calls"] += 1
        if capture:
            ids = kwargs.get("input_ids", args[0] if args else None)
            state["inputs"].append((ids.detach().cpu().tolist(),
                                    kwargs.get("update_past_key_values"),
                                    kwargs.get("use_block_cache"),
                                    kwargs.get("replace_position")))
        return original(*args, **kwargs)

    model.forward = wrapped
    return state, original


def run(model, ids, args, policy, capture=False, collect=False):
    source = torch.tensor([ids], device=model.device)
    events = []
    state, original = count_forwards(model, capture=capture)
    try:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        options = dict(max_new_tokens=args.max_new_tokens, block_size=args.block_size,
                       small_block_size=args.small_block_size, threshold=args.threshold,
                       temperature=0, use_block_cache=args.use_block_cache)
        if policy == "threshold90":
            options["threshold"] = 0.90
        if policy in ("official", "threshold90"):
            output = model.generate(source, **options)
        else:
            preset = policy if policy in PRESETS else getattr(args, "preset", "original")
            margin, guard = PRESETS[preset]
            if preset == "original" and args.mode != "sweep":
                margin = args.margin
            output = generate(model, source, **options,
                              policy="budget" if policy in PRESETS else policy,
                              margin=margin, guard=guard,
                              observer=events.append if collect else None)
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        generated = output[0, len(ids):].tolist()
        return dict(tokens=generated, calls=state["calls"], inputs=state["inputs"],
                    seconds=seconds, peak_gib=torch.cuda.max_memory_allocated() / 2**30,
                    events=events)
    finally:
        model.forward = original


def parity_check(model, tokenizer, args, sample):
    ids = prompt_ids(tokenizer, sample["question"])
    official = run(model, ids, args, "official", capture=True)
    copied = run(model, ids, args, "native", capture=True)
    if (official["tokens"] != copied["tokens"] or official["calls"] != copied["calls"]
            or official["inputs"] != copied["inputs"]):
        raise AssertionError("Pinned official/native parity failed: stop before evaluating")
    return dict(tokens=len(official["tokens"]), calls=official["calls"],
                same_tokens=True, same_forward_inputs=True)


def observe_stats(events, original_tokens, prompt_length):
    extra = 0
    first_proposal = {}
    for step in events:
        for candidate in step["extras"]:
            extra += 1
            first_proposal.setdefault(candidate["position"], candidate["token"])
    comparable = match = 0
    for position, token in first_proposal.items():
        local = position - prompt_length
        if 0 <= local < len(original_tokens):
            comparable += 1
            match += original_tokens[local] == token
    return dict(steps=len(events), steps_with_extra=sum(len(e["proposed"]) > len(e["baseline"]) for e in events),
                baseline_commits=sum(len(e["baseline"]) for e in events),
                proposed_commits=sum(len(e["proposed"]) for e in events),
                plusplus_certified=sum(len(e["plusplus"]) for e in events),
                extra_proposals=extra, unique_extra=len(first_proposal),
                comparable_unique_extra=comparable, unique_extra_matching_native_final=match)


def method_names(mode):
    if mode == "observe":
        return ("observe",)
    if mode == "compare":
        return ("official", "budget")
    return ("official", "threshold90", *PRESETS)


def method_order(names, index, world):
    # Rotate within each GPU as well as across GPUs, even if world == len(names).
    offset = (index // world + index % world) % len(names)
    return names[offset:] + names[:offset]


def summarize(records, mode):
    result = {}
    names = method_names(mode)
    for name in names:
        rows = [record[name] for record in records]
        result[name] = dict(
            examples=len(rows), accuracy=sum(row["correct"] for row in rows) / len(rows),
            mean_nfe=sum(row["calls"] for row in rows) / len(rows),
            mean_seconds=sum(row["seconds"] for row in rows) / len(rows),
            truncation_rate=sum(row["length_capped"] for row in rows) / len(rows),
            mean_generated_tokens=sum(row["tokens"] for row in rows) / len(rows),
            total_nfe=sum(row["calls"] for row in rows), total_seconds=sum(row["seconds"] for row in rows),
        )
    if mode == "observe":
        counts = [r["opportunity"] for r in records]
        keys = counts[0].keys()
        result["opportunity"] = {key: sum(row[key] for row in counts) for key in keys}
        result["opportunity"]["step_hit_rate"] = (result["opportunity"]["steps_with_extra"] /
            result["opportunity"]["steps"] if result["opportunity"]["steps"] else 0.0)
        result["opportunity"]["future_token_agreement"] = (result["opportunity"]["unique_extra_matching_native_final"] /
            result["opportunity"]["comparable_unique_extra"] if result["opportunity"]["comparable_unique_extra"] else None)
    else:
        comparisons = {}
        for name in names[1:]:
            comparisons[name] = dict(
                nfe_ratio=result[name]["total_nfe"] / result["official"]["total_nfe"],
                aggregate_sample_latency_speedup=result["official"]["total_seconds"] / result[name]["total_seconds"],
                budget_minus_official_accuracy=result[name]["accuracy"] - result["official"]["accuracy"],
                correct_to_wrong=sum(r["official"]["correct"] and not r[name]["correct"] for r in records),
                wrong_to_correct=sum(not r["official"]["correct"] and r[name]["correct"] for r in records),
            )
        result["paired"] = comparisons["budget"] if mode == "compare" else comparisons
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True, help="Prepared GSM8K JSON list with id/question/answer")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("observe", "compare", "sweep"), default="observe")
    parser.add_argument("--preset", choices=tuple(PRESETS), default="original",
                        help="compare/observe policy; sweep runs every preset plus native threshold=0.90")
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.95)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--use-block-cache", action="store_true")
    args = parser.parse_args()
    if args.limit < 1 or not (0 <= args.threshold <= 1):
        parser.error("Positive limit and threshold in [0,1] required")
    if not 0 <= args.margin < 1:
        parser.error("margin must be in [0,1)")
    if (args.mode == "sweep" or args.preset != "original") and args.margin != 0:
        parser.error("Fixed presets define their own margin; omit --margin")
    samples = json.loads(args.dataset.read_text(encoding="utf-8"))
    if args.limit > len(samples):
        parser.error(f"Requested {args.limit} samples, only {len(samples)} available")
    samples = samples[:args.limit]
    if args.mode == "sweep" and not all(str(s["id"]).startswith("train:") for s in samples):
        parser.error("Tune the sweep on train-split development questions, then use compare on held-out test")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world > len(samples):
        parser.error("Use at most one GPU per example")
    torch.cuda.set_device(local_rank)
    if world > 1:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            parser.error("Output directory must be new or empty")
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    root = model_snapshot()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True,
        local_files_only=True, dtype=torch.bfloat16).to(f"cuda:{local_rank}").eval()
    parity_args = argparse.Namespace(**{**vars(args), "max_new_tokens": 2 * args.block_size})
    parity = parity_check(model, tokenizer, parity_args, samples[rank % len(samples)])
    print(f"rank={rank} pinned native parity {parity}", flush=True)

    # Identical unrelated warmup, excluded from both measured methods.
    warmup = prompt_ids(tokenizer, "What is one plus one?")
    names = method_names(args.mode)
    for name in names:
        run(model, warmup, args, name)
    path = args.output / f"rank_{rank}.jsonl"
    for index in range(rank, len(samples), world):
        sample = samples[index]
        ids = prompt_ids(tokenizer, sample["question"])
        target = extract_answer(sample["answer"], gold=True)
        record = dict(index=index, id=sample["id"], target=target)
        if args.mode == "observe":
            measured = run(model, ids, args, "observe", collect=True)
            record["opportunity"] = observe_stats(measured["events"], measured["tokens"], len(ids))
            record["events"] = measured.pop("events")
            methods = (("observe", measured),)
        else:
            order = method_order(names, index, world)
            methods = tuple((name, run(model, ids, args, name)) for name in order)
        for name, outcome in methods:
            output_text = tokenizer.decode(outcome["tokens"], skip_special_tokens=True)
            prediction = extract_answer(output_text)
            record[name] = dict(text=output_text, prediction=prediction,
                correct=prediction is not None and prediction == target,
                length_capped=len(outcome["tokens"]) >= args.max_new_tokens and 151645 not in outcome["tokens"],
                tokens=len(outcome["tokens"]), calls=outcome["calls"],
                seconds=outcome["seconds"], peak_gib=outcome["peak_gib"])
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"rank={rank} {index + 1}/{len(samples)} id={sample['id']} "
              f"calls=" + ",".join(f"{name}:{record[name]['calls']}" for name, _ in methods), flush=True)
    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        records = []
        for worker in range(world):
            records.extend(json.loads(line) for line in (args.output / f"rank_{worker}.jsonl").read_text(encoding="utf-8").splitlines())
        records.sort(key=lambda row: row["index"])
        if [row["index"] for row in records] != list(range(len(samples))):
            raise AssertionError("Distributed evaluation missed or duplicated examples")
        summary = dict(mode=args.mode, model=MODEL_ID, revision=REVISION, dataset=str(args.dataset),
            dataset_sha256=hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
            implementation_sha256={name: hashlib.sha256((Path(__file__).parent / name).read_bytes()).hexdigest()
                for name in ("budget.py", "decode.py", "evaluate.py", "presets.py")},
            ids=[s["id"] for s in samples], threshold=args.threshold, margin=args.margin,
            preset=args.preset,
            policy_configs={name: dict(margin=margin, guard=asdict(guard) if guard is not None else None)
                            for name, (margin, guard) in PRESETS.items()},
            threshold90_reference=0.90 if args.mode == "sweep" else None,
            max_new_tokens=args.max_new_tokens, block_size=args.block_size,
            small_block_size=args.small_block_size, block_cache=args.use_block_cache,
            gpus=world, parity=parity, results=summarize(records, args.mode),
            scope="Exploratory fixed GSM8K prompts. Certificate assumes compatible joint marginals; no lossless claim.")
        (args.output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(summary["results"], ensure_ascii=False, indent=2), flush=True)
        from .report import render
        print(render(summary), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

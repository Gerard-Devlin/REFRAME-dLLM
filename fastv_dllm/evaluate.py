"""Audit, probe, smoke and paired evaluation entry point."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from .backend import AttentionBackend
from .common import EOS_ID, MODEL_ID, REVISION, extract_answer, load_samples, prompt_ids, sha256, snapshot, write_json
from .decode import generate
from .pruning import BlockForward, FastVConfig, add_stability
from .report import summarize


METHODS = ("sdpa_native", "flash_native", "sdpa_fastv", "flash_fastv", "sdpa_cache", "flash_cache")


def load_model(device):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    root = snapshot()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        root, trust_remote_code=True, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    return model, tokenizer


def options(args):
    return dict(
        max_new_tokens=args.max_new_tokens,
        block_size=args.block_size,
        small_block_size=args.small_block_size,
        threshold=args.threshold,
        temperature=0,
    )


def run_method(model, ids, args, method, probe=False):
    backend_name = "flash" if method.startswith("flash") else "sdpa"
    use_cache = method.endswith("cache")
    use_fastv = method.endswith("fastv")
    block_forward = None
    if use_fastv or probe:
        observed = args.observe_layers if probe else ()
        block_forward = BlockForward(
            model,
            FastVConfig(args.prune_after_layer, args.support_keep, args.block_size, args.small_block_size),
            observe_layers=observed,
            observe_keeps=args.observe_keeps,
        )
    source = torch.tensor([ids], device=model.device)
    with AttentionBackend(backend_name, block_size=args.block_size, verify_masks=True) as backend:
        result = generate(
            model, source, **options(args), use_block_cache=use_cache,
            block_forward=block_forward, prune=use_fastv,
        )
    records = [] if block_forward is None else add_stability(block_forward.records, args.observe_keeps)
    tokens = result.output[0, len(ids):].tolist()
    return dict(
        token_ids=tokens,
        seconds=result.seconds,
        peak_gib=result.peak_gib,
        logical_forwards=result.logical_forwards,
        ordinary_denoise=result.ordinary_denoise,
        cache_writes=result.cache_writes,
        prefill=result.prefill,
        backend=backend.report(),
        probe_records=records,
        deep_tokens=[row["deep_tokens"] for row in records] if use_fastv else [],
    )


def official_run(model, ids, args):
    source = torch.tensor([ids], device=model.device)
    return model.generate(source, **options(args), use_block_cache=False)


def capture_forwards(model, callback):
    """Capture outer-call semantics for the engineering parity gate."""
    calls = []
    original = model.forward

    def wrapped(*positional, **kwargs):
        ids = kwargs.get("input_ids", positional[0] if positional else None)
        calls.append(dict(
            input_ids=ids.detach().cpu().tolist(),
            update=bool(kwargs.get("update_past_key_values", False)),
            block_cache=bool(kwargs.get("use_block_cache", False)),
            replace=kwargs.get("replace_position"),
        ))
        return original(*positional, **kwargs)

    model.forward = wrapped
    try:
        value = callback()
    finally:
        model.forward = original
    return value, calls


def audit(model, tokenizer, sample, args):
    ids = prompt_ids(tokenizer, sample["question"])
    original_max = args.max_new_tokens
    args.max_new_tokens = min(original_max, 64)
    official, official_calls = capture_forwards(model, lambda: official_run(model, ids, args))
    native, native_calls = capture_forwards(model, lambda: run_method(model, ids, args, "sdpa_native"))
    official_suffix = official[0, len(ids):].tolist()
    if native["token_ids"] != official_suffix:
        raise AssertionError("Copied native decoder differs from pinned official decoder")
    if native_calls != official_calls:
        raise AssertionError("Copied native decoder changed outer forward inputs or cache flags")

    # Keep every support token.  This executes the custom layer loop and token-
    # shift gather without changing the mathematical sequence.
    previous_keep = args.support_keep
    args.support_keep = args.block_size
    all_support = run_method(model, ids, args, "sdpa_fastv")
    args.support_keep = previous_keep
    if all_support["token_ids"] != official_suffix:
        raise AssertionError("All-support custom forward changes native actions")
    flash = run_method(model, ids, args, "flash_native")
    if flash["backend"]["flash_calls"] <= 0:
        raise AssertionError("FlashAttention was requested but no flash kernel call was recorded")
    args.max_new_tokens = original_max
    return dict(
        copied_native_exact=True,
        copied_forward_sequence_exact=True,
        all_support_exact=True,
        official_tokens=len(official_suffix),
        native_nfe=native["logical_forwards"],
        flash_backend=flash["backend"],
        flash_same_tokens=flash["token_ids"] == official_suffix,
        flash_first_difference=next((i for i, (a, b) in enumerate(zip(flash["token_ids"], official_suffix)) if a != b), None),
        scope="Engineering audit on one prompt; Flash BF16 kernels may change rounding and are evaluated as a separate baseline.",
    )


def aggregate_probe(rows, keeps):
    grouped = {}
    for row in rows:
        for call in row["probe_records"]:
            for layer in call["layers"]:
                item = grouped.setdefault(str(layer["layer"]), dict(calls=0, score_seconds=[], masses={}, jaccard={}))
                item["calls"] += 1
                item["score_seconds"].append(layer["scoring_seconds"])
                for keep in keeps:
                    key = str(keep)
                    item["masses"].setdefault(key, []).append(layer["retained_support_mass"][key])
                    value = layer.get(f"jaccard_prev_top{keep}")
                    if value is not None:
                        item["jaccard"].setdefault(key, []).append(value)
    result = {}
    for layer, item in grouped.items():
        result[layer] = dict(
            calls=item["calls"],
            mean_score_seconds=sum(item["score_seconds"]) / len(item["score_seconds"]),
            retained_support_mass={key: sum(values) / len(values) for key, values in item["masses"].items()},
            mean_jaccard_previous={key: (sum(values) / len(values) if values else None)
                                   for key, values in item["jaccard"].items()},
        )
    return result


def method_order(index, world):
    offset = (index // world + index % world) % len(METHODS)
    return METHODS[offset:] + METHODS[:offset]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "probe", "smoke", "evaluate"), required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--small-block-size", type=int, default=8)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--prune-after-layer", type=int, default=4)
    parser.add_argument("--support-keep", type=int, default=8)
    parser.add_argument("--observe-layers", type=lambda x: tuple(int(v) for v in x.split(",")), default=(2, 4, 8, 12))
    parser.add_argument("--observe-keeps", type=lambda x: tuple(int(v) for v in x.split(",")), default=(4, 8, 12, 16))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.stage == "audit":
        args.limit = 1
    elif args.stage == "smoke":
        args.limit = min(args.limit, 2)
    samples = load_samples(args.dataset, args.limit)
    rank = int(os.environ.get("RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if world > len(samples):
        raise ValueError("Use no more GPUs than prompts")
    torch.cuda.set_device(local_rank)
    if world > 1:
        torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local_rank))
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory must be new or empty")
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1:
        torch.distributed.barrier()
    model, tokenizer = load_model(f"cuda:{local_rank}")

    if args.stage == "audit":
        report = audit(model, tokenizer, samples[0], args)
        if rank == 0:
            write_json(args.output / "audit.json", report)
            print(json.dumps(report, indent=2), flush=True)
        if world > 1:
            torch.distributed.destroy_process_group()
        return

    # One unrelated warmup per execution path; all warmups are excluded.
    warmup = prompt_ids(tokenizer, "What is one plus one?")
    if args.stage == "probe":
        run_method(model, warmup, args, "sdpa_native", probe=True)
    else:
        for method in METHODS:
            run_method(model, warmup, args, method)

    output_path = args.output / f"rank_{rank}.jsonl"
    for index in range(rank, len(samples), world):
        sample = samples[index]
        ids = prompt_ids(tokenizer, sample["question"])
        target = extract_answer(sample["answer"], gold=True)
        record = dict(index=index, id=sample["id"], target=target)
        if args.stage == "probe":
            outcome = run_method(model, ids, args, "sdpa_native", probe=True)
            record["probe_records"] = outcome.pop("probe_records")
            text = tokenizer.decode(outcome["token_ids"], skip_special_tokens=True)
            record["native"] = dict(text=text, prediction=extract_answer(text), seconds=outcome["seconds"],
                                    logical_forwards=outcome["logical_forwards"])
        else:
            for method in method_order(index, world):
                outcome = run_method(model, ids, args, method)
                token_ids = outcome.pop("token_ids")
                text = tokenizer.decode(token_ids, skip_special_tokens=True)
                prediction = extract_answer(text)
                outcome.pop("probe_records", None)
                record[method] = dict(
                    **outcome, text=text, prediction=prediction,
                    correct=prediction is not None and prediction == target,
                    tokens=len(token_ids),
                    length_capped=len(token_ids) >= args.max_new_tokens and EOS_ID not in token_ids,
                )
        with output_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"rank={rank} completed {index + 1}/{len(samples)} id={sample['id']}", flush=True)

    if world > 1:
        torch.distributed.barrier()
    if rank == 0:
        records = []
        for worker in range(world):
            records.extend(json.loads(line) for line in (args.output / f"rank_{worker}.jsonl").read_text(encoding="utf-8").splitlines())
        records.sort(key=lambda row: row["index"])
        if [row["index"] for row in records] != list(range(len(samples))):
            raise AssertionError("Distributed evaluation missed or duplicated prompts")
        if args.stage == "probe":
            results = dict(probe=aggregate_probe(records, args.observe_keeps))
        else:
            results = summarize(records, METHODS)
            if results["flash_native"]["flash_calls"] <= 0 or results["flash_fastv"]["flash_calls"] <= 0:
                raise AssertionError("Flash variants did not execute FlashAttention")
        report = dict(
            stage=args.stage, model=MODEL_ID, revision=REVISION,
            dataset=str(args.dataset), dataset_sha256=sha256(args.dataset),
            ids=[row["id"] for row in samples], world_size=world,
            configuration=dict(max_new_tokens=args.max_new_tokens, block_size=args.block_size,
                               small_block_size=args.small_block_size, threshold=args.threshold,
                               prune_after_layer=args.prune_after_layer, support_keep=args.support_keep,
                               observe_layers=args.observe_layers, observe_keeps=args.observe_keeps),
            implementation={name: sha256(Path(__file__).parent / name) for name in (
                "backend.py", "common.py", "decode.py", "evaluate.py", "pruning.py", "report.py")},
            results=results,
            scope=("Training-free development experiment. Method speedup is Flash native / Flash FastV; "
                   "Flash engineering speedup is SDPA native / Flash native. Official cache remains a baseline."),
        )
        write_json(args.output / "summary.json", report)
        print(json.dumps(results, ensure_ascii=False, indent=2), flush=True)
    if world > 1:
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

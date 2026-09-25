"""Paired FastV evaluation on the original LLaDA-8B-Instruct checkpoint."""

import argparse
import json
import os
from pathlib import Path

import torch

from .llada_backend import LLaDAAttentionBackend
from .llada_common import MODEL_ID, REVISION, extract_answer, load_samples, prompt_ids, sha256, snapshot, write_json
from .llada_decode import generate
from .llada_pruning import Config, LLaDABlockForward

METHODS = ("torch_native", "flash_native", "torch_fastv", "flash_fastv")


def load_model(device):
    from transformers import AutoConfig, AutoTokenizer
    from v1.llada.model.modeling_llada import LLaDAModelLM

    root = snapshot()
    config = AutoConfig.from_pretrained(root, local_files_only=True)
    config.flash_attention = True
    model = LLaDAModelLM.from_pretrained(
        root, config=config, local_files_only=True, torch_dtype=torch.bfloat16
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True)
    return model, tokenizer


def run_method(model, ids, args, method, probe=False):
    backend_name = "flash" if method.startswith("flash") else "torch"
    use_fastv = method.endswith("fastv")
    forward = None
    if use_fastv or probe:
        forward = LLaDABlockForward(model, Config(args.prune_after_layer, args.support_keep_ratio))
    source = torch.tensor([ids], device=model.device)
    with LLaDAAttentionBackend(model, backend_name) as backend:
        result = generate(
            model, source, gen_length=args.gen_length, block_length=args.block_length,
            threshold=args.threshold, block_forward=forward, prune=use_fastv,
        )
    tokens = result.output[0, len(ids):].tolist()
    return dict(token_ids=tokens, nfe=result.nfe, seconds=result.seconds, peak_gib=result.peak_gib,
                backend=backend.report(), records=[] if forward is None else forward.records)


def aggregate(records):
    output = {}
    for method in METHODS:
        rows = [row[method] for row in records]
        scored = [row["correct"] for row in rows if row["correct"] is not None]
        output[method] = dict(
            examples=len(rows), accuracy=(sum(scored) / len(scored) if scored else None),
            mean_seconds=sum(row["seconds"] for row in rows) / len(rows),
            total_seconds=sum(row["seconds"] for row in rows),
            mean_nfe=sum(row["nfe"] for row in rows) / len(rows),
            mean_tokens=sum(row["tokens"] for row in rows) / len(rows),
            flash_calls=sum(row["backend"]["flash_calls"] for row in rows),
            torch_sdpa_calls=sum(row["backend"]["torch_sdpa_calls"] for row in rows),
            mean_deep_tokens=(sum(v for row in rows for v in row.get("deep_tokens", [])) /
                              sum(len(row.get("deep_tokens", [])) for row in rows)
                              if any(row.get("deep_tokens") for row in rows) else None),
            mean_retained_mass=(sum(v for row in rows for v in row.get("retained_mass", [])) /
                                sum(len(row.get("retained_mass", [])) for row in rows)
                                if any(row.get("retained_mass") for row in rows) else None),
        )
    output["attribution"] = dict(
        flash_engineering_speedup=output["torch_native"]["total_seconds"] / output["flash_native"]["total_seconds"],
        fastv_speedup_same_torch=output["torch_native"]["total_seconds"] / output["torch_fastv"]["total_seconds"],
        fastv_speedup_same_flash=output["flash_native"]["total_seconds"] / output["flash_fastv"]["total_seconds"],
        fastv_accuracy_delta_same_flash=(output["flash_fastv"]["accuracy"] - output["flash_native"]["accuracy"]
                                         if output["flash_fastv"]["accuracy"] is not None else None),
    )
    return output


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=("audit", "probe", "smoke", "evaluate"), required=True)
    p.add_argument("--task", choices=("gsm8k", "humaneval"), default="gsm8k")
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--block-length", type=int, default=32)
    p.add_argument("--threshold", type=float, default=0.90)
    p.add_argument("--prune-after-layer", type=int, default=4)
    p.add_argument("--support-keep-ratio", type=float, default=0.5)
    return p.parse_args()


def main():
    args = parse_args()
    if args.stage == "audit": args.limit, args.gen_length = 1, min(args.gen_length, 64)
    if args.stage == "smoke": args.limit = min(args.limit, 2)
    samples = load_samples(args.dataset, args.limit, args.task)
    rank, world, local = (int(os.environ.get(k, d)) for k, d in (("RANK", 0), ("WORLD_SIZE", 1), ("LOCAL_RANK", 0)))
    if world > len(samples): raise ValueError("More GPUs than prompts")
    torch.cuda.set_device(local)
    if world > 1: torch.distributed.init_process_group("nccl", device_id=torch.device("cuda", local))
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()): raise ValueError("Output must be empty")
        args.output.mkdir(parents=True, exist_ok=True)
    if world > 1: torch.distributed.barrier()
    model, tokenizer = load_model(f"cuda:{local}")

    official_parity = None
    if args.stage == "audit":
        import sys
        llada_dir = Path(__file__).resolve().parents[1] / "v1" / "llada"
        sys.path.insert(0, str(llada_dir))
        from generate import generate as official_generate
        if args.task != "gsm8k": raise ValueError("Audit uses the deterministic GSM8K fixture")
        ids = prompt_ids(tokenizer, samples[0]["question"], args.task)
        source = torch.tensor([ids], device=model.device)
        with LLaDAAttentionBackend(model, "torch"):
            official, official_nfe = official_generate(
                model, source, steps=args.gen_length // args.block_length,
                gen_length=args.gen_length, block_length=args.block_length,
                temperature=0, remasking="low_confidence", threshold=args.threshold,
            )
        native = run_method(model, ids, args, "torch_native")
        if official[0].tolist() != [*ids, *native["token_ids"]] or official_nfe != native["nfe"]:
            raise AssertionError("Copied decoder differs from original LLaDA generate()")
        official_parity = dict(tokens=len(native["token_ids"]), nfe=official_nfe, exact=True)

    warmup = prompt_ids(tokenizer, "What is one plus one?", "gsm8k")
    if args.stage != "audit":
        for method in METHODS:
            run_method(model, warmup, args, method)

    path = args.output / f"rank_{rank}.jsonl"
    for index in range(rank, len(samples), world):
        sample = samples[index]
        source_text = sample["question"] if args.task == "gsm8k" else sample["prompt"]
        ids = prompt_ids(tokenizer, source_text, args.task)
        target = extract_answer(sample["answer"], gold=True) if args.task == "gsm8k" else sample["task_id"]
        record = dict(index=index, id=sample.get("id", sample.get("task_id")), target=target)
        methods = METHODS if args.stage != "probe" else ("torch_native",)
        if args.stage == "audit": methods = ("torch_native", "flash_native", "torch_fastv", "flash_fastv")
        for method in methods:
            outcome = run_method(model, ids, args, method, probe=args.stage == "probe")
            text = tokenizer.decode(outcome.pop("token_ids"), skip_special_tokens=True)
            rows = [r for r in outcome.pop("records") if r is not None]
            prediction = extract_answer(text) if args.task == "gsm8k" else None
            record[method] = dict(
                **outcome, text=text, prediction=prediction,
                correct=(prediction is not None and prediction == target) if args.task == "gsm8k" else None,
                tokens=args.gen_length,
                deep_tokens=[r["deep_tokens"] for r in rows],
                retained_mass=[r["retained_support_mass"] for r in rows],
                score_seconds=[r["score_seconds"] for r in rows],
            )
        if args.stage == "audit":
            if record["torch_native"]["text"] != record["torch_fastv"]["text"] and args.support_keep_ratio == 1:
                raise AssertionError("All-support LLaDA path changes output")
            if record["flash_native"]["backend"]["flash_calls"] <= 0:
                raise AssertionError("Explicit flash-attn did not execute")
        with path.open("a", encoding="utf-8") as f: f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"rank={rank} {index + 1}/{len(samples)} {record['id']}", flush=True)
    if world > 1: torch.distributed.barrier()
    if rank == 0:
        records=[]
        for worker in range(world):
            records += [json.loads(x) for x in (args.output/f"rank_{worker}.jsonl").read_text(encoding="utf-8").splitlines()]
        records.sort(key=lambda x:x["index"])
        if [x["index"] for x in records] != list(range(len(samples))): raise AssertionError("Bad distributed shard")
        if args.stage == "probe":
            rows=[r for record in records for r in record["torch_native"]["deep_tokens"]]
            masses=[r for record in records for r in record["torch_native"]["retained_mass"]]
            scores=[r for record in records for r in record["torch_native"]["score_seconds"]]
            results=dict(probe=dict(calls=len(rows), mean_deep_tokens=sum(rows)/len(rows),
                mean_retained_support_mass=sum(masses)/len(masses), mean_score_seconds=sum(scores)/len(scores)))
        else:
            results=aggregate(records)
        report=dict(stage=args.stage, model=MODEL_ID, revision=REVISION, dataset=str(args.dataset),
            dataset_sha256=sha256(args.dataset), ids=[x.get("id", x.get("task_id")) for x in samples], world_size=world,
            configuration=vars(args) | {"dataset":str(args.dataset),"output":str(args.output)}, results=results,
            official_parity=official_parity,
            scope="Original LLaDA-8B-Instruct weights; training-free support-token pruning; paired backend attribution.")
        write_json(args.output/"summary.json",report); print(json.dumps(results,indent=2),flush=True)
    if world > 1: torch.distributed.destroy_process_group()


if __name__ == "__main__": main()

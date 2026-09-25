from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from .audit import audit
from .collect import collect
from .core import ExperimentConfig
from .data import prepare
from .evaluate import compare_runs, evaluate
from .export import export
from .model import load_tokenizer
from .train import train


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="python -m llada_step_distill")
    sub = root.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--nemotron-root", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--gsm8k-json", type=Path)
    command.add_argument("--train-size", type=int, default=10_000_000)
    command.add_argument("--validation-size", type=int, default=20_000)
    command.add_argument("--shard-size", type=int, default=2048)
    command.add_argument("--oversample", type=float, default=1.05)

    command = sub.add_parser("collect")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("all", "train", "validation"), default="all")
    command.add_argument("--attempts", type=int, default=8)

    command = sub.add_parser("audit")
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--limit", type=int, default=256)

    for name in ("smoke", "train"):
        command = sub.add_parser(name)
        command.add_argument("--prepared", type=Path, required=True)
        command.add_argument("--supervision", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--stage", choices=("a", "b"), default="a")
        command.add_argument("--resume", type=Path)
        command.add_argument("--init-adapter", type=Path)
        command.add_argument("--dev-dataset", type=Path)
        command.add_argument("--updates", type=int, default=200 if name == "smoke" else None)
        command.add_argument("--overfit-records", type=int)

    command = sub.add_parser("evaluate")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "holdout", "full"), default="dev")
    command.add_argument("--steps", type=int, nargs="+", default=(8, 16, 32))
    command.add_argument("--adapter", type=Path)
    command.add_argument("--limit", type=int)

    command = sub.add_parser("export")
    command.add_argument("--checkpoint", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)

    command = sub.add_parser("report")
    command.add_argument("--teacher", type=Path, required=True)
    command.add_argument("--stage-a", type=Path, required=True)
    command.add_argument("--stage-b", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "prepare":
        result = prepare(args.nemotron_root, args.output, load_tokenizer(), args.gsm8k_json, args.train_size, args.validation_size, args.shard_size, args.oversample)
    elif args.command == "collect":
        result = collect(args.prepared, args.output, args.split, args.attempts)
    elif args.command == "audit":
        args.output.mkdir(parents=True, exist_ok=True)
        engineering = audit(args.output / "engineering.json", "cuda:0")
        quality = evaluate(args.dataset, args.output / "teacher_quality", split="dev", steps=(16, 32), limit=args.limit)
        result = {"engineering": engineering, "quality": quality}
        if quality:
            same = quality["summary"]["16"]["accuracy"] == quality["summary"]["32"]["accuracy"]
            result["teacher_16_equals_32_on_audit"] = same
            result["recommendation"] = "stop_distillation_directly_reduce_steps" if same else "continue_collection"
            (args.output / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    elif args.command in {"smoke", "train"}:
        result = train(args.prepared, args.supervision, args.output, stage=args.stage, resume=args.resume,
                       init_adapter=args.init_adapter, dev_dataset=args.dev_dataset,
                       smoke_updates=args.updates if args.command == "smoke" else None,
                       overfit_records=args.overfit_records,
                       config=ExperimentConfig())
    elif args.command == "evaluate":
        result = evaluate(args.dataset, args.output, split=args.split, steps=tuple(args.steps), adapter=args.adapter, limit=args.limit)
    elif args.command == "export":
        result = export(args.checkpoint, args.output)
    else:
        result = compare_runs(args.teacher, args.stage_a, args.stage_b, args.output)
    rank = int(os.environ.get("RANK", 0))
    if rank == 0 and result is not None:
        print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()

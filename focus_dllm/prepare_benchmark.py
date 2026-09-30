"""Materialize cached benchmark rows into the runner's deterministic JSON format."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=("gsm8k", "humaneval"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    from datasets import load_dataset
    if args.task == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        rows = [dict(id=f"test:{i}", question=x["question"], answer=x["answer"])
                for i, x in enumerate(dataset)]
    else:
        dataset = load_dataset("openai/openai_humaneval", split="test")
        keys = ("task_id", "prompt", "canonical_solution", "test", "entry_point")
        rows = [{key: x[key] for key in keys} for x in dataset]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()

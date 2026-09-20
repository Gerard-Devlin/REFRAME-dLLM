import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    args = p.parse_args()
    rows = [json.loads((d / "summary.json").read_text()) for d in args.runs]
    for key in ("ids", "max_new_tokens", "data_hash", "evaluation_data_hash", "block_size", "gpu", "torch"):
        if any(x[key] != rows[0][key] for x in rows):
            raise ValueError(f"Evaluation mismatch: {key}")
    trained = [x["checkpoint"] for x in rows if x["checkpoint"]]
    for key in ("steps", "completed_steps", "original_tokens", "global_batch", "lr", "seed", "objective", "adaptation", "implementation"):
        if trained and any(x[key] != trained[0][key] for x in trained):
            raise ValueError(f"Training mismatch: {key}")
    print("arm\trounds\taccuracy\tseconds\ttokens/s\tcalls\ttruncated")
    for x in rows:
        for rounds, r in x["results"].items():
            print(f"{x['arm']}\t{rounds}\t{r['accuracy']:.4f}\t{r['mean_seconds']:.3f}\t{r['tokens_per_second']:.2f}\t{r['mean_calls']:.2f}\t{r['truncation_rate']:.4f}")
    print("No automatic same-quality speedup verdict. Small dev sets are screening only; inspect paired outputs and uncertainty.")


if __name__ == "__main__":
    main()

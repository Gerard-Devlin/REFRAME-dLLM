"""Reject unmatched training budgets before reporting quality/latency curves."""
import argparse
import json
from pathlib import Path


MATCH = ("model", "parameter_count", "global_batch", "trained_steps", "seen_symbols",
         "dtype", "lr", "planned_steps", "train_sha256", "data_sha256", "examples", "gpu")


def compare(results):
    seeds = sorted({r["seed"] for r in results})
    rows, candidates = [], []
    for seed in seeds:
        group = [r for r in results if r["seed"] == seed]
        baselines = [r for r in group if r["codec"] == "identity" and r["objective"] == "diffusion"]
        if len(baselines) != 1:
            raise ValueError(f"Need exactly one identity/diffusion baseline for seed {seed}")
        baseline = baselines[0]
        seen = set()
        for result in group:
            name = result["codec"] + "/" + result["objective"]
            if name in seen:
                raise ValueError(f"Duplicate method: {seed}/{name}")
            seen.add(name)
            for key in MATCH:
                if result[key] != baseline[key]:
                    raise ValueError(f"Unmatched {key} for {seed}/{name}; compare actual completed budgets, not requested budgets")
            for score in result["scores"]:
                rows.append(dict(seed=seed, method=name, steps=score["steps"],
                                 bits_per_symbol=score["path_bits_per_symbol"],
                                 seconds=score["timing"]["mean_seconds"],
                                 decode_seconds=score["timing"]["mean_decode_seconds"]))
                for base in baseline["scores"]:
                    if score["steps"] < base["steps"] and result["codec"].startswith("relation"):
                        candidates.append(dict(seed=seed, method=name, steps=score["steps"],
                                               baseline_steps=base["steps"],
                                               delta_bits=score["path_bits_per_symbol"] - base["path_bits_per_symbol"],
                                               measured_time_ratio=base["timing"]["mean_seconds"] / score["timing"]["mean_seconds"],
                                               lower_or_equal_mean_path_nll=score["path_bits_per_symbol"] <= base["path_bits_per_symbol"]))
    return dict(rows=rows, fewer_step_comparisons=candidates,
                decision="Descriptive pilot only. No automatic promotion; require repeated seeds, enough training and text/sample checks before BPE or LLaDA.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.root.rglob("evaluation.json"))
    if not paths or args.output.exists():
        p.error("Need evaluation.json files and a fresh output")
    result = compare([json.loads(path.read_text(encoding="utf-8")) for path in paths])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print("seed method steps path_bits/symbol seconds inverse_seconds")
    for r in result["rows"]:
        print(f"{r['seed']} {r['method']} {r['steps']} {r['bits_per_symbol']:.4f} {r['seconds']:.6f} {r['decode_seconds']:.6f}")
    print(result["decision"])


if __name__ == "__main__":
    main()

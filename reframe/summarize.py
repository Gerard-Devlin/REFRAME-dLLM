"""Summarize JSONL without treating oracle/audit/tiny runs as speed evidence."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
from statistics import mean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise SystemExit("No records")
    groups = defaultdict(list)
    if rows[0].get("oracle"):
        for r in rows:
            groups[(r["kind"], r["age_blocks"], r["pilots"], r["side"])].append(r)
        print("ORACLE ONLY; attention error uses true full-state queries and pilots.")
        print("kind age pilots side observations mean_K_error mean_V_error mean_attention_error")
        for key, values in sorted(groups.items()):
            print(*key, len(values), *(f"{mean(r[m] for r in values):.6f}" for m in
                                       ("key_error", "value_error", "attention_error")))
        return
    diagnostic = any(r.get("tiny") or r["stats"].get("diagnostic_run") for r in rows)
    print("TINY/AUDIT: these times are NOT real-model acceleration evidence." if diagnostic else
          "Generation-only timings; includes per-request initialization and all fallback work.")
    for r in rows:
        groups[r["method"]].append(r)
    print("method samples mean_seconds mean_NFE mean_full mean_fallback useful_tok_per_second")
    for method, values in sorted(groups.items()):
        elapsed = sum(r["stats"]["elapsed_seconds"] for r in values)
        tokens = [r.get("useful_tokens") for r in values]
        tps = (f"{sum(tokens) / elapsed:.3f}" if not diagnostic and all(t is not None for t in tokens) else "NA")
        print(method, len(values), *(f"{mean(r['stats'].get(m, 0) for r in values):.3f}" for m in
                                    ("elapsed_seconds", "nfe", "full_forwards", "fallback_refreshes")), tps)


if __name__ == "__main__":
    main()

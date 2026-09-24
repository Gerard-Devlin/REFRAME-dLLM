"""Print a compact paired quality/latency table from an evaluation summary."""

import argparse
import json
from pathlib import Path


def render(summary):
    results = summary["results"]
    baseline = results.get("official")
    lines = [f"mode={summary['mode']}  model={summary['model']}",
             f"{'method':<14} {'acc%':>7} {'delta pp':>9} {'NFE':>8} {'sec':>8} {'speed':>7} {'cap%':>7} {'lost/gained':>12}"]
    for name, row in results.items():
        if "accuracy" not in row:
            continue
        delta = 100 * (row["accuracy"] - baseline["accuracy"]) if baseline else 0
        speed = baseline["total_seconds"] / row["total_seconds"] if baseline else 1
        paired = results.get("paired", {})
        pair = paired.get(name, {}) if summary["mode"] == "sweep" else paired if name == "budget" else {}
        transitions = f"{pair['correct_to_wrong']}/{pair['wrong_to_correct']}" if "correct_to_wrong" in pair else "-"
        lines.append(f"{name:<14} {row['accuracy'] * 100:7.2f} {delta:+9.2f} {row['mean_nfe']:8.2f} "
                     f"{row['mean_seconds']:8.3f} {speed:6.2f}x {row['truncation_rate'] * 100:7.2f} {transitions:>12}")
    lines.append("speed = summed native sample latency / summed method latency; not multi-GPU throughput.")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("summary", type=Path)
    args = parser.parse_args()
    print(render(json.loads(args.summary.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()

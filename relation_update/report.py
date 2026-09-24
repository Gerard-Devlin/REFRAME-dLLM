"""Compact next-forward diagnosis; no task-accuracy or speedup claims."""

import argparse
import json
from pathlib import Path


def render(report):
    lines = ["Offline next-forward prediction (held-out prompts)",
             f"{'head':<13} {'KL':>9} {'changed KL':>11} {'top1%':>8} {'changed%':>10} {'commit prec%':>13} {'exact step%':>12}"]
    def number(value, scale=1):
        return "n/a" if value is None else f"{value*scale:.4f}"
    for name, row in report["results"].items():
        lines.append(f"{name:<13} {number(row.get('mean_kl')):>9} "
                     f"{number(row.get('changed_mean_kl')):>11} "
                     f"{number(row.get('next_top1_agreement'),100):>8} "
                     f"{number(row.get('changed_top1_agreement'),100):>10} "
                     f"{number(row.get('proposal_precision'),100):>13} "
                     f"{number(row.get('exact_transition_agreement'),100):>12}")
    baseline = report["results"]["reuse"]
    lines.append(f"Teacher top1 candidate coverage: {number(baseline.get('teacher_top1_coverage'),100)}%; "
                 f"on changed positions: {number(baseline.get('changed_teacher_top1_coverage'),100)}%")
    lines.append("Changed% counts uncovered teacher tokens as failures. OTHER is not a selectable token.")
    lines.append("No online model calls replaced yet. Task accuracy and end-to-end speedup are unmeasured.")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("summary", type=Path)
    args = p.parse_args()
    print(render(json.loads(args.summary.read_text(encoding="utf-8"))))


if __name__ == "__main__":
    main()

"""CPU-only accounting of the completed atomic-epoch mechanism screen."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def analyze(audit, flash):
    result = {}
    for k in (2, 4, 8):
        rows = [r for r in audit["records"] if r["k"] == k]
        passed = [r for r in rows if r["atomic_pass"]]
        reference_complete = [r for r in rows if all(r["teacher_final_matches"])]
        cost = sum(r["regular_ms"]*(1+int(not all(r["teacher_final_matches"]))) for r in rows)
        progress = sum(r["safe_progress"]+k*int(all(r["teacher_final_matches"])) for r in rows)
        flash_cost = sum(r["regular_ms"]+r["verify_ms"] for r in rows)
        flash_progress = sum(r["safe_progress"]+r["official_accepted"] for r in rows)
        result[str(k)] = {
            "windows": len(rows), "actual_passed_epochs": len(passed),
            "epochs_entirely_matching_reference_final": len(reference_complete),
            "reference_all_match_optimistic_rate_ratio": (progress/cost)/(flash_progress/flash_cost),
            "individual_top1_matches": sum(sum(a == b for a,b in zip(r["drafts"],r["top1"])) for r in rows),
            "individual_candidates": len(rows)*k,
            "scope": "Teacher final is a retrospective reference, not gold or a deployable audit. This is not a universal speed bound."
        }
    cycles = flash["cycles"]
    adjacent = [(a,b) for a,b in zip(cycles, cycles[1:]) if (a["task"], a["id"]) == (b["task"], b["id"])]
    distinct_accepted = {(r["task"],r["id"],p,v) for r in audit["records"] if r["atomic_pass"]
                         for p,v in zip(r["positions"],r["drafts"])}
    return {
        "atomic_reference_analysis": result,
        "distinct_accepted_position_value_tuples_across_k": len(distinct_accepted),
        "flash_prefix_length_predictability": {
            "cycles": len(cycles), "zero_accepted": sum(r["accepted"] == 0 for r in cycles),
            "histogram": dict(sorted(Counter(r["accepted"] for r in cycles).items())),
            "same_prompt_adjacent_pairs": len(adjacent),
            "predict_previous_count_correct": sum(a["accepted"] == b["accepted"] for a,b in adjacent),
            "scope": "Descriptive six-prompt trace only. Rejects blind fixed/previous-count branch guessing; does not evaluate logits-based predictors."
        },
        "own_label_private_controls": {
            "checks": sum(len(r["controls"]) for r in audit["private_controls"]),
            "all_own_errors_zero": all(c["own_max_logit_error"] == 0 for r in audit["private_controls"] for c in r["controls"])
        },
        "decision": "Stop the deferred all-or-none epoch architecture. Do not implement fusion or free generation from these data.",
        "interpretation": "Removing source contamination is feasible, but low audit acceptance and whole-epoch abort costs both hurt. Even a retrospective all-reference-matching batch rule lacks a strong cost signal here.",
        "statistical_scope": "Six reused development prompts, 12 correlated windows; descriptive counts, no independent holdout or task accuracy claim."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--flash", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = analyze(json.loads(args.audit.read_text()), json.loads(args.flash.read_text()))
    report["source_sha256"] = {str(p): digest(p) for p in (args.audit, args.flash)}
    args.output.write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

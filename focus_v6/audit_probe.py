"""Fixed-state mechanism gate for independent-source atomic epochs.

Official Flash owns every online commit. All audit forwards are paid shadows.
No speed or task-accuracy result follows from this diagnostic.
"""
import argparse
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import statistics
from types import SimpleNamespace

import torch

from focus_v5.relay_joint_probe import TimedModel
from .audit import build_call, instrument
from .epoch import atomic_pass


@contextmanager
def forbid_sdpa():
    original = torch.nn.functional.scaled_dot_product_attention
    def forbidden(*args, **kwargs):
        raise AssertionError("unexpected SDPA call; require pinned fused Flash kernels")
    torch.nn.functional.scaled_dot_product_attention = forbidden
    try:
        yield
    finally:
        torch.nn.functional.scaled_dot_product_attention = original


def summarize(records):
    result = {}
    for k in (2, 4, 8):
        rows = [r for r in records if r["k"] == k]
        if not rows:
            result[str(k)] = {"windows": 0, "gate_passed": False}
            continue
        passed = [r for r in rows if r["atomic_pass"]]
        accepted = len(passed) * k
        matched = sum(sum(r["teacher_final_matches"]) for r in passed)
        task_rates = {}
        task_agreement = {}
        for task in ("humaneval", "mbpp", "math"):
            group = [r for r in rows if r["task"] == task]
            good = [r for r in group if r["atomic_pass"]]
            task_agreement[task] = (sum(sum(r["teacher_final_matches"]) for r in good) / (len(good)*k)
                                    if good else None)
            if group:
                # Theoretical lower cost: one regular call per joint cycle,
                # another per abort. Audit rows, setup, drain and transactions
                # are absent, so this is an optimistic necessary condition.
                cost = sum(r["regular_ms"] * (1 + int(not r["atomic_pass"])) for r in group)
                progress = sum(r["safe_progress"] + k * int(r["atomic_pass"]) for r in group)
                native_cost = sum(r["regular_ms"] + r["verify_ms"] for r in group)
                native_progress = sum(r["safe_progress"] + r["official_accepted"] for r in group)
                task_rates[task] = (progress/cost)/(native_progress/native_cost)
        group_cost = sum(r["regular_ms"] * (1 + int(not r["atomic_pass"])) for r in rows)
        progress = sum(r["safe_progress"] + k * int(r["atomic_pass"]) for r in rows)
        baseline_cost = sum(r["regular_ms"] + r["verify_ms"] for r in rows)
        baseline_progress = sum(r["safe_progress"] + r["official_accepted"] for r in rows)
        ratio = (progress/group_cost)/(baseline_progress/baseline_cost)
        agreement = matched/accepted if accepted else None
        result[str(k)] = {
            "windows": len(rows), "passed_epochs": len(passed),
            "pass_fraction": len(passed)/len(rows), "accepted_epoch_tokens": accepted,
            "teacher_final_matches": matched, "teacher_final_agreement": agreement,
            "task_agreement": task_agreement, "task_optimistic_rate_ratio": task_rates,
            "optimistic_effective_rate_ratio": ratio,
            "median_paid_shadow_audit_ms": statistics.median(r["audit_ms"] for r in rows),
            "prompt_count": len({(r["task"], r["id"]) for r in rows}),
            "gate_passed": accepted >= 12 and agreement is not None and agreement >= .98
                           and all(v is not None and v >= .95 for v in task_agreement.values())
                           and ratio >= 1.2 and all(v >= 1.0 for v in task_rates.values()),
            "scope": "Optimistic necessary screen on reused fixed states; not achieved speedup or task accuracy.",
        }
    return result


@torch.no_grad()
def main():
    from focus_dllm.common import sha256, write_json
    from focus_dllm.llada_common import MODEL_ID, REVISION, prompt_ids
    from focus_dllm.tuning.competitors import generation_prompt, load_external, load_model, select_samples
    from focus_dllm.tuning.flash_readout import suppress_official_prints
    from focus_dllm.tuning.gpu_contract import check_binding

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, nargs=3, required=True)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    cls, external, adaptation = load_external(args.third_party, "flash_verify")
    official_source = Path(inspect.getsourcefile(inspect.unwrap(external)))
    binding = check_binding(required=True)
    raw, tokenizer = load_model(SimpleNamespace(method="flash_verify"), cls)
    model = TimedModel(raw)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    forbidden = set(tokenizer.all_special_ids) | {126336, 126081}
    reference = {(r["task"], str(r["id"])): r for r in json.loads(args.reference.read_text())["prompts"]}
    report = {
        "model": MODEL_ID, "revision": REVISION, "binding": binding,
        "official_source_sha256": sha256(official_source), "adaptation": adaptation,
        "implementation": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "datasets": {str(p): sha256(p) for p in args.datasets},
        "configuration": {"length": 256, "block": 32, "threshold": .9, "gamma": .8,
                          "k": [2, 4, 8], "windows_per_prompt": 2,
                          "prompts_per_task": 2, "seed": 51713, "offset": 0,
                          "first_eligible": "search>=8, first8 no special tokens; one pending batch"},
        "records": [], "prompts": [], "private_controls": [],
        "scope": "Read-only paid shadow audit; official Flash commits unchanged. No gold input."
    }
    write_json(args.output / "diagnostic.json", report)
    windows = []
    current_cycle = [0]
    pending = [None]
    final = [None]
    controlled_tasks = set()

    def run_audit(query, positions, lengths, layout, drafts):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        logits = raw(query, use_cache=True, positions=positions, lengths=lengths).logits
        end.record()
        audit = logits[0, layout.audit].detach()
        probability = audit.double().softmax(-1).gather(1, drafts[:, None]).squeeze(1)
        return audit, probability, (start, end)

    def before_verify(proxy, state):
        cycle = current_cycle[0]
        current_cycle[0] += 1
        if len(windows) >= 2 or int(state["num_verify"]) < 8:
            return
        decoded = int(state["num_decoded"][0])
        candidate_pos = state["full_pos"][0, decoded:decoded+8]
        candidate_tokens = state["x_draft"][candidate_pos]
        if any(int(t) in forbidden for t in candidate_tokens.tolist()):
            return
        if pending[0] is not None:
            raise AssertionError("pending official decision")
        blocks = raw.model.transformer.blocks
        # Contents as well as mutation counters are checked for first/last
        # public layers; the official verify kernel cannot write public K/V.
        snapshots = [(b.k_cache.clone(), b.v_cache.clone()) for b in (blocks[0], blocks[-1])]
        versions = [(b.k_cache._version, b.v_cache._version) for b in blocks]
        window = {**model.context, "cycle": cycle,
                  "safe_progress": int(state["num_newly_decoded"][0]), "audits": []}
        for k in (2, 4, 8):
            query, pos, lengths, layout, candidates, drafts = build_call(state, k)
            audit, probability, events = run_audit(query, pos, lengths, layout, drafts)
            row = {**model.context, "cycle": cycle, "k": k,
                   "positions": candidates.tolist(), "drafts": drafts.tolist(),
                   "probabilities": probability.tolist(), "top1": audit.argmax(-1).tolist(),
                   "safe_progress": window["safe_progress"], "event": events}
            row["atomic_pass"] = atomic_pass(row["probabilities"], row["top1"], row["drafts"])
            window["audits"].append(row)
            if k == 4 and model.context["task"] not in controlled_tasks:
                controls = []
                for i in range(k):
                    perturbed = query.clone()
                    original = int(perturbed[0, layout.draft.start+i])
                    replacement = 15 if original != 15 else 16  # tokenizer IDs for ordinary numeric text
                    if replacement in forbidden:
                        raise AssertionError("perturbation token is special")
                    perturbed[0, layout.draft.start+i] = replacement
                    changed, _, _ = run_audit(perturbed, pos, lengths, layout, drafts)
                    own_error = float((changed[i].float()-audit[i].float()).abs().max())
                    peer_error = float((changed.float()-audit.float()).abs().max())
                    if own_error != 0:
                        raise AssertionError(f"own label leaked through private source paths: {own_error}")
                    controls.append({"candidate": i, "own_max_logit_error": own_error,
                                     "peer_max_logit_change": peer_error})
                report["private_controls"].append({**model.context, "cycle": cycle, "controls": controls})
                controlled_tasks.add(model.context["task"])
        if versions != [(b.k_cache._version, b.v_cache._version) for b in blocks]:
            raise AssertionError("shadow mutated a public cache tensor")
        for saved, block in zip(snapshots, (blocks[0], blocks[-1])):
            if any(not torch.allclose(old, new, rtol=0, atol=0, equal_nan=True)
                   for old, new in zip(saved, (block.k_cache, block.v_cache))):
                raise AssertionError("shadow changed public cache contents")
        windows.append(window)
        pending[0] = window

    def official_accept(probability, drafts, gamma):
        if pending[0] is not None:
            pending[0]["official_accepted"] = int((probability.double().cumprod(0) >= gamma).sum())
            pending[0]["official_probabilities"] = probability.tolist()
            pending[0] = None

    def final_canvas(canvas, prompt_length, max_length, gen_length):
        if final[0] is not None or gen_length != 256:
            raise AssertionError("unexpected final canvas observation")
        final[0] = canvas[:max_length].detach().cpu().tolist()

    traced = instrument(external, before_verify, official_accept, final_canvas)
    for task, path in zip(("humaneval", "mbpp", "math"), args.datasets):
        for sample in select_samples(path, 2, 0):
            ident = str(sample.get("id", sample.get("task_id")))
            model.context = {"task": task, "id": ident}
            windows.clear(); current_cycle[0] = 0; final[0] = None
            event_start = len(model.events)
            ids = prompt_ids(tokenizer, generation_prompt(sample), task, preformatted=True)
            response, steps = [None], [0]
            with forbid_sdpa(), suppress_official_prints():
                traced(model, [torch.tensor(ids, device=raw.device)], [len(ids)], 1, response, steps,
                       gen_length=256, block_length=32, threshold=.9, gamma=.8,
                       track_num=4, mask_num=4, verify=True, tokenizer=tokenizer, stop_tokens=[])
            torch.cuda.synchronize()
            if pending[0] is not None or final[0] is None:
                raise AssertionError("incomplete observer ledger")
            events = model.events[event_start:]
            regular = [e for e in events if e["kind"] == "regular"]
            verify = [e for e in events if e["kind"] == "verify"]
            if not len(regular) == len(verify) == current_cycle[0] == int(steps[0]):
                raise AssertionError("official iterations changed")
            text_sha = hashlib.sha256(response[0].encode()).hexdigest()
            if text_sha != reference[(task, ident)]["text_sha256"]:
                raise AssertionError("read-only audits changed official output")
            # Warm-up/compiler work cannot create fictitious fusion savings.
            # Use steady official calls at the same query geometry when
            # available, otherwise the median steady official geometry. The
            # latter is deliberately optimistic and documented per record.
            stable_regular = regular[3:] or regular
            stable_verify = verify[3:] or verify
            for window in windows:
                index = window["cycle"]
                reg, ver = regular[index], verify[index]
                for row in window["audits"]:
                    start, end = row.pop("event")
                    row["audit_ms"] = start.elapsed_time(end)
                    same = [e for e in stable_regular if e["rows"] == reg["rows"]]
                    candidates = same or stable_regular
                    row["observed_regular_ms"] = reg["start"].elapsed_time(reg["end"])
                    row["observed_verify_ms"] = ver["start"].elapsed_time(ver["end"])
                    row["regular_ms"] = statistics.median(e["start"].elapsed_time(e["end"]) for e in candidates)
                    row["verify_ms"] = statistics.median(e["start"].elapsed_time(e["end"]) for e in stable_verify)
                    row["cost_reference"] = "steady same-row geometry" if same else "optimistic steady median geometry"
                    row["official_accepted"] = window["official_accepted"]
                    row["teacher_final_matches"] = [final[0][p] == v for p, v in zip(row["positions"], row["drafts"])]
                    row["public_cache_unchanged"] = True
                    report["records"].append(row)
            report["prompts"].append({"task": task, "id": ident, "windows": len(windows),
                "official_cycles": steps[0], "text_sha256": text_sha,
                "final_canvas_sha256": hashlib.sha256(json.dumps(final[0]).encode()).hexdigest(),
                "official_output_unchanged": True, "sdpa_calls": 0})
            write_json(args.output / "diagnostic.json", report)
            print("PROMPT", task, ident, "windows", len(windows), flush=True)
    report["summary"] = summarize(report["records"])
    report["gate_passed"] = any(row["gate_passed"] for row in report["summary"].values())
    if report["implementation"] != {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}:
        raise AssertionError("research source changed")
    if sha256(official_source) != report["official_source_sha256"]:
        raise AssertionError("official source changed")
    write_json(args.output / "diagnostic.json", report)
    (args.output / "complete").write_text("OK\n")
    print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()

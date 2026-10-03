"""Real GPU preflight for the FOCUS-v5 Relay joint Transformer pass.

Official Flash generation remains authoritative and unchanged.  Immediately
before each official verification call, the probe executes a read-only
96-live-row Relay shadow call in a padded 128-row Triton tile.  It measures
CUDA-event latency, applies the same cumulative verification rule, and proves
that the public cache is not modified.  This is still a state-level preflight:
it does not claim end-to-end Relay quality or speed.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
import statistics
from types import SimpleNamespace
from typing import Any

import torch

from .relay_joint import accepted_prefix, build_joint_call, instrument_before_verify


class TimedModel:
    """Transparent model proxy that records only official model calls."""

    def __init__(self, model):
        self.model_ref = model
        self.events: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}

    def __getattr__(self, name):
        return getattr(self.model_ref, name)

    def __call__(self, *args, **kwargs):
        lengths = kwargs.get("lengths")
        if lengths is None:
            raise AssertionError("unexpected non-Flash model call")
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        output = self.model_ref(*args, **kwargs)
        end.record()
        self.events.append({
            **self.context,
            "kind": "verify" if bool(lengths[-1]) else "regular",
            "rows": int(args[0].shape[1]),
            "start": start,
            "end": end,
        })
        return output


def _sha_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _timing(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("no timing samples")
    return {
        "count": len(values),
        "mean_ms": statistics.fmean(values),
        "median_ms": statistics.median(values),
        "min_ms": min(values),
        "max_ms": max(values),
        "sum_ms": sum(values),
    }


@torch.no_grad()
def main() -> None:
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
    parser.add_argument("--prompts-per-task", type=int, default=1)
    parser.add_argument("--joint-repeats", type=int, default=3)
    parser.add_argument("--discard-cycles", type=int, default=3)
    args = parser.parse_args()
    if args.prompts_per_task < 1 or args.joint_repeats < 2 or args.discard_cycles < 0:
        raise ValueError("invalid preflight size")
    args.output.mkdir(exist_ok=False)

    binding = check_binding(required=True)
    cls, external, adaptation = load_external(args.third_party, "flash_verify")
    official_file = Path(inspect.getsourcefile(inspect.unwrap(external)))
    raw_model, tokenizer = load_model(SimpleNamespace(method="flash_verify"), cls)
    model = TimedModel(raw_model)
    torch.set_num_threads(1)
    torch.manual_seed(1234)

    reference = json.loads(args.reference.read_text())
    reference_prompts = {(row["task"], str(row["id"])): row for row in reference["prompts"]}
    task_paths = dict(zip(("humaneval", "mbpp", "math"), args.datasets))
    report: dict[str, Any] = {
        "model": MODEL_ID,
        "revision": REVISION,
        "binding": binding,
        "official_generate_sha256": sha256(official_file),
        "implementation": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "datasets": {name: sha256(path) for name, path in task_paths.items()},
        "reference_sha256": sha256(args.reference),
        "adaptation": adaptation,
        "configuration": {
            "length": 256, "block": 32, "threshold": .9, "gamma": .8,
            "track_num": 4, "mask_num": 4,
            "prompts_per_task": args.prompts_per_task,
            "joint_repeats": args.joint_repeats,
            "joint_live_rows": 96, "joint_kernel_rows": 128,
        },
        "prompts": [],
        "cycles": [],
        "scope": (
            "Read-only Relay shadow calls on reused development prompts. Official Flash owns all commits. "
            "No gold, tests, solutions, future tokens, or teacher state enter generation."
        ),
    }
    write_json(args.output / "diagnostic.json", report)

    joint_records: list[dict[str, Any]] = []
    pending_official: list[dict[str, Any] | None] = [None]
    cache_checked = [False]

    def joint_shadow(proxy, state):
        if proxy is not model:
            raise AssertionError("generator model proxy changed")
        call = build_joint_call(state)
        record: dict[str, Any] = {
            **model.context,
            "cycle": sum(1 for row in joint_records if row["task"] == model.context["task"]
                         and row["id"] == model.context["id"]),
            "search": call.layout.search,
            "tracked": call.layout.tracked,
            "clean": 32,
            "live_rows": call.layout.total,
            "kernel_rows": 128,
            "key_rows": int(call.positions[1].numel()),
            "joint_events": [],
            "official_accepted": 0 if call.layout.search == 0 else None,
        }
        if pending_official[0] is not None:
            raise AssertionError("official acceptance callback missing")

        cache_before = None
        if not cache_checked[0]:
            blocks = raw_model.model.transformer.blocks
            cache_before = [
                (blocks[index].k_cache.clone(), blocks[index].v_cache.clone())
                for index in (0, len(blocks) - 1)
            ]

        result = None
        decision = None
        first_logits = None
        for repeat in range(args.joint_repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            result = raw_model(call.input_ids, use_cache=True, lengths=call.lengths, positions=call.positions)
            end.record()
            record["joint_events"].append((start, end))
            current = accepted_prefix(result.logits, call, float(state["gamma"]))
            if decision is None:
                decision = current
                first_logits = result.logits.detach().clone()
            else:
                if current != decision or not torch.equal(result.logits, first_logits):
                    raise AssertionError("repeated joint pass changed logits or decisions")
        del result, first_logits
        record["relay"] = decision

        if cache_before is not None:
            blocks = raw_model.model.transformer.blocks
            for saved, index in zip(cache_before, (0, len(blocks) - 1)):
                for old, new in zip(saved, (blocks[index].k_cache, blocks[index].v_cache)):
                    if not torch.allclose(old, new, rtol=0, atol=0, equal_nan=True):
                        raise AssertionError("read-only joint verify mutated the public cache")
            cache_checked[0] = True
            record["cache_immutability_checked"] = True

        joint_records.append(record)
        if call.layout.search:
            pending_official[0] = record

    def official_accept(probability, draft, positions, top1, gamma):
        record = pending_official[0]
        if record is None:
            raise AssertionError("official acceptance without a pending Relay state")
        if record["search"] != int(probability.numel()):
            raise AssertionError("official and Relay search sizes differ")
        cumulative = probability.detach().double().cumprod(0)
        record["official_probabilities"] = probability.detach().cpu().tolist()
        record["official_top1"] = top1.detach().cpu().tolist()
        record["official_drafts"] = draft.detach().cpu().tolist()
        record["official_positions"] = positions.detach().cpu().tolist()
        record["official_accepted"] = int((cumulative >= float(gamma)).sum().item())
        pending_official[0] = None

    traced = instrument_before_verify(external, joint_shadow, official_accept)

    for task, path in task_paths.items():
        for sample in select_samples(path, args.prompts_per_task, 0):
            ident = str(sample.get("id", sample.get("task_id")))
            ids = prompt_ids(tokenizer, generation_prompt(sample), task, preformatted=True)
            prompt = torch.tensor(ids, device=raw_model.device)
            event_start = len(model.events)
            joint_start = len(joint_records)
            model.context = {"task": task, "id": ident}
            responses, steps = [None], [0]
            with suppress_official_prints():
                traced(model, [prompt], [len(ids)], 1, responses, steps,
                       gen_length=256, block_length=32, threshold=.9, gamma=.8,
                       track_num=4, mask_num=4, verify=True,
                       tokenizer=tokenizer, stop_tokens=[])
            if pending_official[0] is not None:
                raise AssertionError("unfinished official acceptance record")
            torch.cuda.synchronize()

            events = model.events[event_start:]
            regular = [row for row in events if row["kind"] == "regular"]
            verify = [row for row in events if row["kind"] == "verify"]
            joints = joint_records[joint_start:]
            if not len(regular) == len(verify) == len(joints) == int(steps[0]):
                raise AssertionError("official/joint cycle alignment changed")
            for index, (reg, ver, joint) in enumerate(zip(regular, verify, joints)):
                reg_ms = reg["start"].elapsed_time(reg["end"])
                ver_ms = ver["start"].elapsed_time(ver["end"])
                joint_ms = [start.elapsed_time(end) for start, end in joint.pop("joint_events")]
                joint["regular_rows"] = reg["rows"]
                joint["verify_rows"] = ver["rows"]
                joint["regular_ms"] = reg_ms
                joint["verify_ms"] = ver_ms
                joint["flash_pair_ms"] = reg_ms + ver_ms
                joint["joint_repeat_ms"] = joint_ms
                joint["joint_ms"] = statistics.median(joint_ms)
                joint["state_speedup"] = (reg_ms + ver_ms) / joint["joint_ms"]
                joint["acceptance_delta"] = joint["relay"]["accepted"] - int(joint["official_accepted"])

            text_hash = _sha_text(responses[0])
            expected = reference_prompts.get((task, ident))
            if expected is None or expected["text_sha256"] != text_hash:
                raise AssertionError("read-only probe changed the pinned official Flash output")
            report["prompts"].append({
                "task": task, "id": ident, "prompt_tokens": len(ids),
                "cycles": len(joints), "official_iterations": int(steps[0]),
                "text_sha256": text_hash,
            })
            report["cycles"].extend(joints)
            write_json(args.output / "diagnostic.json", report)

    usable = report["cycles"][args.discard_cycles:]
    if not usable:
        raise AssertionError("all cycles were discarded")
    baseline = [float(row["flash_pair_ms"]) for row in usable]
    joint = [float(row["joint_ms"]) for row in usable]
    acceptance_equal = sum(int(row["acceptance_delta"]) == 0 for row in report["cycles"])
    report["summary"] = {
        "cycles": len(report["cycles"]),
        "discarded_initial_cycles": args.discard_cycles,
        "timed_cycles": len(usable),
        "flash_regular_plus_verify": _timing(baseline),
        "relay_joint_shadow": _timing(joint),
        "aggregate_speedup": sum(baseline) / sum(joint),
        "median_state_speedup": statistics.median(row["state_speedup"] for row in usable),
        "official_proposals": sum(int(row["search"]) for row in report["cycles"]),
        "official_accepted": sum(int(row["official_accepted"]) for row in report["cycles"]),
        "relay_accepted": sum(int(row["relay"]["accepted"]) for row in report["cycles"]),
        "acceptance_equal_cycles": acceptance_equal,
        "acceptance_equal_fraction": acceptance_equal / len(report["cycles"]),
        "cache_immutability_checked": cache_checked[0],
        "claim": (
            "Measured state-level joint-forward preflight. Official generation is unchanged; "
            "no end-to-end Relay latency or benchmark-accuracy claim."
        ),
    }
    write_json(args.output / "diagnostic.json", report)
    (args.output / "complete").write_text("OK\n")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()

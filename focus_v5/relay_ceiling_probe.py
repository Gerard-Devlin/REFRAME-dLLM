"""Read-only row-work ceiling for a joint cache+verify Relay transition.

The probe runs the pinned official Flash generator on six already-used
development prompts.  It records real Transformer query-row counts and
acceptance decisions without changing inputs, masks, caches, or commits.  The
reported 96-row Relay number is an algorithmic screening ceiling, not measured
latency or an accuracy result.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import torch

from focus_v5.cost_model import Cycle, joint_row_ceiling


def summarize(records: list[dict], block: int = 32) -> dict:
    if not records:
        raise ValueError("no verification cycles recorded")
    cycles = [Cycle(
        regular_rows=int(row["regular_rows"]),
        verify_rows=int(row["verify_rows"]),
        accepted=int(row["accepted"]),
        next_regular_contains_accepted=int(row["accepted"]),
    ) for row in records]
    result = joint_row_ceiling(cycles, block)
    result.update(
        proposals=sum(row["search"] for row in records),
        accepted=sum(row["accepted"] for row in records),
        mean_regular_rows=sum(row["regular_rows"] for row in records) / len(records),
        min_regular_rows=min(row["regular_rows"] for row in records),
        max_regular_rows=max(row["regular_rows"] for row in records),
        all_verify_rows=sorted(set(row["verify_rows"] for row in records)),
        claim="Row-work ceiling only; no Relay generation, latency, or quality claim.",
    )
    return result


@torch.no_grad()
def main() -> None:
    from focus_dllm.common import sha256, write_json
    from focus_dllm.llada_common import MODEL_ID, REVISION, prompt_ids
    from focus_dllm.tuning.competitors import (
        generation_prompt,
        load_external,
        load_model,
        select_samples,
    )
    from focus_dllm.tuning.flash_readout import suppress_official_prints
    from focus_dllm.tuning.gpu_contract import check_binding
    from focus_dllm.tuning.verification_flow import observe_cumulative

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--datasets", type=Path, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)

    binding = check_binding(required=True)
    cls, external, adaptation = load_external(args.third_party, "flash_verify")
    official_file = Path(inspect.getsourcefile(inspect.unwrap(external)))
    official_hash = sha256(official_file)
    model, tokenizer = load_model(SimpleNamespace(method="flash_verify"), cls)
    torch.set_num_threads(1)
    torch.manual_seed(1234)
    task_paths = dict(zip(("humaneval", "mbpp", "math"), args.datasets))
    report = {
        "model": MODEL_ID,
        "revision": REVISION,
        "binding": binding,
        "official_generate_sha256": official_hash,
        "implementation": {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")},
        "datasets": {name: sha256(path) for name, path in task_paths.items()},
        "adaptation": adaptation,
        "configuration": {"length": 256, "block": 32, "threshold": .9, "gamma": .8,
                          "track_num": 4, "mask_num": 4, "prompts_per_task": 2},
        "prompts": [],
        "cycles": [],
        "scope": "Read-only official Flash trajectory on reused development prompts. No gold, tests, solutions, future tokens, or teacher state enter generation.",
    }
    write_json(args.output / "diagnostic.json", report)

    for task, path in task_paths.items():
        for sample in select_samples(path, 2, 0):
            ident = str(sample.get("id", sample.get("task_id")))
            ids = prompt_ids(tokenizer, generation_prompt(sample), task, preformatted=True)
            prompt = torch.tensor(ids, device=model.device)
            events: list[dict] = []
            prompt_cycles: list[dict] = []
            pending_regular = [None]

            def pre_hook(_model, inputs, kwargs):
                lengths = kwargs.get("lengths")
                if lengths is None:
                    raise AssertionError("unexpected non-Flash model call")
                kind = "verify" if bool(lengths[-1]) else "regular"
                event = {"kind": kind, "rows": int(inputs[0].shape[1])}
                events.append(event)
                if kind == "regular":
                    if pending_regular[0] is not None:
                        raise AssertionError("two regular calls without verification")
                    pending_regular[0] = event
                else:
                    if pending_regular[0] is None:
                        raise AssertionError("verification without a regular call")
                    search = int((inputs[0][0] == 126336).sum().item())
                    prompt_cycles.append({
                        "task": task,
                        "id": ident,
                        "cycle": len(prompt_cycles),
                        "regular_rows": pending_regular[0]["rows"],
                        "verify_rows": event["rows"],
                        "search": search,
                        "accepted": 0 if search == 0 else None,
                    })
                    pending_regular[0] = None

            def observe(probability, _draft, _positions, _top1, gamma):
                if not prompt_cycles or prompt_cycles[-1]["accepted"] is not None:
                    raise AssertionError("probability callback without a pending non-empty verification")
                cumulative = probability.detach().double().cumprod(0)
                accepted = int((cumulative >= float(gamma)).sum().item())
                if prompt_cycles[-1]["search"] != int(probability.numel()):
                    raise AssertionError("verification search count changed")
                prompt_cycles[-1]["accepted"] = accepted

            traced = observe_cumulative(external, observe)
            handle = model.register_forward_pre_hook(pre_hook, with_kwargs=True)
            responses, steps = [None], [0]
            try:
                with suppress_official_prints():
                    traced(model, [prompt], [len(ids)], 1, responses, steps,
                           gen_length=256, block_length=32, threshold=.9, gamma=.8,
                           track_num=4, mask_num=4, verify=True,
                           tokenizer=tokenizer, stop_tokens=[])
                torch.cuda.synchronize()
            finally:
                handle.remove()
            if (pending_regular[0] is not None or len(events) != 2 * len(prompt_cycles)
                    or any(row["accepted"] is None for row in prompt_cycles)):
                raise AssertionError(f"unpaired model calls: events={len(events)} cycles={len(prompt_cycles)}")
            report["cycles"].extend(prompt_cycles)
            report["prompts"].append({
                "task": task,
                "id": ident,
                "prompt_tokens": len(ids),
                "cycles": len(prompt_cycles),
                "model_calls": len(events),
                "official_iterations": int(steps[0]),
                "text_sha256": hashlib.sha256(responses[0].encode()).hexdigest(),
            })
            report["summary"] = summarize(report["cycles"])
            write_json(args.output / "diagnostic.json", report)
            print(json.dumps({"prompts": len(report["prompts"]), **report["summary"]}), flush=True)

    if sha256(official_file) != official_hash:
        raise AssertionError("official Flash source changed")
    report["summary"] = summarize(report["cycles"])
    write_json(args.output / "diagnostic.json", report)
    (args.output / "complete").write_text("OK\n")


if __name__ == "__main__":
    main()

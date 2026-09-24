"""Full native trajectories and a hindsight bound for isolated action replay.

No gate, no training, no online skipping. Bounds apply ONLY to the explicitly
supported same-subblock, no-EOS, read-only-cache calls of the pinned decoder.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
import torch.distributed as dist

from .collect import MODEL_ID, REVISION, CODE_HASH, distributed, barrier
from .data import load_prompts, sha256, write_json
from .teacher import active_mask


def action(tokens, confidence, eligible, threshold):
    """Exact greedy threshold + forced argmax; preserve native confidence dtype."""
    scores = torch.where(eligible, confidence, -torch.inf)
    chosen = (scores > threshold) & eligible
    if eligible.any():
        chosen[scores.argmax()] = True
    return [(int(i), int(tokens[i])) for i in chosen.nonzero().flatten()]


def max_independent(weights, allowed):
    """Maximum saved time on a path: two consecutive calls cannot both be skipped."""
    if len(weights) != len(allowed) or any(w < 0 for w in weights):
        raise ValueError("Nonnegative matching call weights required")
    values = [0.0] * (len(weights) + 1)
    take = [False] * len(weights)
    for i, (weight, legal) in enumerate(zip(weights, allowed)):
        yes = values[max(0, i-1)] + weight
        if legal and yes > values[i]:
            values[i+1], take[i] = yes, True
        else:
            values[i+1] = values[i]
    selected, i = [], len(weights)-1
    while i >= 0:
        if take[i]:
            selected.append(i)
            i -= 2
        else:
            i -= 1
    return values[-1], list(reversed(selected))


def call_kind(ids, kwargs, block_size):
    normal = (ids.shape == (1, block_size)
              and kwargs.get("update_past_key_values") is False
              and not kwargs.get("use_block_cache", False))
    return "denoise" if normal else "prefill_or_cache_write"


def fingerprint(ids):
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def classify_replay(previous, current, mask_id, stop_id):
    """Only actual adjacent calls with equal contexts and an exact prior reveal."""
    if current["kind"] != "denoise":
        return "cache_or_prefill"
    if previous is None or previous["kind"] != "denoise":
        return "no_previous_denoise"
    if previous["subblock"] != current["subblock"]:
        return "subblock_boundary"
    if not previous.get("action_verified", False):
        raise AssertionError("Previous predicted action was not verified against native state")
    if (stop_id in previous["state"] or stop_id in current["state"]
            or any(token in (mask_id, stop_id) for _, token in previous["action"] + current["action"])):
        return "eos_or_mask_proposal"
    return "supported"


class CallTimer:
    """Event timing pass. GPU input clones are OUTSIDE the forward event interval.

    Events resolve only after generation; no per-forward host synchronization.
    The separate unwrapped pass supplies the end-to-end latency denominator.
    """

    def __init__(self, model, block_size=32):
        self.model, self.block_size = model, block_size
        self.calls = []

    def __enter__(self):
        self.original = self.model.forward
        self.model.forward = self.forward
        return self

    def __exit__(self, *_):
        self.model.forward = self.original

    def forward(self, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        state = ids.detach().clone()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = self.original(*args, **kwargs)
        end.record()
        self.calls.append((call_kind(ids, kwargs, self.block_size), state, start, end))
        return result

    def finish(self):
        torch.cuda.synchronize()
        return [dict(kind=kind, state_hash=fingerprint(state[0].cpu().tolist()),
                     forward_seconds=start.elapsed_time(end)/1000)
                for kind, state, start, end in self.calls]


class ActionObserver:
    """Observe the actual native sampler, retaining EVERY call, not a reservoir.

    The next native input verifies each recorded action. A last action without
    a next input is conservatively not skippable (e.g. EOS termination).
    """

    def __init__(self, model, threshold=0.9, block_size=32, small_block_size=8,
                 mask_id=151665, stop_id=151645):
        self.model, self.threshold = model, threshold
        self.block_size, self.small = block_size, small_block_size
        self.mask_id, self.stop_id = mask_id, stop_id
        self.calls = []

    def __enter__(self):
        self.original_forward = self.model.forward
        self.original_sample = self.model.sample_with_top_p
        self.model.forward, self.model.sample_with_top_p = self.forward, self.sample
        return self

    def __exit__(self, *_):
        self.model.forward = self.original_forward
        self.model.sample_with_top_p = self.original_sample

    def forward(self, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        state = ids[0].detach().cpu().clone()
        if self.calls and self.calls[-1]["kind"] == "denoise":
            previous = self.calls[-1]
            if "action" not in previous:
                raise AssertionError("Normal call did not execute one native sampling action")
            expected = list(previous["state"])
            for position, token in previous["action"]:
                expected[position] = token
            if state.tolist() != expected:
                raise AssertionError("Reconstructed action differs from next actual native input")
            previous["action_verified"] = True
        kind = call_kind(ids, kwargs, self.block_size)
        row = dict(index=len(self.calls), kind=kind, state=state.tolist(),
                   state_hash=fingerprint(state.tolist()), action_verified=False)
        if kind == "denoise":
            eligible = active_mask(state[None], self.mask_id, self.small)[0]
            if not eligible.any():
                raise AssertionError("Denoise call with no remaining MASK")
            row["subblock"] = int(eligible.nonzero()[0]) // self.small
        self.calls.append(row)
        return self.original_forward(*args, **kwargs)

    def sample(self, logits, top_p=0.95, temperature=0):
        if temperature != 0:
            raise ValueError("Oracle only supports the deterministic native greedy decoder")
        tokens, probs = self.original_sample(logits, top_p=top_p, temperature=temperature)
        current = self.calls[-1]
        if current["kind"] != "denoise" or "action" in current:
            raise AssertionError("Unexpected native sampler placement")
        if tokens.shape != (1, self.small):
            raise AssertionError("Pinned native sampler no longer uses one subblock")
        start = current["subblock"] * self.small
        eligible = torch.tensor(current["state"][start:start+self.small]) == self.mask_id
        confidence = probs.gather(-1, tokens.unsqueeze(-1)).squeeze(-1)[0].detach().cpu()
        values = tokens[0].detach().cpu()
        selected = action(values, confidence, eligible, self.threshold)
        current.update(action=[(p+start, t) for p, t in selected],
                       tokens=values.tolist(), confidence=confidence.float().tolist(),
                       confidence_dtype=str(confidence.dtype))
        previous = self.calls[-2] if len(self.calls) > 1 else None
        reason = classify_replay(previous, current, self.mask_id, self.stop_id)
        current["reason"] = reason
        if reason == "supported":
            # Restore native dtype: a BF16 confidence comparison against threshold
            # is NOT necessarily the same operation after promotion to float32.
            old_conf = torch.tensor(previous["confidence"], dtype=confidence.dtype)
            replay = action(torch.tensor(previous["tokens"]), old_conf, eligible, self.threshold)
            current["replay_action"] = [(p+start, t) for p, t in replay]
            current["action_equal"] = current["replay_action"] == current["action"]
        return tokens, probs

    def finish(self):
        for row in self.calls:
            row.setdefault("reason", "cache_or_prefill")
            if row["kind"] == "denoise" and not row["action_verified"]:
                row["reason"] = "unverified_terminal_action"
            row["legal_equal"] = (row["reason"] == "supported" and row.get("action_equal", False))
        return self.calls


def summarize(calls, native_seconds):
    weights = [row["forward_seconds"] for row in calls]
    allowed = [row["legal_equal"] for row in calls]
    saved, indices = max_independent(weights, allowed)
    normal = [row["kind"] == "denoise" for row in calls]
    absolute_saved, _ = max_independent(weights, normal)
    forward_seconds = sum(weights)
    # A separate timing pass cannot be combined into a trustworthy cost estimate
    # when measured forwards alone exceed unwrapped total latency.
    timing_valid = 0 <= saved < native_seconds and forward_seconds <= native_seconds
    return dict(calls=len(calls), denoise_calls=sum(normal), supported_calls=sum(
                row["reason"] == "supported" for row in calls),
                equal_supported_calls=sum(allowed), selected_calls=len(indices),
                selected_indices=indices, native_seconds=native_seconds,
                all_forward_seconds=forward_seconds,
                denoise_forward_seconds=sum(w for w,n in zip(weights,normal) if n),
                saved_forward_seconds=saved,
                saved_native_fraction=saved/native_seconds if timing_valid else None,
                zero_overhead_modeled_speedup=native_seconds/(native_seconds-saved) if timing_valid else None,
                all_denoise_alternating_ceiling_seconds=absolute_saved,
                timing_valid=timing_valid, exclusions=dict(Counter(row["reason"] for row in calls)),
                overhead_sensitivity=[dict(gate_ms=ms,
                    modeled_speedup=native_seconds/(native_seconds-saved+(sum(normal)-len(indices))*ms/1000)
                    if timing_valid else None) for ms in (0.1, 0.5, 0.85)],
                note="Hindsight same-subblock no-EOS oracle. Saves forward only; native commit work remains. "
                     "Sensitivity charges a gate on every retained denoise call, no replay overhead. "
                     "No learned gate, online skipping or measured speedup.")


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()
    if min(args.prompts,args.repeats) < 1 or args.max_new_tokens < 64 or args.max_new_tokens % 32 or not 0 < args.threshold <= 1:
        parser.error("Positive sizes/repeats, generation multiple of 32 >=64, threshold in (0,1]")
    source = json.loads((args.data/"manifest.json").read_text())
    if source["revision"] != REVISION:
        raise ValueError("Prepared tokenizer revision mismatch")
    jobs = load_prompts(args.data, 1, args.prompts, args.seed)["heldout"]
    rank, world, device = distributed()
    if world > len(jobs):
        raise ValueError("Need at least one prompt per GPU")
    if rank == 0:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Oracle output must be new/empty")
        args.output.mkdir(parents=True, exist_ok=True)
    barrier()
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    snapshot = Path(snapshot_download(MODEL_ID, revision=REVISION, local_files_only=True))
    if sha256(snapshot/"modeling.py") != CODE_HASH:
        raise ValueError("Pinned official model code mismatch")
    model = AutoModelForCausalLM.from_pretrained(snapshot, trust_remote_code=True,
                local_files_only=True, dtype=torch.bfloat16).to(device).eval()
    model.requires_grad_(False)
    options = dict(max_new_tokens=args.max_new_tokens, block_size=32, small_block_size=8,
                   threshold=args.threshold, temperature=0, use_block_cache=False)
    warm_ids = torch.tensor([jobs[rank]["ids"]],device=device)
    model.generate(warm_ids.clone(), **{**options,"max_new_tokens":64})
    torch.cuda.synchronize()
    rows = []
    for index in range(rank,len(jobs),world):
        row = jobs[index]
        ids = torch.tensor([row["ids"]],device=device)
        native_times, expected = [], None
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            begin = time.perf_counter()
            output = model.generate(ids.clone(), **options)
            torch.cuda.synchronize()
            native_times.append(time.perf_counter()-begin)
            if expected is not None and not torch.equal(expected,output):
                raise AssertionError("Repeated native greedy outputs differ; cannot infer exact oracle")
            expected = output
        begin = time.perf_counter()
        with CallTimer(model) as timer:
            timed_output = model.generate(ids.clone(), **options)
        times = timer.finish()
        timed_wall = time.perf_counter()-begin
        begin = time.perf_counter()
        with ActionObserver(model,threshold=args.threshold) as observer:
            observed_output = model.generate(ids.clone(), **options)
        torch.cuda.synchronize()
        observed_wall = time.perf_counter()-begin
        calls = observer.finish()
        if not torch.equal(expected,timed_output) or not torch.equal(expected,observed_output):
            raise AssertionError("Observation/timing changed native output")
        if len(calls) != len(times):
            raise AssertionError("Observation changed forward count")
        for call, timing in zip(calls,times):
            if call["state_hash"] != timing["state_hash"] or call["kind"] != timing["kind"]:
                raise AssertionError("Observation changed forward inputs/cache role")
            call["forward_seconds"] = timing["forward_seconds"]
        result = summarize(calls,statistics.median(native_times))
        result.update(prompt_id=row["id"],native_repeats_seconds=native_times,
                      timed_pass_wall_seconds=timed_wall, observed_wall_seconds=observed_wall,
                      same_tokens_and_forward_inputs=True,
                      generated_tokens=expected.shape[-1]-ids.shape[-1])
        write_json(args.output/f"prompts/{row['id']}.json",dict(summary=result,calls=calls))
        rows.append(result)
        print(f"rank={rank} prompt={index+1}/{len(jobs)} calls={len(calls)} "
              f"legal_equal={result['equal_supported_calls']} selected={result['selected_calls']} "
              f"modeled_speedup={result['zero_overhead_modeled_speedup']}",flush=True)
    write_json(args.output/f"rank_{rank}.json",rows)
    barrier()
    if rank == 0:
        all_rows = [row for worker in range(world)
                    for row in json.loads((args.output/f"rank_{worker}.json").read_text())]
        if len(all_rows) != len(jobs) or len({r['prompt_id'] for r in all_rows}) != len(jobs):
            raise AssertionError("Missing or duplicate prompts")
        native = sum(r['native_seconds'] for r in all_rows)
        saved = sum(r['saved_forward_seconds'] for r in all_rows)
        valid = all(r['timing_valid'] for r in all_rows)
        summary = dict(status="complete",model=MODEL_ID,revision=REVISION,
            scope="Same-subblock only; no EOS/cache/prefill skips; no adjacent skips; perfect future oracle",
            prompts=len(all_rows),native_seconds=native,saved_forward_seconds=saved,
            selected_calls=sum(r['selected_calls'] for r in all_rows),
            total_calls=sum(r['calls'] for r in all_rows),
            timing_valid=valid,zero_overhead_modeled_speedup=native/(native-saved) if valid else None,
            saved_native_fraction=saved/native if valid else None,
            reaches_1_5x_cost_target=(native/(native-saved)>=1.5) if valid else None,
            learned_identifiability_measured=False,online_speedup_measured=False,
            settings=vars(args)|dict(data=str(args.data),output=str(args.output),world_size=world),
            implementation_sha256={name:sha256(Path(__file__).parent/name)
                for name in ('oracle.py','teacher.py','collect.py','data.py')},
            source_sha256={name:sha256(args.data/name) for name in ('manifest.json','heldout.json')},
            results=all_rows,
            note="Modeled savings of separately timed native forwards, not a deployed speedup. "
                 "No train/validation samples treated as independent prompts. "
                 "Below target rejects only this restricted one-skip policy, not all selective computation.")
        write_json(args.output/'summary.json',summary)
        print(json.dumps({k:v for k,v in summary.items() if k not in ('results','settings','implementation_sha256','source_sha256')}),flush=True)
    barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()

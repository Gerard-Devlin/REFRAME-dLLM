"""Free-running GSM8K accuracy / total latency; no gold tokens in generation."""
import argparse
from decimal import Decimal, InvalidOperation
import json
import math
from pathlib import Path
import re
import time
import torch
from .common import digest, manifest, snapshot, write_json
from .codec import Codec
from .model import Model, clean_mask, load_adapter


def answer(text, gold=False):
    if gold:
        text = text.split("####")[-1]
    else:
        explicit = re.findall(r"####\s*([-+]?\d[\d,]*(?:\.\d+)?)", text)
        boxed = re.findall(r"\\boxed\{\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*\}", text)
        values = explicit or boxed or re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
        if not values:
            return None
        text = values[-1]
    try:
        return str(Decimal(text.strip().replace(",", "")).normalize())
    except InvalidOperation:
        return None


def prompt_ids(tok, question):
    return tok.apply_chat_template([
        {"role": "user", "content": question + "\nExplain your reasoning and end with #### followed by the final number."}
    ], tokenize=True, add_generation_prompt=True)


@torch.no_grad()
def generate(model, prompt, codec, mask_id, eos_id, rounds, max_new_tokens):
    """Shared fixed-budget confidence sampler, not official v2's threshold sampler.

    Complete blocks are committed as original text. Logits at i-1 predict i.
    Boundary tokens are seeded by the preceding clean block (counted forward).
    The first partial prompt block is not cached until it is completed.
    """
    if not prompt or rounds < 1 or max_new_tokens < 1:
        raise ValueError("Nonempty prompt and positive budgets required")
    device = next(model.parameters()).device
    block = codec.block_size
    prefix = len(prompt)
    output = torch.tensor([prompt], device=device)
    complete = prefix // block * block
    cache, seed = None, None
    calls = dict(prefill=0, denoise=0, clean_encode=0)
    if complete:
        x = output[:, :complete]
        logits, cache = model(x, torch.arange(complete, device=device)[None],
                              clean_mask(complete, block, device), cache=True)
        calls["prefill"] += 1
        logits[..., mask_id] = -torch.inf
        seed = logits[:, -1].argmax(-1)
    start = complete
    while output.shape[1] - prefix < max_new_tokens:
        raw = torch.full((1, block), mask_id, device=device, dtype=torch.long)
        known = output.shape[1] - start
        raw[:, :known] = output[:, start:]
        if known == 0:
            if seed is None:
                raise ValueError("Missing boundary seed")
            raw[:, 0] = seed
        # Prompt pairs are unchanged; unknown response masks stay masks.
        z = codec(raw, prefix, offset=start)
        unknown_initial = int((z == mask_id).sum())
        positions = torch.arange(start, start + block, device=device)[None]
        mask = clean_mask(block, block, device, past=start)
        for step in range(min(rounds, unknown_initial)):
            remaining = (z[0] == mask_id).nonzero().flatten()
            if not len(remaining):
                break
            if (remaining == 0).any():
                raise AssertionError("Block-first token must be seeded")
            logits, _ = model(z, positions, mask, past=cache)
            calls["denoise"] += 1
            scores = logits[0, remaining - 1].float()
            scores[:, mask_id] = -torch.inf
            probs = scores.softmax(-1)
            confidence, proposed = probs.max(-1)
            # Cumulative equal quota, ties deterministically follow position.
            target = math.ceil(unknown_initial * (step + 1) / min(rounds, unknown_initial))
            count = target - (unknown_initial - len(remaining))
            chosen = torch.argsort(confidence, descending=True, stable=True)[:count]
            z[0, remaining[chosen]] = proposed[chosen]
        if (z == mask_id).any():
            raise AssertionError("Unfilled block")
        raw = codec(z, prefix, offset=start)
        output = torch.cat((output[:, :start], raw), 1)
        generated = output[0, prefix:prefix + max_new_tokens]
        stop = (generated == eos_id).nonzero().flatten()
        if len(stop):
            return generated[:int(stop[0])].tolist(), calls
        if len(generated) >= max_new_tokens:
            return generated.tolist(), calls
        # Re-encode COMPLETED ORIGINAL text, not noisy / relation KV states.
        logits, cache = model(raw, positions, mask, past=cache, cache=True)
        calls["clean_encode"] += 1
        logits[..., mask_id] = -torch.inf
        seed = logits[:, -1].argmax(-1)
        start += block
    return output[0, prefix:prefix + max_new_tokens].tolist(), calls


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--official", action="store_true")
    p.add_argument("--split", choices=["dev", "test"], default="dev")
    p.add_argument("--limit", type=int, default=32)
    p.add_argument("--rounds", default="2,4,8,16")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--threshold", type=float, default=.9)
    args = p.parse_args()
    m = manifest(args.data)
    from .preflight import require_gate, implementation_hashes
    require_gate(args.data)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError("Evaluation output must be new")
    args.output.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer, AutoModelForCausalLM
    root = snapshot()
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    spec = json.loads((args.data / "codec.json").read_text())
    metadata = None
    if args.official:
        if args.checkpoint:
            raise ValueError("Official baseline is unadapted")
        model = AutoModelForCausalLM.from_pretrained(root, trust_remote_code=True,
                local_files_only=True, dtype=torch.bfloat16).cuda().eval()
        schedules = ["official-threshold"]
        arm = "official-native"
    else:
        model = Model.load(root)
        if args.checkpoint:
            metadata = load_adapter(model, args.checkpoint)
            if metadata["data_hash"] != digest(args.data / "manifest.json"):
                raise ValueError("Checkpoint data differs")
            if metadata["status"] != "complete":
                raise ValueError("Incomplete training budget; resume first")
            if metadata["implementation"] != implementation_hashes():
                raise ValueError("Checkpoint was trained with a different implementation")
        arm = metadata["arm"] if metadata else "unadapted-token"
        codec = Codec(spec, identity=arm != "relation").cuda()
        model.eval()
        schedules = [int(s) for s in args.rounds.split(",")]
        if not schedules or min(schedules) < 1:
            raise ValueError("Invalid rounds")
    samples = json.loads((args.data / f"gsm8k_{args.split}.json").read_text())
    samples = samples[:args.limit]
    if not samples:
        raise ValueError("Empty evaluation")
    def run(ids, schedule):
        if args.official:
            calls = [0]
            handle = model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
            try:
                x = torch.tensor([ids], device="cuda")
                out = model.generate(x, tokenizer=tok, max_new_tokens=args.max_new_tokens,
                    block_size=m["block_size"], small_block_size=8,
                    threshold=args.threshold, temperature=0., use_block_cache=False)
                return out[0, len(ids):].tolist(), {"total": calls[0]}
            finally:
                handle.remove()
        return generate(model, ids, codec, m["mask_id"], m["eos_id"], schedule, args.max_new_tokens)
    # Same unrelated warmup, excluded from metrics, no evaluation labels used.
    warmup = prompt_ids(tok, "What is 1 plus 1?")
    results = {}
    for schedule in schedules:
        run(warmup, schedule)
        records = []
        for sample in samples:
            ids = prompt_ids(tok, sample["question"])
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            generated, calls = run(ids, schedule)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            text = tok.decode(generated, skip_special_tokens=True)
            pred, target = answer(text), answer(sample["answer"], gold=True)
            record = dict(id=sample["id"], prediction=text, extracted=pred, target=target,
                          correct=pred is not None and pred == target, seconds=seconds,
                          tokens=len(generated), calls=calls, peak_gib=torch.cuda.max_memory_allocated() / 2**30)
            records.append(record)
            with (args.output / f"samples_{schedule}.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(f"{arm} rounds={schedule} {len(records)}/{len(samples)} correct={record['correct']} seconds={seconds:.3f}", flush=True)
        times = sorted(x["seconds"] for x in records)
        results[str(schedule)] = dict(accuracy=sum(x["correct"] for x in records) / len(records),
            mean_seconds=sum(times) / len(times), p95_seconds=times[min(len(times)-1, math.ceil(.95*len(times))-1)],
            tokens_per_second=sum(x["tokens"] for x in records) / sum(times),
            mean_calls=sum(sum(x["calls"].values()) for x in records) / len(records))
    write_json(args.output / "summary.json", dict(arm=arm, split=args.split, examples=len(samples),
        ids=[s["id"] for s in samples], max_new_tokens=args.max_new_tokens, block_size=m["block_size"],
        data_hash=digest(args.data / "manifest.json"), checkpoint=metadata,
        gpu=torch.cuda.get_device_name(), torch=torch.__version__, results=results,
        scope="0-shot GSM8K, custom numeric extraction; exploratory dev if split=dev; not official lm-eval reproduction",
        sampler="official threshold, block cache only" if args.official else "shared fixed-round confidence quota, block cache only"))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()

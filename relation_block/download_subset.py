"""Server-side bounded Nemotron SFT streaming export; never fetches the full repo."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from .common import MODEL_ID, REVISION, digest, snapshot, write_json

DATASET = "nvidia/Llama-Nemotron-Post-Training-Dataset"


def normalize(row):
    messages = row.get("input")
    if isinstance(messages, str):
        messages = [{"role": "user", "content": messages}]
    if not isinstance(messages, list) or not messages or messages[-1].get("role") != "user":
        raise ValueError("Expected input conversation ending in a user message")
    result = []
    for m in messages:
        if m.get("role") not in ("system", "user", "assistant") or not isinstance(m.get("content"), str):
            raise ValueError("Unsupported message")
        result.append({"role": m["role"], "content": m["content"]})
    output = row.get("output")
    if not isinstance(output, str) or not output.strip():
        raise ValueError("Missing assistant output")
    # Keep the full reasoning + answer. Never strip <think> or cut off an answer.
    return result + [{"role": "assistant", "content": output}]


def tokenized(messages, tok, length):
    prefix = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
    if ids[:len(prefix)] != prefix or len(ids) <= len(prefix):
        raise ValueError("Template prefix mismatch")
    if len(prefix) > length // 2 or len(ids) > length:
        raise OverflowError("Whole response does not fit requested training window")
    return ids, prefix


def fingerprint(messages):
    return hashlib.sha256(json.dumps(messages, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tokens", type=int, default=100_000_000)
    p.add_argument("--length", type=int, default=2048)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--buffer", type=int, default=256)
    p.add_argument("--max-scan", type=int, default=1_000_000)
    p.add_argument("--reasoning", choices=["any", "on", "off"], default="any")
    args = p.parse_args()
    if args.tokens < 2 * args.length or args.buffer < 1 or args.max_scan < 1:
        raise ValueError("Invalid subset budget")
    root = snapshot(offline=True)  # Weights/tokenizer downloaded in a separate phase.
    from transformers import AutoTokenizer
    from huggingface_hub import HfApi
    from datasets import load_dataset
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / "request.json"
    settings = dict(dataset=DATASET, config="SFT", tokens=args.tokens, length=args.length,
                    seed=args.seed, buffer=args.buffer, max_scan=args.max_scan, reasoning=args.reasoning,
                    tokenizer_model=MODEL_ID, tokenizer_revision=REVISION)
    if request_path.exists():
        request = json.loads(request_path.read_text())
        if any(request[k] != v for k, v in settings.items()):
            raise ValueError("Existing subset has different settings; use a new output directory")
    else:
        # Persist the resolved dataset revision BEFORE reading records.
        request = dict(settings, revision=HfApi().dataset_info(DATASET).sha)
        write_json(request_path, request)
    print(json.dumps(request), flush=True)
    seen, records = set(), []
    mask_id = tok.convert_tokens_to_ids("|<MASK>|")
    for category in ("math", "code"):
        dest, info = out / f"{category}.jsonl", out / f"{category}.metadata.json"
        if dest.exists() and info.exists():
            meta = json.loads(info.read_text())
            if digest(dest) != meta["sha256"]:
                raise ValueError(f"Modified completed subset: {dest}")
            with dest.open(encoding="utf-8") as f:
                for line in f:
                    seen.add(json.loads(line)["id"])
            records.append(meta)
            print(f"Reuse completed {category}: {meta['tokens']} tokens", flush=True)
            continue
        stream = load_dataset(DATASET, "SFT", split=category, streaming=True,
                              revision=request["revision"])
        stream = stream.shuffle(seed=args.seed, buffer_size=args.buffer)
        budget = args.tokens // 2
        count = total = scanned = 0
        rejected = Counter()
        partial = out / f"{category}.jsonl.partial"
        # Completed categories are reusable. An interrupted category restarts
        # deterministically; no claim of byte-exact HTTP resume for JSON streams.
        with partial.open("w", encoding="utf-8") as f:
            for row in stream:
                scanned += 1
                if scanned > args.max_scan:
                    break
                if scanned % 1000 == 0:
                    print(json.dumps(dict(category=category, scanned=scanned, kept=count,
                                          tokens=total, rejected=dict(rejected))), flush=True)
                    f.flush()
                if args.reasoning != "any" and row.get("reasoning") != args.reasoning:
                    rejected["reasoning"] += 1
                    continue
                try:
                    messages = normalize(row)
                    ids, prefix = tokenized(messages, tok, args.length)
                except OverflowError:
                    rejected["too_long_whole_example"] += 1
                    continue
                except (ValueError, TypeError, KeyError):
                    rejected["format"] += 1
                    continue
                key = fingerprint(messages)
                if key in seen or mask_id in ids:
                    rejected["duplicate_or_literal_mask"] += 1
                    continue
                if total + len(ids) > budget:
                    break
                seen.add(key)
                value = dict(id=key, messages=messages, category=category, tokens=len(ids),
                             prefix_tokens=len(prefix), shuffled_stream_index=scanned - 1,
                             source_metadata={k: row.get(k) for k in
                                              ("reasoning", "generator", "license", "version", "system_prompt")})
                f.write(json.dumps(value, ensure_ascii=False) + "\n")
                count += 1
                total += len(ids)
        # Do not silently call an exhausted/filtered stream a full requested subset.
        if total < budget - args.length:
            raise RuntimeError(f"{category}: only {total}/{budget} tokens after {scanned} scanned; inspect length/filter settings. Partial file retained, no completion manifest.")
        partial.replace(dest)
        meta = dict(category=category, examples=count, tokens=total, scanned=scanned,
                    rejected=dict(rejected), file=dest.name, sha256=digest(dest))
        write_json(info, meta)
        records.append(meta)
    write_json(out / "subset_manifest.json", dict(request=request, categories=records,
        total_tokens=sum(r["tokens"] for r in records),
        note="Bounded shuffled streaming subset, not a uniform sample of the full corpus. Whole examples only; no truncation. Network bytes can exceed retained subset size."))
    print(f"Complete: {out / 'subset_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()

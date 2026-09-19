"""Server-side download and reproducible, train-only codec preparation."""
import argparse
from collections import Counter
import json
import hashlib
from pathlib import Path
import random
import torch
from .common import MODEL_ID, REVISION, digest, snapshot, write_json
from .codec import Codec, fit


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--download", action="store_true")
    p.add_argument("--source", type=Path, default=Path("v2/data/alpaca/train_conversation/train_52002.json"))
    p.add_argument("--source-dir", type=Path, help="Completed Nemotron export from download_subset")
    p.add_argument("--length", type=int, default=512)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args()
    if args.length % args.block_size or args.block_size % 2:
        raise ValueError("Length must be a multiple of even block size")
    if (args.data / "manifest.json").exists():
        raise ValueError("Completed data already exists; reuse it or select a new path")
    root = snapshot(offline=not args.download)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True, trust_remote_code=False)
    config = json.loads((root / "config.json").read_text())
    mask_id = tok.convert_tokens_to_ids("|<MASK>|")
    if mask_id is None or tok.convert_ids_to_tokens(mask_id) != "|<MASK>|":
        raise ValueError("Tokenizer lacks expected Fast-dLLM MASK")
    args.data.mkdir(parents=True, exist_ok=True)
    # GSM8K is only evaluation here. No fitting on GSM8K labels/questions.
    from datasets import load_dataset
    gsm = load_dataset("openai/gsm8k", "main")
    for split, name, limit in (("train", "gsm8k_dev", 128), ("test", "gsm8k_test", 1319)):
        write_json(args.data / (name + ".json"),
                   [{"id": f"{split}:{i}", **gsm[split][i]} for i in range(min(limit, len(gsm[split])))])
    gsm_questions = {r["question"].strip() for split in gsm.values() for r in split}
    if args.source_dir:
        subset_path = args.source_dir / "subset_manifest.json"
        subset = json.loads(subset_path.read_text())
        if subset["request"]["tokenizer_revision"] != REVISION:
            raise ValueError("Subset tokenizer revision differs from model")
        if subset["request"]["length"] != args.length:
            raise ValueError("Use the same length as subset export; no silent truncation")
        for part in subset["categories"]:
            if digest(args.source_dir / part["file"]) != part["sha256"]:
                raise ValueError("Subset file changed")
        def stream_source():
            for part in subset["categories"]:
                with (args.source_dir / part["file"]).open(encoding="utf-8") as f:
                    for line in f:
                        yield json.loads(line)
        source = stream_source()
        source_name, source_hash = str(args.source_dir), digest(subset_path)
    else:
        source = json.loads(args.source.read_text(encoding="utf-8"))["instances"]
        source_name, source_hash = str(args.source), digest(args.source)
    rows, seen = [], set()
    excluded = Counter()
    for item in source:
        messages = item["messages"]
        if len(messages) < 2 or messages[-1]["role"] != "assistant":
            excluded["format"] += 1
            continue
        if any(m["content"].strip() in gsm_questions for m in messages[:-1]):
            excluded["exact_gsm_question"] += 1
            continue
        prompt = tok.apply_chat_template(messages[:-1], tokenize=True, add_generation_prompt=True)
        full = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
        if full[:len(prompt)] != prompt or len(prompt) > args.length // 2 or len(full) <= len(prompt):
            excluded["prefix_or_length"] += 1
            continue
        if args.source_dir and len(full) > args.length:
            raise ValueError("Export/preparation tokenizer mismatch: whole example exceeds length")
        full = full[:args.length]
        if mask_id in full:
            excluded["literal_mask"] += 1
            continue
        key = tuple(full)
        if key in seen:
            excluded["duplicate"] += 1
            continue
        seen.add(key)
        prompt_hash = hashlib.sha256(json.dumps(prompt).encode()).hexdigest()
        rows.append({"ids": full, "prefix": len(prompt), "prompt_hash": prompt_hash})
    random.Random(args.seed).shuffle(rows)
    if len(rows) < 1024:
        raise ValueError("Too few eligible examples")
    heldout = rows[:256]
    heldout_prompts = {r["prompt_hash"] for r in heldout}
    train = [r for r in rows[256:] if r["prompt_hash"] not in heldout_prompts]
    excluded["same_prompt_as_heldout"] = len(rows) - 256 - len(train)
    protected = set(tok.all_special_ids) | {mask_id}
    spec = fit(train, args.block_size, config["vocab_size"], protected)
    codec = Codec(spec)
    changed, eligible, max_spread = 0, 0, 0
    # Exact heldout invertibility, change coverage and a bounded error test.
    for row in heldout:
        x = torch.tensor([row["ids"]])
        z = codec(x, row["prefix"])
        if not torch.equal(codec(z, row["prefix"]), x):
            raise AssertionError("Codec round trip failed")
        changed += (z != x).sum().item()
        eligible += len(row["ids"]) - row["prefix"]
        corrupt = z.clone()
        corrupt[0, row["prefix"]] = (corrupt[0, row["prefix"]] + 1) % config["vocab_size"]
        max_spread = max(max_spread, (codec(corrupt, row["prefix"]) != x).sum().item())
    for name, value in (("train", train), ("heldout", heldout), ("codec", spec)):
        write_json(args.data / f"{name}.json", value)
    write_json(args.data / "codec_diagnostic.json", dict(
        response_change_fraction=changed / eligible, supported_anchors=len(spec["swaps"]),
        heldout_roundtrip=True, checked_single_error_max_spread=max_spread,
        warning="Coverage/invertibility only. Not proof of fewer denoising steps."))
    files = ["train.json", "heldout.json", "codec.json", "gsm8k_dev.json", "gsm8k_test.json"]
    m = dict(version=1, model_id=MODEL_ID, revision=REVISION, source=source_name,
             source_sha256=source_hash, length=args.length, block_size=args.block_size,
             mask_id=mask_id, pad_id=tok.pad_token_id, eos_id=tok.eos_token_id,
             train_examples=len(train), heldout_examples=len(heldout), seed=args.seed,
             train_original_tokens=sum(len(r["ids"]) for r in train),
             excluded=dict(excluded), files={f: digest(args.data / f) for f in files},
             scope=("Nemotron SFT math/code subset" if args.source_dir else "Alpaca feasibility adaptation")
                   + "; not reproduction of v2 full training recipe; exact-only contamination check")
    write_json(args.data / "manifest.json", m)
    print(json.dumps(m, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()

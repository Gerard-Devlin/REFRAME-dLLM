"""Bounded continuation diagnostics. Never modifies existing checkpoints or gates."""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
from .common import digest, manifest, snapshot, write_json
from .full_state import load_full_model, resolve_checkpoint
from .model import Model, clean_mask
from .codec import Codec
from .evaluate import generate, prompt_ids
from .preflight import require_gate, implementation_hashes


def repetition(text, width=4):
    """Fraction of whitespace n-gram occurrences beyond their first occurrence."""
    words = text.split()
    grams = [tuple(words[i:i + width]) for i in range(len(words) - width + 1)]
    return (len(grams) - len(set(grams))) / len(grams) if grams else 0.


def summarize_pairs(short, long):
    if [r["id"] for r in short] != [r["id"] for r in long]:
        raise ValueError("Length comparison must use the same questions in the same order")
    n = len(short)
    if not n:
        raise ValueError("Empty comparison")
    return dict(examples=n,
                wrong_to_correct=sum(not a["correct"] and b["correct"] for a, b in zip(short, long)),
                correct_to_wrong=sum(a["correct"] and not b["correct"] for a, b in zip(short, long)),
                capped_to_correct=sum(a["length_capped"] and not a["correct"] and b["correct"] for a, b in zip(short, long)),
                short_accuracy=sum(a["correct"] for a in short) / n,
                long_accuracy=sum(a["correct"] for a in long) / n,
                short_capped=sum(a["length_capped"] for a in short) / n,
                long_capped=sum(a["length_capped"] for a in long) / n,
                short_repeated_4gram=sum(repetition(a["prediction"]) for a in short) / n,
                long_repeated_4gram=sum(repetition(a["prediction"]) for a in long) / n)


def validate_checkpoint(path, data, arm):
    path = resolve_checkpoint(path)
    meta = json.loads((path / "metadata.json").read_text())
    if meta.get("arm") != arm or meta.get("data_hash") != digest(data / "manifest.json"):
        raise ValueError(f"Checkpoint arm/data mismatch: {path}")
    if meta.get("implementation") != implementation_hashes():
        raise ValueError("Diagnostic requires the unchanged implementation used for training")
    if meta.get("evaluation_data_hash") != digest(data / "gsm8k_dev_full.json"):
        raise ValueError("Checkpoint dev data differs")
    return path


@torch.no_grad()
def reconstruction(model, codec, rows, config, seed=1234):
    """Instrument the EXISTING loss, rather than defining another masking objective."""
    from .train import batch, loss
    length = config["length"]
    target = None
    boundary = None
    offset = 0
    counts = {}
    def before(_module, _args, kwargs):
        nonlocal target, boundary, offset
        target = kwargs["targets"]
        boundary = kwargs["select"][1] >= length
        offset = 0
    def after(_module, _args, logits):
        nonlocal offset
        gold = target[offset:offset + len(logits)]
        edges = boundary[offset:offset + len(logits)]
        eos = gold == config["eos_id"]
        good = logits.argmax(-1) == gold
        for name, mask in (("all", torch.ones_like(eos)), ("eos", eos),
                           ("boundary", edges), ("ordinary", ~eos & ~edges)):
            counts[name][0] += int((good & mask).sum())
            counts[name][1] += int(mask.sum())
        offset += len(logits)
    first = model.register_forward_pre_hook(before, with_kwargs=True)
    second = model.lm_head.register_forward_hook(after)
    results = {}
    try:
        for probability in (.25, .5, .75):
            counts = {k: [0, 0] for k in ("all", "eos", "boundary", "ordinary")}
            ce, denominator = 0., 0
            for index in range(len(rows)):
                raw, response, prefix = batch(rows, [index], length, config["pad_id"], "cuda")
                rng = torch.Generator().manual_seed(seed + index)
                noise = torch.rand(1, length, generator=rng).cuda()
                probabilities = torch.full((1, length // config["block_size"]), probability, device="cuda")
                value = loss(model, raw, response, prefix, codec, noise, probabilities,
                             config["mask_id"], config["block_size"])
                if offset != len(target):
                    raise AssertionError("Head instrumentation missed targets")
                weight = int(response.sum())
                ce += float(value) * weight
                denominator += weight
            results[str(probability)] = dict(complementary_probabilities=[probability, 1 - probability], cross_entropy=ce / denominator,
                categories={k: dict(correct=v[0], total=v[1], accuracy=v[0] / v[1] if v[1] else None)
                            for k, v in counts.items()})
    finally:
        first.remove(); second.remove()
    return results


def length_diagnostic(args):
    config = manifest(args.data)
    spec = json.loads((args.data / "codec.json").read_text())
    checkpoints = dict(original=None,
        token=validate_checkpoint(args.source / "token/train", args.data, "token"),
        relation=validate_checkpoint(args.source / "relation/train", args.data, "relation"))
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = env.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0]
    result = {}
    for arm, checkpoint in checkpoints.items():
        for cap in (512, 1024):
            destination = args.output / arm / f"length_{cap}"
            command = [sys.executable, "-u", "-m", "relation_block.evaluate", "--data", str(args.data),
                       "--output", str(destination), "--limit", str(args.limit), "--rounds", "16",
                       "--max-new-tokens", str(cap)]
            if checkpoint:
                command += ["--checkpoint", str(checkpoint)]
            subprocess.run(command, env=env, check=True)
        records = [[json.loads(line) for line in (args.output / arm / f"length_{cap}" / "samples_16.jsonl").read_text().splitlines()]
                   for cap in (512, 1024)]
        pair = summarize_pairs(*records)
        pair["timings"] = {str(cap): json.loads((args.output / arm / f"length_{cap}/summary.json").read_text())["results"]["16"]
                           for cap in (512, 1024)}
        if args.reconstruction_limit:
            model = load_full_model(checkpoint)[0] if checkpoint else Model.load(snapshot()).eval()
            codec = Codec(spec, identity=arm != "relation").cuda()
            rows = json.loads((args.data / "heldout.json").read_text())[:args.reconstruction_limit]
            pair["reconstruction"] = reconstruction(model, codec, rows, config)
            del model, codec
            gc.collect(); torch.cuda.empty_cache()
        result[arm] = pair
        write_json(args.output / "summary.json", dict(results=result, complete=False))
        print("LENGTH_DIAGNOSTIC", arm, json.dumps(pair), flush=True)
    write_json(args.output / "summary.json", dict(complete=True, results=result,
        implementation=implementation_hashes(), diagnostic_sha256=digest(Path(__file__)),
        data_hash=digest(args.data / "manifest.json"), heldout_hash=digest(args.data / "heldout.json"),
        note="Reconstruction measures this training objective, not official recipe parity. Relation/token CE are not directly comparable. Repeated whitespace 4-grams are a heuristic, not a failure label."))


def compare_weight_files(checkpoint, root):
    """Every state tensor, not a sample. Read large checkpoints through CPU mmap."""
    from safetensors import safe_open
    fp32 = torch.load(checkpoint / "model_fp32.pt", map_location="cpu", weights_only=True, mmap=True)
    bf16 = torch.load(checkpoint / "model_bf16.pt", map_location="cpu", weights_only=True, mmap=True)
    seen, failures = set(), []
    for shard in sorted(root.glob("model*.safetensors")):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for name in handle.keys():
                original = handle.get_tensor(name)
                seen.add(name)
                if name not in fp32 or name not in bf16:
                    failures.append(name + ": missing")
                    continue
                if not torch.equal(fp32[name], original.float()):
                    failures.append(name + ": FP32 changed")
                if not torch.equal(bf16[name], original.bfloat16()):
                    failures.append(name + ": BF16 changed")
    config = json.loads((root / "config.json").read_text())
    if config.get("tie_word_embeddings") and "lm_head.weight" not in seen:
        for state in (fp32, bf16):
            if not torch.equal(state["lm_head.weight"], state["model.embed_tokens.weight"]):
                failures.append("Tied head changed")
        seen.add("lm_head.weight")
    if set(fp32) != seen or set(bf16) != seen:
        failures.append("Unexpected state keys")
    return dict(pass_=not failures, tensors=len(seen), failures=failures)


@torch.no_grad()
def probes(model, tok, rows, config, spec):
    model.eval()
    codec = Codec(spec, identity=True).cuda()
    outputs, logits = [], []
    for row in rows:
        ids = prompt_ids(tok, row["question"])
        x = torch.tensor([ids], device="cuda")
        y = model(x, torch.arange(len(ids), device="cuda")[None], clean_mask(len(ids), config["block_size"], "cuda"))[0]
        logits.append(y[:, -1].cpu())
        generated, _ = generate(model, ids, codec, config["mask_id"], config["eos_id"], 16, 128)
        outputs.append(generated)
    return logits, outputs


def zero_diagnostic(args):
    from transformers import AutoTokenizer
    config = manifest(args.data)
    train = args.output / "train"
    command = [sys.executable, "-u", "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
               f"--nproc_per_node={args.world_size}", "-m", "relation_block.train", "--data", str(args.data),
               "--output", str(train), "--arm", "token", "--steps", "4", "--lr", "0", "--smoke",
               "--eval-every", "0", "--global-batch", str(args.global_batch), "--micro-batch", "1"]
    subprocess.run(command, check=True)
    checkpoint = resolve_checkpoint(train)
    root = snapshot()
    weights = compare_weight_files(checkpoint, root)
    write_json(args.output / "weights.json", weights)
    if not weights["pass_"]:
        raise AssertionError("Zero-LR changed weights; see weights.json")
    tok = AutoTokenizer.from_pretrained(root, local_files_only=True)
    rows = json.loads((args.data / "gsm8k_dev_full.json").read_text())[:8]
    spec = json.loads((args.data / "codec.json").read_text())
    model = Model.load(root).eval()
    before_logits, before_generation = probes(model, tok, rows, config, spec)
    del model
    gc.collect(); torch.cuda.empty_cache()
    model, _ = load_full_model(checkpoint)
    after_logits, after_generation = probes(model, tok, rows, config, spec)
    logits_equal = all(torch.equal(a, b) for a, b in zip(before_logits, after_logits))
    generation_equal = before_generation == after_generation
    maximum_error = max(float((a.float() - b.float()).abs().max()) for a, b in zip(before_logits, after_logits))
    passed = weights["pass_"] and logits_equal and generation_equal
    report = dict(pass_=passed, weights=weights, logits_equal=logits_equal, max_logit_error=maximum_error,
                  generation_equal=generation_equal, probe_ids=[r["id"] for r in rows],
                  before_generation=before_generation, after_generation=after_generation,
                  world_size=args.world_size, implementation=implementation_hashes(),
                  diagnostic_sha256=digest(Path(__file__)),
                  scope="Four real zero-LR full optimizer steps, all state tensors, eight prompt logits and 128-token generation probes. Not a quality or nonzero-LR training guarantee.")
    report["pass"] = report.pop("pass_")
    write_json(args.output / "summary.json", report)
    print("ZERO_LR", json.dumps(report), flush=True)
    if not passed:
        raise AssertionError("Zero-LR roundtrip/probes differ; see summary.json")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("mode", choices=["length", "zero"])
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path)
    p.add_argument("--limit", type=int, default=256)
    p.add_argument("--reconstruction-limit", type=int, default=32)
    p.add_argument("--world-size", type=int, default=6)
    p.add_argument("--global-batch", type=int, default=12)
    args = p.parse_args()
    require_gate(args.data)
    if args.output.exists():
        raise ValueError("Use a new diagnostic output directory")
    if args.mode == "length" and args.source is None:
        raise ValueError("Length diagnostic needs the completed full run via --source")
    if args.mode == "length" and torch.cuda.device_count() != 1:
        raise ValueError("Select exactly one GPU for length comparisons")
    if args.mode == "zero" and (args.world_size < 1 or torch.cuda.device_count() != args.world_size or args.global_batch % args.world_size):
        raise ValueError("Zero-LR diagnostic GPU count/global batch mismatch")
    if args.limit < 1 or args.reconstruction_limit < 0:
        raise ValueError("Invalid sample budget")
    args.output.mkdir(parents=True)
    if args.mode == "length":
        length_diagnostic(args)
    else:
        zero_diagnostic(args)


if __name__ == "__main__":
    main()

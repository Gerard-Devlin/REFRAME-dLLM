"""Local tiny-model smoke, real-model comparisons, or held-out oracle probes."""
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import time

import _bootstrap  # noqa: F401
import torch

from generate import generate_with_dual_cache, generate_with_prefix_cache
from model.configuration_llada import LLaDAConfig
from model.modeling_llada import LLaDAModelLM
from reframe_dllm.generate import generate_reframe, synchronize
from reframe_dllm.model import ReframeConfig
from reframe_dllm.oracle import NativeOracleWrapper, OracleProbe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tiny", action="store_true", help="Random tiny LLaDA; no downloads or quality claims")
    p.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    p.add_argument("--backend", choices=["torch", "flash"], default="torch")
    p.add_argument("--prompt", default="Janet has 12 apples and gives away 5. How many remain? Explain briefly.")
    p.add_argument("--prompts", type=Path, help="JSONL with id, prompt and optional until list; shared by all methods")
    p.add_argument("--chat-template", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--methods", default="native-dual,stale,shift,pair,materialize")
    p.add_argument("--gen-length", type=int, default=64)
    p.add_argument("--block-length", type=int, default=16)
    p.add_argument("--threshold", type=float, default=0.9)
    p.add_argument("--factor", type=float)
    p.add_argument("--pilots", type=int, default=8)
    p.add_argument("--refresh-blocks", type=int, default=2)
    p.add_argument("--max-pilot-error", type=float, default=0.25)
    p.add_argument("--ridge", type=float, default=1e-3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--limit", type=int, default=2)
    p.add_argument("--oracle", action="store_true", help="Native DualCache trajectory; no speed conclusions")
    p.add_argument("--audit-every", type=int, default=0, help="Read-only full-logit comparison every N decisions; not a timing run")
    p.add_argument("--output", type=Path, default=Path("reframe/results/smoke.jsonl"))
    args = p.parse_args()
    methods = args.methods.split(",")
    if set(methods) - {"native-dual", "native-prefix", "stale", "shift", "scale", "pair", "materialize"}:
        p.error("Unknown method")
    if min(args.gen_length, args.block_length, args.limit, args.repeats) < 1 or args.warmup < 0:
        p.error("Lengths, limit and repeats must be positive; warmup nonnegative")
    if args.gen_length % args.block_length:
        p.error("Generation length must be divisible by block length")
    if args.output.exists():
        p.error("Output exists; choose a new output path to avoid mixing runs")
    torch.manual_seed(1234)
    if args.device == "cpu":
        torch.set_num_threads(1)
    dtype = getattr(torch, args.dtype)
    tokenizer = None
    if args.tiny:
        cfg = LLaDAConfig(d_model=32, n_heads=4, n_layers=2, mlp_hidden_size=64,
                         vocab_size=128, embedding_size=128, block_type="llama",
                         activation_type="silu", rope=True, weight_tying=False,
                         max_sequence_length=256, attention_dropout=0.0,
                         residual_dropout=0.0, embedding_dropout=0.0,
                         flash_attention=args.backend == "flash", eos_token_id=126, pad_token_id=126)
        model = LLaDAModelLM(cfg, init_params=True).eval().to(device=args.device, dtype=dtype)
        mask_id = 127
        # Synthetic logits must never propose the reserved MASK ID.
        def suppress_mask(module, inputs, output):
            output[..., mask_id] = -1e4
            return output
        model.model.transformer.ff_out.register_forward_hook(suppress_mask)
    else:
        from transformers import AutoConfig, AutoTokenizer
        cfg = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
        cfg.flash_attention = args.backend == "flash"
        model = LLaDAModelLM.from_pretrained(args.model_path, config=cfg, torch_dtype=dtype,
                                            trust_remote_code=True).eval().to(args.device)
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
        mask_id = 126336
    if args.backend == "flash" and any(b.flash_attn_func is None for b in model.model.transformer.blocks):
        raise RuntimeError("flash requested but not available; refusing silent backend mismatch")
    requests = ([json.loads(line) for line in args.prompts.read_text(encoding="utf-8").splitlines() if line.strip()]
                if args.prompts else [dict(id="smoke", prompt=args.prompt)])[:args.limit]
    if not requests:
        p.error("No prompts")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metadata = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    metadata.update(torch=torch.__version__, cuda=torch.version.cuda,
                    torch_compile_disable=os.environ.get("TORCH_COMPILE_DISABLE", "0"),
                    gpu=torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None)
    args.output.with_suffix(".config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with args.output.open("w", encoding="utf-8") as handle, torch.inference_mode():
        def emit(row):
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        for request in requests:
            if args.tiny:
                prompt = torch.arange(2, 34, device=args.device).unsqueeze(0)
            else:
                text = request["prompt"]
                if args.chat_template:
                    text = tokenizer.apply_chat_template([{"role": "user", "content": text}],
                                                          tokenize=False, add_generation_prompt=True)
                prompt = tokenizer(text, return_tensors="pt").input_ids.to(args.device)
            if args.oracle:
                layers = tuple(sorted({0, len(model.model.transformer.blocks) // 2,
                                       len(model.model.transformer.blocks) - 1}))
                probe = OracleProbe(lambda row: emit(dict(id=request["id"], **row)), layers=layers)
                wrapped = NativeOracleWrapper(model, probe, prompt.shape[1], args.block_length)
                generate_with_dual_cache(wrapped, prompt, steps=args.gen_length,
                                        gen_length=args.gen_length, block_length=args.block_length,
                                        threshold=args.threshold, factor=args.factor, mask_id=mask_id)
                print(f"oracle complete: {request['id']} (diagnostics only)", flush=True)
                continue
            # Alternate order on successive repeats to reduce order bias.
            for repeat in range(-args.warmup, args.repeats):
                order = methods if repeat % 2 == 0 else list(reversed(methods))
                for method in order:
                    config = ReframeConfig(kind="pair" if method == "materialize" else
                                           (method if not method.startswith("native-") else "stale"),
                                           pilots=args.pilots, refresh_blocks=args.refresh_blocks,
                                           backend=args.backend, materialize=method == "materialize",
                                           max_pilot_error=args.max_pilot_error, ridge=args.ridge)
                    if method.startswith("native-"):
                        generator = generate_with_dual_cache if method == "native-dual" else generate_with_prefix_cache
                        synchronize(prompt.device)
                        if prompt.device.type == "cuda":
                            torch.cuda.reset_peak_memory_stats(prompt.device)
                        started = time.perf_counter()
                        out, nfe = generator(model, prompt, steps=args.gen_length, gen_length=args.gen_length,
                                             block_length=args.block_length, threshold=args.threshold,
                                             factor=args.factor, mask_id=mask_id)
                        synchronize(prompt.device)
                        elapsed = time.perf_counter() - started
                        stats = dict(nfe=nfe, elapsed_seconds=elapsed, generated_slots=args.gen_length,
                                     full_forwards=args.gen_length // args.block_length,
                                     slots_per_second=args.gen_length / elapsed,
                                     peak_memory_bytes=(torch.cuda.max_memory_allocated(prompt.device)
                                                        if prompt.device.type == "cuda" else None))
                    else:
                        out, nfe, stats = generate_reframe(model, prompt, gen_length=args.gen_length,
                                                           block_length=args.block_length,
                                                           threshold=args.threshold, factor=args.factor,
                                                           mask_id=mask_id, config=config,
                                                           audit_every=args.audit_every)
                    if repeat < 0:
                        continue
                    ids = out[0, prompt.shape[1]:].tolist()
                    if tokenizer:
                        useful = ids[:ids.index(tokenizer.eos_token_id)] if tokenizer.eos_token_id in ids else ids
                        text = tokenizer.decode(useful, skip_special_tokens=True)
                        for stop in request.get("until", []):
                            if stop:
                                text = text.split(stop)[0]
                        useful_count = len(tokenizer.encode(text, add_special_tokens=False))
                    else:
                        text, useful_count = "<random tiny model; not a task score>", None
                    row = dict(id=request["id"], method=method, repeat=repeat, tiny=args.tiny,
                               config=asdict(config), prompt_tokens=prompt.shape[1],
                               stats=stats, output_ids=ids, text=text,
                               useful_tokens=useful_count,
                               useful_tokens_per_second=(useful_count / stats["elapsed_seconds"]
                                                         if useful_count is not None else None))
                    emit(row)
                    print(f"{request['id']} {method}: {stats['elapsed_seconds']:.3f}s, NFE={nfe}, "
                          f"full={stats['full_forwards']}, fallback={stats.get('fallback_refreshes', 0)}", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()

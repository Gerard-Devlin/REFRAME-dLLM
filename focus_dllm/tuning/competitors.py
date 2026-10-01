"""Pinned third-party LLaDA smoke/screen runner. No main-evaluation mutations.

Each method runs in a separate process, with batch one and offline weights.
The official decoder/cache implementations are imported without modifying their
files. Any backend or integration adaptation is explicitly recorded.
"""
import argparse
from contextlib import contextmanager
import hashlib
import importlib
import json
import math
from pathlib import Path
import random
import sys
import time
import types


METHODS = ("v1", "v1_dual", "focus_v1", "dkv_decode", "dllm_cache",
           "hierarchy_full", "hierarchy_prefix", "elastic_native",
           "elastic_flash", "flash_cache", "flash_verify")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def namespace(name, path):
    """Avoid importing unrelated optional multimodal/serving dependencies."""
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        sys.modules[name] = module


def select_samples(path, limit, offset, seed=51713):
    rows = json.loads(Path(path).read_text(encoding="utf-8"))
    random.Random(seed).shuffle(rows)
    if limit <= 0 or offset < 0 or offset + limit > len(rows):
        raise ValueError("Invalid sample range")
    selected = rows[offset:offset + limit]
    ids = [str(s.get("id", s.get("task_id"))) for s in selected]
    if len(set(ids)) != len(ids) or "None" in ids:
        raise ValueError("Missing or duplicated sample IDs")
    return selected


def generation_prompt(sample):
    # Explicit whitelist: reference answers/tests/solutions never enter input.
    value = sample.get("paper_prompt", sample.get("prompt"))
    if not isinstance(value, str) or not value:
        raise ValueError("No legitimate prompt")
    return value


def verified_nfe(method, reported, top_level_calls, attention_calls, layers):
    if method == "focus_v1":
        # The pruned path executes blocks directly; top-level model hooks only
        # count warm-ups. Every virtual forward still executes every layer.
        if reported is None or attention_calls != reported * layers:
            raise ValueError("FOCUS decoder/attention NFE mismatch")
        return reported
    if not method.startswith("flash_") and reported is not None and reported != top_level_calls:
        raise ValueError("Official/model NFE mismatch")
    return top_level_calls


def load_external(root, method):
    """Import only the official components needed by this process."""
    if method == "dkv_decode":
        source = root / "dkv_cache"
        sys.path.insert(0, str(source))
        model = importlib.import_module("models.modeling_llada_dkv_cache_decode")
        generate = importlib.import_module("generation_utils.llada_dkv_cache_decode").generate
        return model.LLaDAModelLM, generate, []
    if method == "dllm_cache":
        source = root / "dllm_cache"
        sys.path.insert(0, str(source))
        namespace("dllm_cache", source / "dllm_cache")
        namespace("dllm_cache.hooks", source / "dllm_cache/hooks")
        hooks = importlib.import_module("dllm_cache.hooks.cache_hook_LLaDA")
        generate = importlib.import_module("utils.generate_function").generate
        return None, (hooks, generate), ["Only LLaDA hooks loaded; unused multimodal package initializers bypassed"]
    if method.startswith("hierarchy"):
        source = root / "hierarchy/python/dinfer"
        namespace("dinfer", source)
        namespace("dinfer.decoding", source / "decoding")
        strategy = importlib.import_module("dinfer.decoding.parallel_strategy")
        if method == "hierarchy_full":
            generate = importlib.import_module("dinfer.decoding.generate_hierarchy").generate_hierarchy
            return None, generate, ["Official standalone hierarchy_fast_v2 decoder; no cache; no torch.compile"]
        return None, strategy.HierarchyDecoder, [
            "Official HierarchyDecoder integrated with native Fast-dLLM PrefixCache",
            "This is decoder-only integration, not full dInfer/vLLM system reproduction"]
    if method.startswith("elastic") or method.startswith("flash_"):
        key = "elastic_cache" if method.startswith("elastic") else "flash_dllm"
        source = root / key / "llada"
        sys.path.insert(0, str(source))
        model = importlib.import_module("model.modeling_llada")
        generate = importlib.import_module("generate")
        function = generate.generate_with_elastic_cache if method.startswith("elastic") else generate.generate_with_Flash_dLLM
        return model.LLaDAModelLM, function, []
    return None, None, []


@contextmanager
def elastic_attention(model, use_flash):
    """Elastic requires attention weights, which flash-attn does not return.

    Preserve the official score computation; use FlashAttention only for the
    weighted value output. Its additional score materialization cost is paid.
    """
    import torch
    from flash_attn import flash_attn_func
    saved = []
    stats = dict(attention_calls=0, flash_calls=0, torch_sdpa_calls=0,
                 dense_score_calls=0, dense_value_calls=0)
    for block in model.model.transformer.blocks:
        saved.append((block, block._scaled_dot_product_attention))
        def attention(q, k, v, attn_mask=None, dropout_p=0., is_causal=False,
                      need_weights=False):
            assert attn_mask is None and not is_causal and dropout_p == 0.
            stats["attention_calls"] += 1
            stats["dense_score_calls"] += 1
            if q.shape[1] != k.shape[1]:
                copies = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(copies, dim=1)
                v = v.repeat_interleave(copies, dim=1)
            weights = torch.softmax(q @ k.transpose(-2, -1) * (1 / math.sqrt(q.shape[-1])), dim=-1)
            if use_flash:
                stats["flash_calls"] += 1
                value = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2),
                                       v.transpose(1, 2), causal=False).transpose(1, 2)
            else:
                stats["dense_value_calls"] += 1
                value = weights @ v
            return value, weights
        block._scaled_dot_product_attention = attention
    try:
        yield stats
    finally:
        for block, original in saved:
            block._scaled_dot_product_attention = original


def hierarchy_prefix(model, prompt, decoder_class, length, block_length, threshold, low_threshold):
    """Same full warm-up/formal PrefixCache, official hierarchical decisions.

    Full block logits are passed to the official decoder, preserving contiguous
    MASK segment identities; compacting MASK rows would change the algorithm.
    """
    import torch
    from ..llada_common import MASK_ID
    decoder = decoder_class(temperature=0, threshold=threshold, low_threshold=low_threshold)
    canvas = torch.full((1, prompt.shape[1] + length), MASK_ID, device=prompt.device, dtype=torch.long)
    canvas[:, :prompt.shape[1]] = prompt
    for block in range(length // block_length):
        start = prompt.shape[1] + block * block_length
        end = start + block_length
        decoder.block_init(canvas[:, start:end], block)
        output = model(canvas, use_cache=True)
        decoder.decode(output.logits[:, start:end], start, end, canvas)
        past = [tuple(value[:, :, :start] for value in layer) for layer in output.past_key_values]
        while (canvas[:, start:end] == MASK_ID).any():
            output = model(canvas[:, start:], past_key_values=past, use_cache=True)
            decoder.decode(output.logits[:, :block_length], start, end, canvas)
    return canvas


def load_model(args, cls):
    import torch
    from transformers import AutoModel, AutoTokenizer
    from ..llada_common import snapshot
    from ..llada_evaluate import load_model as native
    if args.method == "dllm_cache":
        # Official hooks expect the original model block signature, whereas
        # Fast-dLLM's local class adds a replace_position keyword.
        checkpoint = snapshot()
        model = AutoModel.from_pretrained(checkpoint, trust_remote_code=True,
            local_files_only=True, torch_dtype=torch.bfloat16).to("cuda:0").eval()
        return model, AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    if cls is None:
        return native("cuda:0")
    checkpoint = snapshot()
    config = cls.config_class.from_pretrained(checkpoint, local_files_only=True)
    # Elastic's official Flash branch has a return-signature mismatch; handled
    # by the explicit, cost-accounted adapter above. Flash-dLLM uses Triton.
    config.flash_attention = not args.method.startswith("elastic")
    model, info = cls.from_pretrained(checkpoint, config=config, local_files_only=True,
                                    torch_dtype=torch.bfloat16, output_loading_info=True)
    assert not info["missing_keys"] and not info["unexpected_keys"] and not info["mismatched_keys"], info
    return model.to("cuda:0").eval(), AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)


def run_generation(args, model, tokenizer, ids, external):
    import torch
    from ..llada_backend import LLaDAAttentionBackend
    from ..llada_evaluate import run_method
    prompt = torch.tensor([ids], dtype=torch.long, device=model.device)
    calls = [0]
    def count(_model, _inputs):
        calls[0] += 1
    handle = model.register_forward_pre_hook(count)
    if args.method.startswith("elastic"):
        context = elastic_attention(model, args.method == "elastic_flash")
    elif args.method.startswith("flash_"):
        from contextlib import nullcontext
        context = nullcontext({"engine": "official fused Triton attention/cache", "torch_sdpa_calls": 0})
    else:
        context = LLaDAAttentionBackend(model, "flash")
    official_nfe = None
    try:
        with context as backend, torch.no_grad():
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            name = args.method
            if name in {"v1", "v1_dual", "focus_v1"}:
                from ..llada_decode import generate_prefix_cache, generate_dual_cache
                from ..llada_pruning import Config, LLaDABlockForward
                forward = None if name != "focus_v1" else LLaDABlockForward(model, Config(
                    prune_after_layer=4, support_keep_ratio=.3125, context_dominant_ratio=1.,
                    contextual_ratio=0., support_contextual_ratio=0., secondary_prune_after_layer=0,
                    target_only_head=True))
                fn = generate_dual_cache if name == "v1_dual" else generate_prefix_cache
                result = fn(model, prompt, gen_length=args.length, block_length=args.block,
                            threshold=args.threshold, block_forward=forward, prune=name == "focus_v1")
                output = result.output[:, len(ids):]
                official_nfe = result.nfe
            elif name == "dkv_decode":
                output = external(model, tokenizer, prompt, steps=args.length, gen_length=args.length,
                    block_length=args.block, temperature=0., cfg_scale=0., enable_cache=True,
                    cache_reloading_step=2)[:, len(ids):]
            elif name == "dllm_cache":
                hooks, generate = external
                from dllm_cache.cache import dLLMCache
                dLLMCache.new_instance(prompt_interval_steps=100, gen_interval_steps=7, transfer_ratio=.25)
                hooks.register_cache_LLaDA(model, "model.transformer.blocks")
                try:
                    output = generate(prompt, None, model, steps=args.length, gen_length=args.length,
                                      block_length=args.block, temperature=0.)
                finally:
                    hooks.logout_cache_LLaDA(model, "model.transformer.blocks")
            elif name == "hierarchy_full":
                output, official_nfe = external(model, prompt, steps=args.length, gen_length=args.length,
                    block_length=args.block, temperature=0., decoding="hierarchy_fast_v2",
                    threshold=args.threshold, low_threshold=args.low_threshold)
                output = output[:, len(ids):]
            elif name == "hierarchy_prefix":
                output = hierarchy_prefix(model, prompt, external, args.length, args.block,
                                          args.threshold, args.low_threshold)[:, len(ids):]
            elif name.startswith("elastic"):
                output, official_nfe, _ = external(model, prompt, gen_length=args.length,
                    window_length=args.block, threshold=args.threshold, gamma=.9, track_num=1)
                output = output[:, len(ids):]
            else:
                responses, steps = [None], [0]
                external(model, [prompt[0]], [len(ids)], 1, responses, steps, gen_length=args.length,
                         block_length=args.block, threshold=args.threshold, gamma=.8, track_num=4,
                         mask_num=4, verify=name == "flash_verify", tokenizer=tokenizer, stop_tokens=[])
                # The official function returns text only. Preserve that; do
                # not re-tokenize text and falsely label it raw generated IDs.
                output = None
                official_nfe = steps[0]
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            peak = torch.cuda.max_memory_allocated() / 2**30
            report = backend.report() if hasattr(backend, "report") else dict(backend)
    finally:
        handle.remove()
    nfe = verified_nfe(args.method, official_nfe, calls[0], report.get("attention_calls"),
                       len(model.model.transformer.blocks))
    result = dict(seconds=seconds, nfe=nfe, top_level_model_calls=calls[0], official_iterations=official_nfe,
                  peak_gib=peak, backend=report)
    if output is None:
        result.update(text=responses[0], token_ids=None, raw_token_ids_available=False)
    else:
        assert output.shape[0] == 1 and output.shape[1] == args.length
        assert not (output == 126336).any(), "Unreleased MASK in final canvas"
        result.update(token_ids=output[0].tolist(), raw_token_ids_available=True)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--third-party", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--task", choices=("gsm8k", "humaneval", "mbpp", "math"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--length", type=int, default=256)
    parser.add_argument("--block", type=int, default=32)
    parser.add_argument("--threshold", type=float, default=.90)
    parser.add_argument("--low-threshold", type=float, default=.62)
    parser.add_argument("--imports-only", action="store_true")
    args = parser.parse_args()
    if args.length % args.block or args.block <= 0:
        raise ValueError("Generation length must be divisible by block length")
    samples = select_samples(args.dataset, args.limit, args.offset)
    cls, external, notes = load_external(args.third_party, args.method)
    if args.imports_only:
        print(json.dumps(dict(method=args.method, import_ok=True, adaptations=notes)), flush=True)
        return
    from ..llada_common import MODEL_ID, REVISION, prompt_ids
    from ..llada_evaluate import postprocess_output, percentile
    from .suite import score
    from tqdm import tqdm
    import torch
    torch.manual_seed(1234)
    # Compile is not a shared setting in the paper table. Standalone hierarchy
    # has a decorated decision function; disable only that compile wrapper.
    if args.method == "hierarchy_full":
        module = importlib.import_module("dinfer.decoding.generate_hierarchy")
        fn = module.get_transfer_index_hierarchy_fast_v2
        module.get_transfer_index_hierarchy_fast_v2 = getattr(fn, "_torchdynamo_orig_callable", fn)
    if args.method == "elastic_flash":
        notes += ["Attention value output uses FlashAttention; official dense attention scores retained and timed"]
    if args.method.startswith("flash_"):
        notes += ["Official fused Triton kernels instead of flash-attn library; kernel engineering gains not FOCUS gains",
                  "Official stopping/output formatting retained; raw token IDs unavailable"]
    if args.method == "dllm_cache":
        notes += ["Original pinned HF LLaDA class for official hook signature, not modified local v1 class"]
    identity = dict(model=MODEL_ID, revision=REVISION, method=args.method, task=args.task,
        length=args.length, block=args.block, threshold=args.threshold, low_threshold=args.low_threshold,
        batch_size=1, dtype="bfloat16", seed=1234, sample_seed=51713, offset=args.offset,
        ids=[str(s.get("id", s.get("task_id"))) for s in samples], dataset_sha256=digest(args.dataset),
        sources=json.loads((args.third_party / "sources.json").read_text()), runner_sha256=digest(__file__),
        adaptations=notes, scope="Reused development screen; no independent quality or losslessness claim")
    manifest = args.output / "manifest.json"
    if manifest.exists():
        assert json.loads(manifest.read_text()) == identity, "Resume identity changed"
    else:
        write(manifest, identity)
    model, tokenizer = load_model(args, cls)
    # Warm actual workload; excluded from timing and scored records.
    warm_ids = prompt_ids(tokenizer, generation_prompt(samples[0]), args.task, preformatted=True)
    warm_started = time.perf_counter()
    run_generation(args, model, tokenizer, warm_ids, external)
    warm_seconds = time.perf_counter() - warm_started
    rows = []
    for index, sample in enumerate(tqdm(samples, desc=args.method, ascii=False)):
        path = args.output / "records" / f"{index:05d}.json"
        ident = str(sample.get("id", sample.get("task_id")))
        if path.exists():
            row = json.loads(path.read_text())
            assert row["id"] == ident
        else:
            ids = prompt_ids(tokenizer, generation_prompt(sample), args.task, preformatted=True)
            result = run_generation(args, model, tokenizer, ids, external)
            if result["token_ids"] is not None:
                text, tokens = postprocess_output(tokenizer, result["token_ids"], sample, args.task)
                result.update(text=text, output_tokens=tokens,
                    eos=126081 in result["token_ids"], truncated=126081 not in result["token_ids"])
            else:
                result.update(output_tokens=len(tokenizer.encode(result["text"], add_special_tokens=False)),
                              eos=None, truncated=None)
            row = dict(index=index, id=ident, **{args.method: result})
            write(path, row)
        rows.append(row)
    score_started = time.perf_counter()
    correctness = score(args.task, samples, rows, [args.method])[args.method]
    elapsed = [r[args.method]["seconds"] for r in rows]
    result = dict(method=args.method, examples=len(rows), accuracy=sum(correctness)/len(rows),
        correctness=correctness, ids=identity["ids"], mean_seconds=sum(elapsed)/len(elapsed),
        p50_seconds=percentile(elapsed, .5), p95_seconds=percentile(elapsed, .95),
        mean_nfe=sum(r[args.method]["nfe"] for r in rows)/len(rows),
        mean_output_tokens=sum(r[args.method]["output_tokens"] for r in rows)/len(rows),
        warmup_seconds=warm_seconds, scoring_seconds=time.perf_counter()-score_started,
        adaptations=notes, scope=identity["scope"])
    write(args.output / "summary.json", result)
    (args.output / "complete").write_text("OK\n")
    print("Completed " + json.dumps(result), flush=True)


if __name__ == "__main__":
    main()

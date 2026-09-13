"""Sequential, single-GPU accuracy and matched-prompt timing campaigns.

Run inside tmux, with the lab conda/HF environment already activated.
Calls original v1 evaluators/generators without editing them.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("native-serial", "native-prefix-serial", "native-full", "native-prefix",
           "native-dual", "stale", "shift", "pair")
ALLOWED = set(METHODS) | {"scale", "materialize"}


def model_options(method, model_path, length, log_dir, pilots, refresh):
    common = f"model_path={model_path},gen_length={length},steps={length},block_length=32,show_speed=True"
    if method.startswith("native-"):
        if method in {"native-prefix", "native-prefix-serial"}:
            common += ",use_cache=True"
        elif method == "native-dual":
            common += ",use_cache=True,dual_cache=True"
        if method not in {"native-serial", "native-prefix-serial"}:
            common += ",threshold=0.9"
        return "v1/llada/eval_llada.py", "llada_dist", common
    kind = "pair" if method == "materialize" else method
    common += (f",threshold=0.9,reframe_kind={kind},reframe_pilots={pilots},"
               f"reframe_refresh_blocks={refresh},reframe_backend=flash,reframe_log={log_dir}")
    if method == "materialize":
        common += ",reframe_materialize=True"
    return "reframe/eval_reframe.py", "reframe_llada", common


def make_gsm_task(destination):
    # 0.4.8 uses dataset_path: gsm8k; the lab downloaded openai/gsm8k.
    # Preserve installed task prompts, filters and scoring; change only ID/name.
    import lm_eval
    source = Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml"
    text = source.read_text(encoding="utf-8")
    text, n1 = re.subn(r"(?m)^task: gsm8k\s*$", "task: gsm8k_local", text)
    text, n2 = re.subn(r"(?m)^dataset_path: (?:openai/)?gsm8k\s*$", "dataset_path: openai/gsm8k", text)
    if (n1, n2) != (1, 1):
        raise RuntimeError("Unexpected GSM8K task config; inspect before running")
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "gsm8k_local.yaml").write_text(text, encoding="utf-8")


def logged_generation_args(sample):
    """Read both evaluator's in-memory and EvaluationTracker's saved format."""
    arguments = sample.get("arguments")
    try:
        if isinstance(arguments, dict):
            # lm-eval 0.4.8 rewrites requests to named fields when saving JSONL.
            request = arguments["gen_args_0"]
            prompt, kwargs = request["arg_0"], request["arg_1"]
        elif isinstance(arguments, (list, tuple)):
            prompt, kwargs = arguments[0]
        else:
            raise TypeError("arguments must be a mapping or sequence")
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(f"Unexpected lm-eval arguments for doc_id={sample.get('doc_id')}; "
                         "expected gen_args_0.{arg_0,arg_1} or [[prompt, kwargs]]") from exc
    if not isinstance(prompt, str) or not isinstance(kwargs, dict):
        raise ValueError(f"Invalid prompt/kwargs types for doc_id={sample.get('doc_id')}")
    return prompt, kwargs


def export_prompts(source, task, limit, target):
    files = sorted(source.rglob(f"samples_{'gsm8k_local' if task == 'gsm8k' else 'humaneval'}_*.jsonl"))
    if len(files) != 1:
        raise ValueError(f"Expected ONE {task} sample file under {source}, found {len(files)}")
    rows, seen = [], set()
    for line in files[0].read_text(encoding="utf-8").splitlines():
        sample = json.loads(line)
        doc_id = sample["doc_id"]
        if doc_id in seen:
            continue
        seen.add(doc_id)
        prompt, kwargs = logged_generation_args(sample)
        # Native Instruct HumanEval evaluator deliberately ignores stop strings.
        until = kwargs.get("until", []) if task == "gsm8k" else []
        if isinstance(until, str):
            until = [until]
        if not isinstance(until, list) or not all(isinstance(stop, str) for stop in until):
            raise ValueError(f"Invalid stop strings for doc_id={doc_id}")
        rows.append(dict(id=str(doc_id), prompt=prompt,
                         until=until))
    rows = rows[:limit]
    if not rows:
        raise ValueError("No evaluation prompts found")
    with target.open("x", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return len(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["smoke", "accuracy", "timing", "oracle", "audit"])
    p.add_argument("--task", choices=["gsm8k", "humaneval"], default="gsm8k")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--model-path", default="GSAI-ML/LLaDA-8B-Instruct")
    p.add_argument("--gen-length", type=int, default=256)
    p.add_argument("--limit", type=int, help="smoke default 2; timing default 16; accuracy default all")
    p.add_argument("--source", type=Path, help="One method's lm-eval output directory, for prompt replay")
    p.add_argument("--output", type=Path, required=True, help="New directory, never overwrite existing runs")
    p.add_argument("--pilots", type=int, default=16)
    p.add_argument("--refresh-blocks", type=int, default=2)
    args = p.parse_args()
    methods = args.methods.split(",")
    if not methods or set(methods) - ALLOWED or len(set(methods)) != len(methods):
        p.error("Unknown/duplicate methods")
    if args.gen_length < 32 or args.gen_length % 32 or (args.limit is not None and args.limit < 1):
        p.error("Use positive lengths divisible by 32 and a positive limit")
    if args.stage in {"timing", "oracle", "audit"} and args.source is None:
        p.error("This stage needs --source pointing to one method's sample directory")
    if importlib.metadata.version("lm_eval") != "0.4.8":
        p.error("Use lm_eval==0.4.8 to preserve the checked prompt/scoring format")
    import torch
    import flash_attn
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        p.error("Expose exactly one GPU, for example export CUDA_VISIBLE_DEVICES=0")
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    packages = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "lm_eval", "accelerate", "datasets", "flash_attn")}
    manifest = dict(arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
                    started=datetime.now().isoformat(), packages=packages,
                    gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
                    visible_gpu=os.environ.get("CUDA_VISIBLE_DEVICES"),
                    compile_disable=os.environ.get("TORCH_COMPILE_DISABLE", "0"))
    for name, command in (("git_commit", ["git", "rev-parse", "HEAD"]),
                          ("gpu_status", ["nvidia-smi"])):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        manifest[name] = result.stdout.strip() if result.returncode == 0 else result.stderr.strip()
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    def run(command, name):
        with (args.output / "commands.jsonl").open("a", encoding="utf-8") as record:
            record.write(json.dumps(dict(name=name, command=command)) + "\n")
        print(f"Running {name}; log: {args.output / (name + '.log')}", flush=True)
        with (args.output / f"{name}.log").open("w", encoding="utf-8") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f"{name} failed (exit {result.returncode}); see its log. Campaign stopped.")
        print(f"Completed {name}", flush=True)

    if args.stage in {"smoke", "accuracy"}:
        task, extra = args.task, []
        if task == "gsm8k":
            task_dir = args.output / "task_config"
            make_gsm_task(task_dir)
            task, extra = "gsm8k_local", ["--include_path", str(task_dir)]
        limit = args.limit if args.limit is not None else (2 if args.stage == "smoke" else None)
        for method in methods:
            directory = args.output / method
            # Absolute paths keep the original evaluator usable from repo root.
            entry, model, options = model_options(method, args.model_path, args.gen_length,
                                                  directory / "metrics", args.pilots, args.refresh_blocks)
            command = [sys.executable, "-u", str(ROOT / entry), "--model", model, "--model_args", options,
                       "--tasks", task, "--num_fewshot", "5" if args.task == "gsm8k" else "0",
                       "--batch_size", "1", "--seed", "0,1234,1234,1234",
                       "--confirm_run_unsafe_code", "--log_samples", "--output_path", str(directory), *extra]
            if limit is not None:
                command += ["--limit", str(limit)]
            run(command, method)
            if args.task == "humaneval":
                samples = list(directory.rglob("samples_humaneval_*.jsonl"))
                if len(samples) != 1:
                    raise RuntimeError(f"Expected one HumanEval samples file in {directory}")
                run([sys.executable, "-u", str(ROOT / "v1/llada/postprocess_code.py"), str(samples[0])],
                    method + "_postprocess")
        scores = {}
        for method in methods:
            if args.task == "gsm8k":
                files = list((args.output / method).rglob("results_*.json"))
                if len(files) != 1:
                    raise RuntimeError(f"Expected one results file for {method}")
                scores[method] = json.loads(files[0].read_text(encoding="utf-8"))["results"][task]
            else:
                files = list((args.output / method).rglob("*.jsonl.cleaned"))
                if len(files) != 1:
                    raise RuntimeError(f"Expected one cleaned HumanEval results file for {method}")
                rows = [json.loads(s) for s in files[0].read_text(encoding="utf-8").splitlines()]
                scores[method] = dict(cleaned_pass_at_1=sum(r["pass_at_1"] for r in rows) / len(rows),
                                      samples=len(rows))
        (args.output / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
        print(json.dumps(scores, indent=2), flush=True)
        print("Accuracy stage complete. Smoke/subset scores are not final benchmark results.")
        print("Do not compare printed evaluator TPS across native and REFRAME. Run the timing stage.")
    else:
        limit = args.limit or (16 if args.stage == "timing" else 2)
        prompts = args.output / "prompts.jsonl"
        n = export_prompts(args.source, args.task, limit, prompts)
        output = args.output / "measurements.jsonl"
        command = [sys.executable, "-u", str(ROOT / "reframe/run.py"), "--model-path", args.model_path,
                   "--device", "cuda", "--dtype", "bfloat16", "--backend", "flash",
                   "--prompts", str(prompts), "--limit", str(n), "--gen-length", str(args.gen_length),
                   "--block-length", "32", "--threshold", "0.9", "--pilots", str(args.pilots),
                   "--refresh-blocks", str(args.refresh_blocks), "--output", str(output)]
        if args.stage == "oracle":
            command += ["--oracle"]
        elif args.stage == "audit":
            command += ["--methods", "pair", "--audit-every", "4", "--warmup", "0", "--repeats", "1"]
        else:
            command += ["--methods", args.methods, "--warmup", "1", "--repeats", "3"]
        run(command, args.stage)
        run([sys.executable, str(ROOT / "reframe/summarize.py"), str(output)], "summary")
        print((args.output / "summary.log").read_text(encoding="utf-8"))
    print(f"Saved campaign: {args.output}", flush=True)


if __name__ == "__main__":
    main()

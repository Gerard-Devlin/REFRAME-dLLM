"""Resume the paper matrix by independent prompts, without changing decoding.

Legacy rank shards are immutable. New prompts are atomically committed as one
file each; a crash after that commit can never force the prompt to run again.
GPU workers keep one model loaded and steal work from the longest remaining job.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

from .elastic_work import ElasticQueue, load_completed_records
from .smart_paper_scheduler import build_jobs


CODE_FILES = ("llada_evaluate.py", "llada_decode.py", "llada_backend.py",
              "llada_pruning.py", "llada_common.py", "common.py")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def eval_args(job):
    return dict(stage="evaluate", task=job.task, dataset=job.dataset, limit=job.limit,
                gen_length=job.gen, block_length=32, threshold=0.90,
                decoding_mode=job.decoding, cache_mode=job.cache,
                prune_after_layer=4, support_keep_ratio=0.3125,
                context_dominant_ratio=1.0, contextual_ratio=0.0,
                support_contextual_ratio=0.0, context_merge_weight=0.5,
                secondary_prune_after_layer=0, secondary_support_ratio=1.0,
                methods=job.methods.split())


def record_paths(root, item):
    output = Path(root) / item["name"] / "output"
    # rank_elastic.jsonl is a derived index, never an additional raw input.
    return ([path for path in sorted(output.glob("rank_*.jsonl"))
             if path.name != "rank_elastic.jsonl"] +
            sorted((output / "elastic_records").glob("*.json")))


def records_for(root, item):
    records = load_completed_records(
        record_paths(root, item), expected_count=item["job"]["limit"],
        required_methods=item["args"]["methods"], allow_truncated_tail=True)
    samples = json.loads(Path(item["job"]["dataset"]).read_text(encoding="utf-8"))
    if isinstance(samples, dict):
        samples = samples.get("samples", samples.get("data"))
    if not isinstance(samples, list) or len(samples) < item["job"]["limit"]:
        raise ValueError("Dataset is not the complete expected prompt list")
    for index, record in records.items():
        expected_id = samples[index].get("id", samples[index].get("task_id"))
        if record["id"] != expected_id:
            raise ValueError(f"Dataset identity changed at {item['name']}:{index}")
        if "execution" in record:
            encoded = json.dumps(item["identity"], sort_keys=True, separators=(",", ":"),
                                 ensure_ascii=False).encode()
            if record["execution"].get("identity") != hashlib.sha256(encoded).hexdigest():
                raise ValueError("Elastic output configuration identity mismatch")
        for method in item["args"]["methods"]:
            row = record[method]
            if row["canvas_tokens"] != item["args"]["gen_length"]:
                raise ValueError("Generation length mismatch in existing record")
            if row["seconds"] <= 0 or row["nfe"] <= 0 or row["backend"]["flash_calls"] <= 0:
                raise ValueError("Invalid measured result")
    return records


def queue_for(root, item, completed=()):
    return ElasticQueue(Path(root) / "elastic" / "queues" / item["name"],
                        total=item["job"]["limit"], identity=item["identity"],
                        completed_indices=completed)


def initialize(args):
    root = args.run_root
    manifest_path = root / "elastic" / "manifest.json"
    if manifest_path.exists() and not args.plan_only:
        raise ValueError("Already initialized; run scheduler to resume the existing manifest")
    files = {name: digest(Path(__file__).parent / name) for name in CODE_FILES}
    from .llada_common import MODEL_ID, REVISION
    items = []
    for job in build_jobs({task: getattr(args, task + "_dataset")
                           for task in ("gsm8k", "math", "humaneval", "mbpp")}):
        config = eval_args(job)
        identity = dict(model=MODEL_ID, revision=REVISION, dataset_sha256=digest(job.dataset),
                        configuration=config, implementation=files)
        item = dict(name=job.name, job=asdict(job), args=config, identity=identity)
        # Verify any existing summary's configuration before admitting its results.
        summary = root / job.name / "output" / "summary.json"
        if summary.exists():
            prior = json.loads(summary.read_text())
            if prior.get("model") != MODEL_ID or prior.get("revision") != REVISION:
                raise ValueError(f"Changed model/revision for {job.name}")
            if prior["dataset_sha256"] != identity["dataset_sha256"]:
                raise ValueError(f"Changed dataset for {job.name}")
            for key, value in config.items():
                if key in ("stage", "dataset", "methods", "limit"):
                    continue
                if prior["configuration"].get(key) != value:
                    if (key == "context_merge_weight" and config["contextual_ratio"] == 0
                            and config["support_contextual_ratio"] == 0
                            and not any("zip" in name for name in config["methods"])):
                        # Old completed FastV-only campaigns exposed this unused
                        # VisionZip setting. Keep their recorded value verbatim.
                        config[key] = prior["configuration"][key]
                        continue
                    raise ValueError(f"Changed setting {key} for {job.name}")
        rows = records_for(root, item)
        measured = [sum(row[method]["seconds"] for method in config["methods"])
                    for row in rows.values()]
        # Remaining estimates are based on observed work; initial prior only
        # orders not-yet-started jobs and is replaced in status as workers run.
        estimate = (sum(measured) / len(measured) if measured else
                    job.gen * {"llada": .13, "cache": .046,
                               "parallel_ours": .05, "fastdllm_ours_cache": .023}[job.label])
        item.update(seconds_per_prompt=estimate, imported=len(rows))
        items.append(item)
        print(f"{job.name}: {len(rows)}/{job.limit}, remaining GPU h="
              f"{(job.limit-len(rows))*estimate/3600:.2f}", flush=True)
        if not args.plan_only:
            queue_for(root, item, rows)
    remaining = sum((item["job"]["limit"]-item["imported"])*item["seconds_per_prompt"]
                    for item in items)
    print(f"Estimated remaining GPU-hours={remaining/3600:.2f}; "
          f"six-GPU ideal hours={remaining/3600/6:.2f}", flush=True)
    if not args.plan_only:
        atomic_json(manifest_path, dict(version=1, created_at=time.time(),
                                       jobs=items, implementation=files))


def load_manifest(root):
    manifest = json.loads((Path(root) / "elastic" / "manifest.json").read_text())
    for name, expected in manifest["implementation"].items():
        if digest(Path(__file__).parent / name) != expected:
            raise ValueError(f"Decoder source changed during campaign: {name}")
    return manifest


def make_record(model, tokenizer, sample, index, config):
    # Deliberately use the existing decoder/timer/postprocessor unchanged.
    from .llada_evaluate import run_method, postprocess_output, math_answer
    from .llada_common import prompt_ids, extract_answer
    paper = sample.get("paper_prompt")
    text = paper or sample["question"] if config.task == "gsm8k" else (
        paper if paper is not None else sample["prompt"])
    ids = prompt_ids(tokenizer, text, config.task, preformatted=paper is not None)
    target = (extract_answer(sample["answer"], gold=True) if config.task == "gsm8k" else
              sample["answer"] if config.task == "math" else sample["task_id"])
    record = dict(index=index, id=sample.get("id", sample.get("task_id")), target=target)
    for method in config.methods:
        outcome = run_method(model, ids, config, method)
        text, count = postprocess_output(tokenizer, outcome.pop("token_ids"), sample, config.task)
        diagnostics = [r for r in outcome.pop("records") if r is not None]
        prediction = (extract_answer(text) if config.task == "gsm8k" else
                      math_answer(text) if config.task == "math" else None)
        record[method] = dict(
            **outcome, text=text, prediction=prediction,
            correct=((prediction is not None and prediction == target)
                     if config.task in ("gsm8k", "math") else None),
            canvas_tokens=config.gen_length, output_tokens=count,
            deep_tokens=[r.get("final_deep_tokens", r["deep_tokens"]) for r in diagnostics],
            retained_mass=[r["retained_support_mass"] for r in diagnostics],
            context_tokens=[r["kept_context"] for r in diagnostics],
            score_seconds=[r["score_seconds"] for r in diagnostics])
    return record


def worker(args):
    from .wait_for_idle_gpus import snapshot
    # Check again before CUDA initialization; never use a busy card just because
    # a scheduler saw it idle some seconds earlier.
    rows = snapshot({args.gpu}, 1024, 5)
    if len(rows) != 1 or not rows[0][1]:
        raise RuntimeError("GPU became busy before worker startup")
    manifest = load_manifest(args.run_root)
    jobs = manifest["jobs"]
    queues = {item["name"]: queue_for(args.run_root, item) for item in jobs}
    for item in jobs:
        if digest(item["job"]["dataset"]) != item["identity"]["dataset_sha256"]:
            raise ValueError("Dataset content changed")
    import torch
    from .llada_evaluate import load_model, run_method
    from .llada_common import load_samples, prompt_ids
    torch.cuda.set_device(0)
    model, tokenizer = load_model("cuda:0")
    samples, warmed = {}, set()
    status_path = args.run_root / "elastic" / "workers" / f"{args.gpu}.json"
    stop_path = status_path.with_suffix(".stop")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    count = 0
    while not stopping and not stop_path.exists():
        states = {name: queue.status() for name, queue in queues.items()}
        available = sorted(jobs, key=lambda item: (
            -states[item["name"]]["pending"]*item["seconds_per_prompt"], item["name"]))
        chosen, lease = None, None
        for item in available:
            if states[item["name"]]["pending"]:
                lease = queues[item["name"]].claim()
                if lease is not None:
                    chosen = item
                    break
        if chosen is None:
            if all(state["complete"] for state in states.values()):
                break
            time.sleep(2)
            continue
        item, name = chosen, chosen["name"]
        queue = queues[name]
        index = lease.indices[0]
        path = args.run_root / name / "output" / "elastic_records" / f"{index:06d}.json"
        if path.exists():
            # Crash between atomic output commit and queue.complete: reuse it.
            row = json.loads(path.read_text(encoding="utf-8"))
            if row["index"] != index or row["execution"]["identity"] != queue.identity_sha256:
                raise ValueError("Conflicting durable prompt output")
            queue.complete(lease)
            continue
        config = SimpleNamespace(**item["args"])
        if name not in samples:
            samples[name] = load_samples(Path(config.dataset), config.limit, config.task)
        atomic_json(status_path, dict(pid=os.getpid(), gpu=args.gpu, job=name, status="running",
                                     index=index, since=time.time(), completed=count))
        if name not in warmed:
            warmup = prompt_ids(tokenizer, "What is one plus one?", "gsm8k")
            for method in config.methods:
                run_method(model, warmup, config, method)
            warmed.add(name)
        if stopping or stop_path.exists():
            queue.release(lease)
            break
        record = make_record(model, tokenizer, samples[name][index], index, config)
        # A monitor can ask for drain when someone starts on our card. Keep the
        # contested timing separately and put the prompt back; never mix it into
        # the uncontended paper measurements.
        if stop_path.exists():
            atomic_json(path.parent / "contended" / f"{index:06d}.{os.getpid()}.json", record)
            queue.release(lease)
            break
        record["execution"] = dict(pid=os.getpid(), gpu=args.gpu, finished_at=time.time(),
                                   identity=queue.identity_sha256, scheduling="elastic_batch1")
        atomic_json(path, record)
        queue.complete(lease)
        count += 1
        print(f"gpu={args.gpu} {name} index={index} done={queue.status()['completed']}/"
              f"{config.limit}", flush=True)
    atomic_json(status_path, dict(pid=os.getpid(), gpu=args.gpu, stopped=True, status="stopped",
                                 completed=count, time=time.time()))


def finalize(args):
    from .llada_evaluate import aggregate
    from .llada_common import MODEL_ID, REVISION, load_samples
    manifest = load_manifest(args.run_root)
    item = next(item for item in manifest["jobs"] if item["name"] == args.job)
    rows = records_for(args.run_root, item)
    if sorted(rows) != list(range(item["job"]["limit"])):
        raise ValueError("Cannot finalize incomplete job")
    records = [rows[index] for index in sorted(rows)]
    output = args.run_root / item["name"] / "output"
    parts = output / "elastic_records"
    # Only new records go into the new shard; legacy shards remain untouched.
    if parts.exists():
        target = output / "rank_elastic.jsonl"
        temporary = target.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as stream:
            for part in sorted(parts.glob("*.json")):
                stream.write(part.read_text(encoding="utf-8").strip() + "\n")
        os.replace(temporary, target)
    config = item["args"]
    samples = load_samples(Path(config["dataset"]), config["limit"], config["task"])
    # Retain old summary for full provenance before replacing the derived index.
    old = output / "summary.json"
    if old.exists() and not (output / "summary.before_elastic.json").exists():
        (output / "summary.before_elastic.json").write_bytes(old.read_bytes())
    report = dict(stage="evaluate", model=MODEL_ID, revision=REVISION,
                  dataset=config["dataset"], dataset_sha256=item["identity"]["dataset_sha256"],
                  ids=[s.get("id", s.get("task_id")) for s in samples],
                  world_size=1, scheduling="independent prompt workers, batch=1",
                  configuration=config | {"output":str(output)},
                  implementation=manifest["implementation"],
                  results=aggregate(records, config["methods"]),
                  scope="Original decoder and timers; elastic prompt assignment only")
    atomic_json(old, report)
    task = config["task"]
    if task in ("math", "humaneval", "mbpp"):
        filename = "math_exact_match.json" if task == "math" else f"{task}_pass_at_1.json"
        # Existing code scores are reusable only if they cover the full job.
        score = output / filename
        if score.exists():
            existing_score = json.loads(score.read_text())
            if existing_score.get("examples") != len(records):
                raise ValueError(f"Existing score has wrong coverage: {score}")
            details = existing_score.get("details", [])
            scored_ids = {str(row.get("id", row.get("task_id"))) for row in details}
            if scored_ids != {str(row["id"]) for row in records}:
                raise ValueError(f"Existing score has wrong prompt IDs: {score}")
            if task == "math" and existing_score["provenance"]["dataset_sha256"] != item["identity"]["dataset_sha256"]:
                raise ValueError("Existing MATH score has different dataset provenance")
        if not score.exists():
            # Make a validated view: legacy raw shards may end in a truncated
            # write after interruption, which must not be silently reparsed.
            view = output / "scoring_input"
            view.mkdir(exist_ok=True)
            temporary = view / "rank_0.tmp"
            with temporary.open("w", encoding="utf-8") as stream:
                for record in records:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            os.replace(temporary, view / "rank_0.jsonl")
            command = [sys.executable, "-m", "fastv_dllm.score_"+task,
                       "--dataset", config["dataset"], "--results", str(view),
                       "--output", str(score)]
            if task == "math":
                command += ["--workers", "4", "--math-verify"]
            subprocess.run(command, check=True)
        scored = json.loads(score.read_text(encoding="utf-8"))
        for method in config["methods"]:
            result = report["results"][method]
            result["provisional_accuracy"] = result["accuracy"]
            if task == "math":
                result["accuracy"] = scored["results"][method]["exact_match"]
                result["accuracy_metric"] = "lm_eval.minerva_math.exact_match"
                result["math_verify"] = scored["results"][method].get("math_verify")
                result["valid_extraction_rate"] = scored["results"][method]["valid_extraction_rate"]
            else:
                result["accuracy"] = scored["pass@1"][method]
                result["accuracy_metric"] = "pass@1"
        report["scoring_artifact"] = str(score)
        atomic_json(old, report)
    (output.parent / "exit_code").write_text("0\n")
    atomic_json(args.run_root / "elastic" / "finalized" / f"{args.job}.json",
                dict(time=time.time(), examples=len(records), summary=str(old)))
    print(f"Finalized {args.job}: {len(records)}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser("initialize")
    init.add_argument("--plan-only", action="store_true")
    for task in ("gsm8k", "math", "humaneval", "mbpp"):
        init.add_argument(f"--{task}-dataset", type=Path, required=True)
    work = sub.add_parser("worker")
    work.add_argument("--gpu", type=int, required=True)
    finish = sub.add_parser("finalize")
    finish.add_argument("--job", required=True)
    for command in (init, work, finish):
        command.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    {"initialize":initialize, "worker":worker, "finalize":finalize}[args.command](args)


if __name__ == "__main__":
    main()

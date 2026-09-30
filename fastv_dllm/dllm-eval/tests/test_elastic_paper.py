import copy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from dllm_eval import worker as paper
from fastv_dllm.llada_common import MODEL_ID, REVISION
from fastv_dllm.smart_paper_scheduler import Job


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def row(index, **extra):
    return {"index": index, "id": f"test:{index}", "target": "2", "flash_native": {
        "canvas_tokens": 256, "seconds": 1.5, "nfe": 7, "backend": {"flash_calls": 7},
        "correct": True, "text": "#### 2", "output_tokens": 3,
    }, **extra}


@pytest.fixture
def campaign(tmp_path, monkeypatch):
    dataset = tmp_path / "dataset.json"
    samples = [{"id": f"test:{index}", "question": f"Question {index}", "answer": "#### 2"}
               for index in range(2)]
    dump(dataset, samples)
    job = Job("gsm8k", 256, "llada", "none", "single", "flash_native", str(dataset), 2)
    monkeypatch.setattr(paper, "build_jobs", lambda datasets: [job])
    args = SimpleNamespace(run_root=tmp_path / "runs", plan_only=False,
                           **{name + "_dataset": dataset for name in ("gsm8k", "math", "mbpp", "humaneval")})
    output = args.run_root / job.name / "output"
    return args, job, samples, output


def seed_legacy(campaign, rows, summary=True):
    args, job, _, output = campaign
    output.mkdir(parents=True, exist_ok=True)
    (output / "rank_0.jsonl").write_text("".join(json.dumps(value) + "\n" for value in rows))
    if summary:
        dump(output / "summary.json", {
            "model": MODEL_ID, "revision": REVISION,
            "dataset_sha256": paper.digest(job.dataset),
            "configuration": paper.eval_args(job),
        })


def test_plan_is_read_only_and_initialize_reuses_validated_rows(campaign):
    args, job, _, output = campaign
    seed_legacy(campaign, [row(0)])
    original = (output / "rank_0.jsonl").read_bytes()
    args.plan_only = True
    paper.initialize(args)
    assert not (args.run_root / "elastic").exists()
    args.plan_only = False
    paper.initialize(args)
    manifest = paper.load_manifest(args.run_root)
    item = manifest["jobs"][0]
    assert item["imported"] == 1
    assert item["identity"]["dataset_sha256"] == paper.digest(job.dataset)
    assert item["identity"]["model"] == MODEL_ID
    assert item["identity"]["revision"] == REVISION
    queue = paper.queue_for(args.run_root, item)
    assert queue.status()["completed"] == 1
    assert queue.claim().indices == (1,)
    assert (output / "rank_0.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="Already initialized"):
        paper.initialize(args)


@pytest.mark.parametrize("change", ["dataset", "threshold", "model", "revision"])
def test_initialize_rejects_changed_identity(campaign, change):
    args, _, _, output = campaign
    seed_legacy(campaign, [row(0)])
    path = output / "summary.json"
    prior = json.loads(path.read_text())
    if change == "dataset":
        prior["dataset_sha256"] = "not-the-dataset"
    elif change == "threshold":
        prior["configuration"]["threshold"] = 0.95
    else:
        prior[change] = "another-checkpoint"
    dump(path, prior)
    with pytest.raises(ValueError):
        paper.initialize(args)


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(id="somebody-else"),
    lambda value: value["flash_native"].update(canvas_tokens=512),
    lambda value: value["flash_native"].update(nfe=0),
    lambda value: value["flash_native"]["backend"].update(flash_calls=0),
])
def test_import_rejects_invalid_measurements(campaign, mutate):
    args, _, _, _ = campaign
    value = row(0)
    mutate(value)
    seed_legacy(campaign, [value])
    with pytest.raises(ValueError):
        paper.initialize(args)


def test_manifest_guards_decoder_source_hash(campaign):
    args, _, _, _ = campaign
    paper.initialize(args)
    path = args.run_root / "elastic" / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["implementation"]["llada_decode.py"] = "changed"
    dump(path, manifest)
    with pytest.raises(ValueError, match="Decoder source changed"):
        paper.load_manifest(args.run_root)


def test_records_ignore_derived_shard_but_reject_real_duplicate(campaign):
    args, _, _, output = campaign
    seed_legacy(campaign, [row(0)], summary=False)
    paper.initialize(args)
    item = paper.load_manifest(args.run_root)["jobs"][0]
    dump(output / "rank_elastic.jsonl", row(0))
    assert list(paper.records_for(args.run_root, item)) == [0]
    dump(output / "elastic_records" / "000000.json", row(0))
    with pytest.raises(ValueError, match="Duplicate"):
        paper.records_for(args.run_root, item)


def test_elastic_output_identity_is_checked(campaign):
    args, _, _, output = campaign
    paper.initialize(args)
    item = paper.load_manifest(args.run_root)["jobs"][0]
    dump(output / "elastic_records" / "000000.json",
         row(0, execution={"identity": "wrong-campaign"}))
    with pytest.raises(ValueError):
        paper.records_for(args.run_root, item)


@pytest.mark.parametrize("task,sample", [
    ("gsm8k", {"id": "a", "question": "raw", "answer": "gold", "paper_prompt": "fewshot"}),
    ("math", {"id": "b", "answer": "gold", "paper_prompt": "math fewshot"}),
    ("humaneval", {"task_id": "HE/1", "prompt": "def func():"}),
    ("mbpp", {"task_id": 123, "paper_prompt": "code fewshot"}),
])
def test_make_record_preserves_legacy_fields_and_postprocessing(monkeypatch, task, sample):
    calls = []
    outcome = {"token_ids": [1, 2], "nfe": 3, "seconds": 4.0, "peak_gib": 5.0,
               "backend": {"flash_calls": 6}, "records": [None, {
                   "deep_tokens": 17, "final_deep_tokens": 11,
                   "retained_support_mass": .8, "kept_context": 12, "score_seconds": .001}]}
    def run(model, ids, config, method):
        calls.append((model, ids, method))
        return copy.deepcopy(outcome)
    def postprocess(tokenizer, ids, given_sample, given_task):
        assert ids == [1, 2] and given_sample == sample and given_task == task
        return "generated", 7
    def prompts(tokenizer, text, given_task, *, preformatted):
        assert given_task == task and preformatted == ("paper_prompt" in sample)
        assert text == sample.get("paper_prompt", sample.get("question", sample.get("prompt")))
        return [8, 9]
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_evaluate", SimpleNamespace(
        run_method=run, postprocess_output=postprocess, math_answer=lambda value: "gold"))
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_common", SimpleNamespace(
        prompt_ids=prompts, extract_answer=lambda value, gold=False: "gold"))
    config = SimpleNamespace(task=task, methods=["flash_native"], gen_length=256)
    actual = paper.make_record("model", "tokenizer", sample, 4, config)
    assert actual == {
        "index": 4, "id": sample.get("id", sample.get("task_id")),
        "target": "gold" if task in ("gsm8k", "math") else sample["task_id"],
        "flash_native": {"nfe": 3, "seconds": 4.0, "peak_gib": 5.0,
                         "backend": {"flash_calls": 6}, "text": "generated",
                         "prediction": "gold" if task in ("gsm8k", "math") else None,
                         "correct": True if task in ("gsm8k", "math") else None,
                         "canvas_tokens": 256, "output_tokens": 7,
                         "deep_tokens": [11], "retained_mass": [.8],
                         "context_tokens": [12], "score_seconds": [.001]},
    }
    assert calls == [("model", [8, 9], "flash_native")]


def test_finalize_restarts_without_duplicate_records_or_raw_changes(campaign, monkeypatch):
    args, job, samples, output = campaign
    seed_legacy(campaign, [row(0)])
    original = (output / "rank_0.jsonl").read_bytes()
    paper.initialize(args)
    item = paper.load_manifest(args.run_root)["jobs"][0]
    queue = paper.queue_for(args.run_root, item)
    dump(output / "elastic_records" / "000001.json",
         row(1, execution={"identity": queue.identity_sha256}))
    aggregated = []
    def aggregate(records, methods):
        aggregated.append([record["index"] for record in records])
        return {method: {"examples": len(records)} for method in methods}
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_evaluate", SimpleNamespace(aggregate=aggregate))
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_common", SimpleNamespace(
        MODEL_ID=MODEL_ID, REVISION=REVISION, load_samples=lambda *unused: samples))
    for _ in range(2):
        paper.finalize(SimpleNamespace(run_root=args.run_root, job=job.name))
    assert aggregated == [[0, 1], [0, 1]]
    assert (output / "rank_0.jsonl").read_bytes() == original
    assert len((output / "rank_elastic.jsonl").read_text().splitlines()) == 1
    assert (output / "summary.before_elastic.json").is_file()
    assert (output.parent / "exit_code").read_text().strip() == "0"


def test_finalize_rejects_incomplete_rows(campaign, monkeypatch):
    args, job, _, _ = campaign
    seed_legacy(campaign, [row(0)])
    paper.initialize(args)
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_evaluate", SimpleNamespace(aggregate=None))
    with pytest.raises(ValueError, match="incomplete"):
        paper.finalize(SimpleNamespace(run_root=args.run_root, job=job.name))


def test_math_finalize_uses_official_scores_and_clean_view(campaign, monkeypatch):
    args, original_job, samples, _ = campaign
    job = Job("math", 256, "llada", "none", "single", "flash_native",
              original_job.dataset, 2)
    output = args.run_root / job.name / "output"
    monkeypatch.setattr(paper, "build_jobs", lambda datasets: [job])
    seed_legacy((args, job, samples, output), [row(0), row(1)])
    raw_path = output / "rank_0.jsonl"
    raw_path.write_text(raw_path.read_text() + '{"unfinished":')
    original_raw = raw_path.read_bytes()
    paper.initialize(args)
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_evaluate", SimpleNamespace(
        aggregate=lambda records, methods: {"flash_native": {
            "examples": 2, "accuracy": 0.125, "mean_seconds": 1.5}}))
    monkeypatch.setitem(sys.modules, "fastv_dllm.llada_common", SimpleNamespace(
        MODEL_ID=MODEL_ID, REVISION=REVISION, load_samples=lambda *unused: samples))
    calls = []
    def score(command, *, check):
        calls.append(command)
        assert check and "--math-verify" in command
        assert command[command.index("--workers") + 1] == "4"
        view = Path(command[command.index("--results") + 1])
        assert view == output / "scoring_input"
        scored_rows = [json.loads(line) for line in (view / "rank_0.jsonl").read_text().splitlines()]
        assert [value["index"] for value in scored_rows] == [0, 1]
        dump(Path(command[command.index("--output") + 1]), {
            "examples": 2,
            "results": {"flash_native": {
                "exact_match": 0.5, "math_verify": 1.0, "valid_extraction_rate": 0.5}},
            "provenance": {"dataset_sha256": paper.digest(job.dataset), "math_verify_version": "0.1.0"},
            "details": [{"id": value["id"]} for value in scored_rows],
        })
    monkeypatch.setattr(paper.subprocess, "run", score)
    for _ in range(2):
        paper.finalize(SimpleNamespace(run_root=args.run_root, job=job.name))
        summary = json.loads((output / "summary.json").read_text())
        metrics = summary["results"]["flash_native"]
        assert metrics["accuracy"] == 0.5
        assert metrics["accuracy_metric"] == "lm_eval.minerva_math.exact_match"
        assert metrics["provisional_accuracy"] == 0.125
        assert metrics["math_verify"] == 1.0
        assert metrics["valid_extraction_rate"] == 0.5
        assert metrics["mean_seconds"] == 1.5
        assert summary["scoring_artifact"] == str(output / "math_exact_match.json")
    assert len(calls) == 1  # repeat finalization reuses verified full coverage
    assert raw_path.read_bytes() == original_raw

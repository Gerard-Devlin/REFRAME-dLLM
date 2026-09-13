import json

import pytest

from benchmark import export_prompts, logged_generation_args, make_gsm_task, model_options


def test_native_and_reframe_commands_share_length_and_only_change_method(tmp_path):
    for method in ("native-serial", "native-prefix-serial", "native-full", "native-prefix", "native-dual", "pair"):
        entry, model, options = model_options(method, "model-Instruct", 256, tmp_path, 16, 2)
        assert "gen_length=256,steps=256,block_length=32" in options
        assert "batch_size" not in options
        assert ("threshold=0.9" in options) == (method not in {"native-serial", "native-prefix-serial"})
        assert entry.startswith("v1/") == method.startswith("native-")
    assert "dual_cache=True" in model_options("native-dual", "m", 256, tmp_path, 16, 2)[2]
    assert "reframe_materialize=True" in model_options("materialize", "m", 256, tmp_path, 16, 2)[2]


def test_replay_uses_logged_fewshot_prompt_and_deduplicates_filters(tmp_path):
    source = tmp_path / "native-dual"
    source.mkdir()
    rows = [dict(doc_id=i, arguments=[(f"five-shot context {i}", {"until": ["Question:"]})])
            for i in (0, 0, 1)]
    (source / "samples_gsm8k_local_time.jsonl").write_text("\n".join(map(json.dumps, rows)))
    target = tmp_path / "prompts.jsonl"
    assert export_prompts(source, "gsm8k", 16, target) == 2
    actual = [json.loads(line) for line in target.read_text().splitlines()]
    assert actual == [dict(id=str(i), prompt=f"five-shot context {i}", until=["Question:"])
                      for i in (0, 1)]


@pytest.mark.parametrize("task", ["gsm8k", "humaneval"])
def test_replay_roundtrip_through_actual_lm_eval_writer(tmp_path, task):
    pytest.importorskip("lm_eval")
    from lm_eval.loggers.evaluation_tracker import EvaluationTracker

    source = tmp_path / "samples"
    tracker = EvaluationTracker(output_path=str(source))
    tracker.general_config_tracker.model_name_sanitized = "test-model"
    tracker.date_id = "replay-fixture"
    prompt = 'Five-shot examples\nQuestion: 中文 with "quotes" and \\slashes\nAnswer:'
    rows = [dict(doc_id=i, arguments=[(prompt + str(i), {"until": ["Question:", "</s>"],
                                                             "temperature": 0.0})],
                 resps=[["#### 42"]], filtered_resps=["42"], target="42") for i in (0, 0, 1)]
    task_name = "gsm8k_local" if task == "gsm8k" else task
    tracker.save_results_samples(task_name, rows)
    written = next(source.rglob(f"samples_{task_name}_*.jsonl"))
    first = json.loads(written.read_text(encoding="utf-8").splitlines()[0])
    assert isinstance(first["arguments"], dict)
    target = tmp_path / "prompts.jsonl"
    assert export_prompts(source, task, 2, target) == 2
    actual = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert actual == [dict(id=str(i), prompt=prompt + str(i),
                          until=["Question:", "</s>"] if task == "gsm8k" else []) for i in (0, 1)]
    with pytest.raises(FileExistsError):
        export_prompts(source, task, 2, target)


@pytest.mark.parametrize("arguments", [None, [], {}, {"gen_args_0": {"arg_0": "prompt"}},
                                        [["prompt", "not kwargs"]], [[None, {}]]])
def test_bad_replay_arguments_fail_with_context(arguments):
    with pytest.raises(ValueError, match="doc_id=7"):
        logged_generation_args(dict(doc_id=7, arguments=arguments))


def test_replay_normalizes_single_stop_and_honors_limit(tmp_path):
    rows = [dict(doc_id=i, arguments={"gen_args_0": {"arg_0": f"exact prompt {i}",
                                                    "arg_1": {"until": "Question:"}}}) for i in (0, 1)]
    (tmp_path / "samples_gsm8k_local_saved.jsonl").write_text("\n".join(map(json.dumps, rows)))
    target = tmp_path / "prompts.jsonl"
    assert export_prompts(tmp_path, "gsm8k", 1, target) == 1
    assert json.loads(target.read_text()) == dict(id="0", prompt="exact prompt 0", until=["Question:"])


def test_local_gsm_task_preserves_original_scoring(tmp_path):
    lm_eval = pytest.importorskip("lm_eval")
    import yaml
    from pathlib import Path
    make_gsm_task(tmp_path)
    original = yaml.safe_load((Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml").read_text())
    actual = yaml.safe_load((tmp_path / "gsm8k_local.yaml").read_text())
    original.update(task="gsm8k_local", dataset_path="openai/gsm8k")
    assert actual == original

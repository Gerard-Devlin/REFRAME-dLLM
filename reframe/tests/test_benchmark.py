import json

import pytest

from benchmark import export_prompts, make_gsm_task, model_options


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


def test_local_gsm_task_preserves_original_scoring(tmp_path):
    lm_eval = pytest.importorskip("lm_eval")
    import yaml
    from pathlib import Path
    make_gsm_task(tmp_path)
    original = yaml.safe_load((Path(lm_eval.__file__).parent / "tasks/gsm8k/gsm8k.yaml").read_text())
    actual = yaml.safe_load((tmp_path / "gsm8k_local.yaml").read_text())
    original.update(task="gsm8k_local", dataset_path="openai/gsm8k")
    assert actual == original

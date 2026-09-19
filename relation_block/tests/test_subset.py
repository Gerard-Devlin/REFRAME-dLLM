import json
import sys
from types import SimpleNamespace
import pytest
from relation_block.download_subset import normalize, tokenized, fingerprint, main


class Tokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        ids = []
        for row in messages:
            ids += [1 if row["role"] == "user" else 2]
            ids += [ord(c) % 100 + 3 for c in row["content"]]
            ids += [0]
        if add_generation_prompt:
            ids += [2]
        return ids

    def convert_tokens_to_ids(self, text):
        return 999


def test_nemotron_format_and_full_answer():
    row = dict(input=[dict(role="user", content="Question")],
               output="<think>reason</think> answer", system_prompt="detailed thinking on")
    messages = normalize(row)
    assert len(messages) == 2
    assert messages[-1]["content"] == row["output"]
    ids, prompt = tokenized(messages, Tokenizer(), 128)
    assert ids[:len(prompt)] == prompt
    assert fingerprint(messages) == fingerprint(normalize(row))
    with pytest.raises(OverflowError):
        tokenized(normalize(dict(input="q", output="x"*200)), Tokenizer(), 128)
    with pytest.raises(ValueError):
        normalize(dict(input=[], output="answer"))


def test_bounded_export_pinning_resume_and_budget(tmp_path, monkeypatch):
    import relation_block.download_subset as module
    import transformers
    import datasets
    import huggingface_hub
    monkeypatch.setattr(module, "snapshot", lambda offline: tmp_path)
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(dataset_info=lambda name: SimpleNamespace(sha="fixed-revision")))
    class Stream:
        def __init__(self, split):
            self.split = split
        def shuffle(self, **kwargs):
            return self
        def __iter__(self):
            for i in range(100):
                yield dict(input=f"{self.split}{i}", output="answer", reasoning="off")
    calls = []
    def load(name, config, split, streaming, revision):
        assert config == "SFT" and streaming and revision == "fixed-revision"
        calls.append(split)
        return Stream(split)
    monkeypatch.setattr(datasets, "load_dataset", load)
    monkeypatch.setattr(sys, "argv", ["subset", "--output", str(tmp_path / "subset"), "--tokens", "256", "--length", "64"])
    main()
    manifest = json.loads((tmp_path / "subset/subset_manifest.json").read_text())
    assert 128 <= manifest["total_tokens"] <= 256
    assert {r["category"] for r in manifest["categories"]} == {"math", "code"}
    assert all(r["tokens"] <= 128 for r in manifest["categories"])
    assert calls == ["math", "code"]
    main()  # Reuse completed categories, no second dataset stream.
    assert calls == ["math", "code"]

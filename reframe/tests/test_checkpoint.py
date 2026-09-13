"""Exercise saved weights + tokenizer through the non--tiny runner path."""
import json
import math

import pytest
import torch

from model.configuration_llada import LLaDAConfig, ModelConfig
from run import load_checkpoint, main


@pytest.fixture
def checkpoint(tiny_model, tmp_path):
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    directory = tmp_path / "checkpoint"
    tiny_model.config.mask_token_id = 127
    # Persist weights that cannot propose MASK, without runtime forward hooks.
    with torch.no_grad():
        head = tiny_model.model.transformer.ff_out.weight
        head[127].zero_()
        head[0].copy_(-head[1])
    tiny_model.save_pretrained(directory)
    config_file = directory / "config.json"
    cfg = json.loads(config_file.read_text())
    cfg.pop("train_max_sequence_length", None)
    cfg["auto_map"] = {"AutoConfig": "configuration_remote.LLaDAConfig"}
    config_file.write_text(json.dumps(cfg), encoding="utf-8")
    (directory / "configuration_remote.py").write_text(
        "from transformers import PretrainedConfig\n"
        "class LLaDAConfig(PretrainedConfig):\n"
        "    model_type = 'llada'\n", encoding="utf-8")
    vocab = {f"t{i}": i for i in range(128)}
    for i, token in ((1, "<unk>"), (126, "</s>"), (127, "<mask>")):
        del vocab[f"t{i}"]
        vocab[token] = i
    inner = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<unk>"))
    inner.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=inner, unk_token="<unk>",
                                       eos_token="</s>", pad_token="</s>", mask_token="<mask>")
    tokenizer.chat_template = "{{ messages[0]['content'] }}"
    tokenizer.save_pretrained(directory)
    return directory


def test_checkpoint_uses_local_config_defaults_and_preserves_weights(checkpoint, tiny_model):
    model, tokenizer, mask_id = load_checkpoint(checkpoint, torch.float32, "cpu", "torch")
    assert type(model.config) is LLaDAConfig
    assert model.config.train_max_sequence_length == ModelConfig().train_max_sequence_length
    assert model.config.max_sequence_length == tiny_model.config.max_sequence_length
    assert mask_id == tokenizer.mask_token_id == 127
    for name, weight in tiny_model.state_dict().items():
        torch.testing.assert_close(model.state_dict()[name], weight, rtol=0, atol=0)
    ids = torch.arange(2, 20).unsqueeze(0)
    with torch.inference_mode():
        torch.testing.assert_close(model(ids).logits, tiny_model(ids).logits, rtol=0, atol=0)


@pytest.mark.parametrize("stage", ["timing", "audit", "oracle"])
def test_non_tiny_runner_from_saved_checkpoint(checkpoint, tmp_path, monkeypatch, stage):
    output = tmp_path / f"{stage}.jsonl"
    argv = ["run.py", "--model-path", str(checkpoint), "--device", "cpu", "--backend", "torch",
            "--prompt", " ".join(f"t{i}" for i in range(2, 18)), "--gen-length", "12",
            "--block-length", "2", "--threshold", "1", "--pilots", "4", "--warmup", "0",
            "--repeats", "1", "--methods", "native-dual,stale,shift,pair", "--output", str(output)]
    if stage == "audit":
        argv += ["--methods", "pair", "--audit-every", "2"]
    elif stage == "oracle":
        argv += ["--oracle"]
    monkeypatch.setattr("sys.argv", argv)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    main()
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows
    if stage == "oracle":
        assert {r["kind"] for r in rows} == {"stale", "shift", "scale", "pair"}
        assert all(r["oracle"] and math.isfinite(r["attention_error"]) for r in rows)
    else:
        expected = {"pair"} if stage == "audit" else {"native-dual", "stale", "shift", "pair"}
        assert {r["method"] for r in rows} == expected
        assert all(not r["tiny"] and 127 not in r["output_ids"] for r in rows)
        if stage == "audit":
            assert rows[0]["stats"]["diagnostic_run"] and rows[0]["stats"]["audits"]

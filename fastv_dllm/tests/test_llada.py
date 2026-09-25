import torch

from fastv_dllm.common import prompt_ids
from fastv_dllm.llada_pruning import Config, choose_support


def test_targets_are_never_pruned():
    relevance = torch.arange(10, dtype=torch.float32)
    keep = choose_support(relevance, targets=[1, 7], keep_ratio=0.25)
    assert 1 in keep and 7 in keep
    assert len(keep) == 4  # two targets plus ceil(8 * .25) support


def test_zero_and_full_support_ratios():
    relevance = torch.randn(8)
    assert choose_support(relevance, [2, 5], 0) == [2, 5]
    assert choose_support(relevance, [2, 5], 1) == list(range(8))


def test_config_rejects_invalid_ratio():
    for value in (-0.1, 1.1):
        try:
            Config(support_keep_ratio=value).validate(32)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid ratio accepted")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return messages[0]["content"]


def test_task_specific_prompt_does_not_leak_gsm_instruction():
    tokenizer = FakeTokenizer()
    assert "####" in prompt_ids(tokenizer, "2+2?", "gsm8k")
    assert prompt_ids(tokenizer, "def f():", "humaneval") == "def f():"

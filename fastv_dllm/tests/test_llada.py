import torch

from fastv_dllm.common import prompt_ids
from fastv_dllm.llada_pruning import Config, choose_support, compress_context


def test_targets_are_never_pruned():
    relevance = torch.arange(10, dtype=torch.float32)
    keep = choose_support(relevance, targets=[1, 7], keep_ratio=0.25)
    assert 1 in keep and 7 in keep
    assert len(keep) == 4  # two targets plus ceil(8 * .25) support


def test_zero_and_full_support_ratios():
    relevance = torch.randn(8)
    assert choose_support(relevance, [2, 5], 0) == [2, 5]
    assert choose_support(relevance, [2, 5], 1) == list(range(8))


def test_known_language_tokens_are_always_protected():
    relevance = torch.tensor([100.0, 1.0, 2.0, 3.0, 4.0, 99.0])
    keep = choose_support(relevance, targets=[4], keep_ratio=0.33,
                          candidates=[1, 2, 3], protected=[0, 5])
    assert keep == [0, 3, 4, 5]


def test_config_rejects_invalid_ratio():
    for value in (-0.1, 1.1):
        try:
            Config(support_keep_ratio=value).validate(32)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid ratio accepted")


def test_context_compression_keeps_dominant_and_contextual_tokens():
    hidden = torch.tensor([[
        [1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9], [1.0, 1.0],
    ]])
    relevance = torch.tensor([0.9, 0.8, 0.1, 0.2, 0.7])
    kept, merged, dominant, contextual = compress_context(
        hidden, relevance, context=[0, 1, 2, 3, 4],
        dominant_ratio=0.4, contextual_ratio=0.2, merge_weight=0.5,
    )
    assert dominant == 2 and contextual == 1
    assert {0, 1}.issubset(kept)
    assert len(kept) == 3 and set(merged).issubset(kept)
    assert all(value.shape == (2,) for value in merged.values())


def test_context_compression_exact_configuration():
    hidden = torch.randn(1, 6, 4)
    kept, merged, dominant, contextual = compress_context(
        hidden, torch.randn(6), range(6), 1.0, 0.0, 0.5,
    )
    assert kept == list(range(6)) and not merged
    assert dominant == 6 and contextual == 0


def test_spatial_support_compression_uses_contiguous_pooling():
    hidden = torch.tensor([[[float(i), 0.0] for i in range(8)]])
    kept, merged, dominant, contextual = compress_context(
        hidden, torch.arange(8, dtype=torch.float32), range(8),
        dominant_ratio=0.0, contextual_ratio=0.25, merge_weight=1.0,
        assignment="spatial",
    )
    assert dominant == 0 and contextual == 2 and len(kept) == 2
    assert torch.allclose(merged[2], torch.tensor([1.5, 0.0]))
    assert torch.allclose(merged[6], torch.tensor([5.5, 0.0]))


def test_config_rejects_invalid_context_parameters():
    for field in ("context_dominant_ratio", "contextual_ratio",
                  "support_contextual_ratio", "context_merge_weight"):
        for value in (-0.1, 1.1):
            kwargs = {field: value}
            try:
                Config(**kwargs).validate(32)
            except ValueError:
                pass
            else:
                raise AssertionError(f"Invalid {field} accepted")


def test_config_rejects_invalid_secondary_prune_point():
    for value in (1, 4, 32):
        try:
            Config(prune_after_layer=4, secondary_prune_after_layer=value).validate(32)
        except ValueError:
            pass
        else:
            raise AssertionError("Invalid secondary prune point accepted")


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize and add_generation_prompt
        return messages[0]["content"]


def test_task_specific_prompt_does_not_leak_gsm_instruction():
    tokenizer = FakeTokenizer()
    assert "####" in prompt_ids(tokenizer, "2+2?", "gsm8k")
    assert prompt_ids(tokenizer, "def f():", "humaneval") == "def f():"

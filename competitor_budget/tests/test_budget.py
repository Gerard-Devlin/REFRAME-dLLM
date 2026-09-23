import pytest

torch = pytest.importorskip("torch")

from competitor_budget.budget import decide


def probabilities(rows):
    return torch.tensor(rows, dtype=torch.float32).unsqueeze(0)


def test_diffuse_runner_up_releases_two_position_prefix():
    p = probabilities([[0.6, 0.1, 0.1, 0.1, 0.1],
                       [0.6, 0.1, 0.1, 0.1, 0.1]])
    result = decide(p, p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 1.0,
                    include_top1_bound=True)
    assert result.baseline.tolist() == [[True, False]]
    assert result.proposed.tolist() == [[True, True]]
    assert result.plusplus.tolist() == [[True, False]]


def test_strong_runner_up_does_not_release_pair():
    p = probabilities([[0.6, 0.4], [0.6, 0.4]])
    result = decide(p, p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 1.0)
    assert result.proposed.tolist() == [[True, False]]


def test_native_set_is_kept_if_it_exceeds_certificate():
    p = probabilities([[0.6, 0.4], [0.6, 0.4]])
    result = decide(p, p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 0.5)
    assert result.baseline.tolist() == [[True, True]]
    assert result.proposed.tolist() == [[True, True]]


def test_filled_positions_cannot_affect_selection():
    p = probabilities([[0.99, 0.01], [0.6, 0.4], [0.6, 0.4]])
    result = decide(p, p.argmax(-1), torch.tensor([[False, True, True]]), 1.0)
    assert result.baseline.tolist() == [[False, True, False]]
    assert result.proposed.tolist() == [[False, True, False]]


def test_non_greedy_tokens_are_rejected():
    p = probabilities([[0.6, 0.4]])
    with pytest.raises(ValueError, match="greedy"):
        decide(p, torch.tensor([[1]]), torch.tensor([[True]]), 1.0)


def test_margin_can_remove_only_the_extension():
    p = probabilities([[0.6, 0.1, 0.1, 0.1, 0.1],
                       [0.6, 0.1, 0.1, 0.1, 0.1]])
    result = decide(p, p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 1.0, margin=0.11)
    assert result.proposed.tolist() == [[True, False]]

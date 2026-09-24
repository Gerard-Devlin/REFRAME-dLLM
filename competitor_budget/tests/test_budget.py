import pytest

torch = pytest.importorskip("torch")

from competitor_budget.budget import CommitGuard, PredictionHistory, decide


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


def test_guards_filter_only_extras_and_keep_most_confident():
    p = probabilities([[0.97, 0.03], [0.93, 0.07], [0.90, 0.10]])
    args = (p, p.argmax(-1), torch.ones((1, 3), dtype=torch.bool), 0.95)
    limited = decide(*args, margin=0.2, guard=CommitGuard(min_confidence=0.85, max_extra=1))
    assert limited.proposed.tolist() == [[True, True, False]]
    floor = decide(*args, guard=CommitGuard(min_confidence=0.94))
    assert torch.equal(floor.proposed, floor.baseline)
    disabled = decide(*args, guard=CommitGuard(max_extra=0))
    assert torch.equal(disabled.proposed, disabled.baseline)


def test_stop_guard_never_removes_native_eos():
    p = probabilities([[0.03, 0.97], [0.07, 0.93], [0.90, 0.10]])
    result = decide(p, p.argmax(-1), torch.ones((1, 3), dtype=torch.bool), 0.95,
                    guard=CommitGuard(protect_stop=True), stop_token=1)
    assert result.baseline.tolist() == [[True, False, False]]
    assert result.proposed.tolist() == [[True, False, True]]


def test_stability_requires_history_and_cannot_suppress_forced_commit():
    p = probabilities([[0.6, 0.1, 0.1, 0.1, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1]])
    args = (p, p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 1.0)
    guard = CommitGuard(min_observations=2)
    with pytest.raises(ValueError, match="historical"):
        decide(*args, guard=guard)
    result = decide(*args, guard=guard, observations=torch.tensor([[0, 1]]))
    assert result.proposed.tolist() == [[True, False]]
    result = decide(*args, guard=guard, observations=torch.tensor([[0, 2]]))
    assert result.proposed.tolist() == [[True, True]]


def test_history_counts_consecutive_confident_predictions_only():
    history = PredictionHistory()
    def observe(tokens, confidence, eligible=(True, True)):
        return history.update(torch.tensor([tokens]), torch.tensor([confidence]),
                              torch.tensor([eligible]), 0.85).tolist()
    assert observe([0, 0], [0.9, 0.9]) == [[1, 1]]
    assert observe([0, 1], [0.9, 0.9]) == [[2, 1]]
    assert observe([0, 1], [0.8, 0.9]) == [[0, 2]]
    assert observe([0, 1], [0.9, 0.9], (True, False)) == [[1, 0]]


def test_guarded_sets_are_subsets_of_original_and_supersets_of_native():
    generator = torch.Generator().manual_seed(7)
    for _ in range(20):
        p = (torch.randn(3, 8, 5, generator=generator) * 3).softmax(-1)
        eligible = torch.rand(3, 8, generator=generator) > 0.2
        args = (p, p.argmax(-1), eligible, 0.95)
        original = decide(*args)
        guarded = decide(*args, guard=CommitGuard(min_confidence=0.85, max_extra=2,
                                                min_observations=2, protect_stop=True),
                         observations=torch.randint(0, 4, (3, 8), generator=generator), stop_token=0)
        assert not (guarded.baseline & ~guarded.proposed).any()
        assert not (guarded.proposed & ~original.proposed).any()
        assert not (guarded.proposed & ~eligible).any()
        assert ((guarded.proposed & ~guarded.baseline).sum(-1) <= 2).all()


def test_bfloat16_top_two_cast_preserves_previous_rule():
    p = probabilities([[0.6, 0.1, 0.1, 0.1, 0.1], [0.6, 0.1, 0.1, 0.1, 0.1]]).bfloat16()
    args = (p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 1.0)
    old = decide(p.float(), *args)
    current = decide(p, *args)
    assert torch.equal(old.proposed, current.proposed)
    assert torch.equal(old.top2, current.top2)


def test_native_bfloat16_threshold_comparison_is_kept():
    p = probabilities([[0.9, 0.1], [0.8, 0.2]]).bfloat16()
    confidence = p.max(-1).values
    native = confidence > 0.8
    native[0, confidence[0].argmax()] = True
    result = decide(p, p.argmax(-1), torch.ones((1, 2), dtype=torch.bool), 0.8)
    assert torch.equal(result.baseline, native)

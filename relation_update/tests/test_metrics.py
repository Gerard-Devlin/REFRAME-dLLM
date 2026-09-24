import json

import pytest
import torch

from relation_update.metrics import finalize, metrics_batch


def example(predicted, teacher, truth, selected, *, candidates=None, eligible=None, base=None):
    predicted = torch.tensor(predicted, dtype=torch.float32)
    b, length, bins = predicted.shape
    if candidates is None:
        candidates = torch.arange(10, 10 + bins - 1).expand(b, length, bins - 1)
    data = {
        "candidates": torch.as_tensor(candidates),
        "base_log_probs": (predicted if base is None else torch.tensor(base)).log(),
        "eligible": torch.ones(b, length, dtype=torch.bool) if eligible is None else torch.tensor(eligible),
        "teacher_probs": torch.tensor(teacher, dtype=torch.float32),
        "teacher_top1": torch.tensor(truth),
        "teacher_selected": torch.tensor(selected),
    }
    return predicted.log(), data


def test_changed_target_outside_k_is_failure_even_when_other_bin_is_largest():
    prediction, data = example(
        [[[.1, .2, .7], [.7, .2, .1]]],
        [[[.05, .05, .9], [.7, .2, .1]]],
        [[99, 10]], [[True, True]],
    )
    result = finalize(metrics_batch(prediction, data))
    assert result["teacher_top1_coverage"] == .5
    assert result["changed_teacher_top1_coverage"] == 0
    assert result["next_top1_agreement"] == .5
    assert result["changed_top1_agreement"] == 0
    assert result["changed_covered_top1_agreement"] is None
    assert result["stale_top1_agreement"] == .5
    assert result["counts"]["changed_tokens"] == 1


def test_large_tail_cannot_inflate_candidate_confidence_by_renormalization():
    prediction, data = example(
        [[[.1, .001, .899], [.8, .1, .1]]],
        [[[.1, .001, .899], [.8, .1, .1]]],
        [[10, 10]], [[False, True]],
    )
    # Renormalizing the first position within K would incorrectly cross .9.
    result = finalize(metrics_batch(prediction, data, threshold=.9))
    assert result["counts"]["proposed_tokens"] == 1
    assert result["proposal_precision"] == 1
    assert result["commit_recall"] == 1
    assert result["exact_transition_agreement"] == 1
    assert result["predicted_other_mass"] == pytest.approx(.4995)


def test_perfect_conditioned_prediction_recovers_changes_and_native_commits():
    teacher = [[[.03, .95, .02], [.8, .1, .1], [.96, .02, .02]]]
    prediction, data = example(
        teacher, teacher, [[11, 10, 10]], [[True, False, True]],
        base=[[[.8, .1, .1], [.8, .1, .1], [.8, .1, .1]]],
    )
    result = finalize(metrics_batch(prediction, data))
    assert result["mean_kl"] == pytest.approx(0, abs=1e-7)
    assert result["base_mean_kl"] > 0
    assert result["next_top1_agreement"] == 1
    assert result["changed_top1_agreement"] == 1
    assert result["changed_covered_top1_agreement"] == 1
    assert result["stale_top1_agreement"] == pytest.approx(2 / 3)
    assert result["proposal_precision"] == result["commit_recall"] == 1
    assert result["exact_transition_agreement"] == 1
    assert result["stale_exact_transition_agreement"] == 0


def test_committed_positions_must_match_identity_not_only_position():
    prediction, data = example([[[.96, .02, .02]]], [[[.02, .96, .02]]], [[11]], [[True]])
    result = finalize(metrics_batch(prediction, data))
    assert result["proposal_position_precision"] == 1
    assert result["proposal_precision"] == 0
    assert result["commit_recall"] == 0
    assert result["exact_transition_agreement"] == 0


def test_no_eligible_tokens_are_counted_without_vacuous_success():
    prediction, data = example([[[.8, .1, .1]]], [[[.8, .1, .1]]], [[10]], [[False]], eligible=[[False]])
    raw = metrics_batch(prediction, data)
    result = finalize(raw)
    assert raw["transitions"] == raw["empty_transitions"] == 1
    assert raw["eligible_tokens"] == raw["proposed_tokens"] == 0
    assert result["mean_kl"] is None
    assert result["exact_transition_agreement"] is None
    json.dumps(result, allow_nan=False)


def test_eos_teacher_selection_is_included_and_reported():
    prediction, data = example(
        [[[.03, .95, .02], [.96, .02, .02]]],
        [[[.03, .95, .02], [.96, .02, .02]]],
        [[151645, 10]], [[True, True]],
        candidates=[[[10, 151645], [10, 11]]],
    )
    result = finalize(metrics_batch(prediction, data))
    assert result["counts"]["teacher_eos_tokens"] == 1
    assert result["counts"]["proposed_tokens"] == 2
    assert result["eos_proposal_precision"] == result["eos_commit_recall"] == 1
    assert result["eos_exact_transition_agreement"] == 1


def test_metrics_are_additive_and_token_weighted_across_uneven_batches():
    prediction, data = example(
        [[[.8, .1, .1], [.8, .1, .1]], [[.03, .95, .02], [.8, .1, .1]]],
        [[[.8, .1, .1], [.8, .1, .1]], [[.03, .95, .02], [.8, .1, .1]]],
        [[10, 10], [11, 10]], [[True, False], [True, False]],
        eligible=[[True, True], [True, False]],
    )
    full = metrics_batch(prediction, data)
    shards = [metrics_batch(prediction[i:i + 1], {k: v[i:i + 1] for k, v in data.items()}) for i in range(2)]
    combined = {k: sum(shard[k] for shard in shards) for k in full}
    assert combined == pytest.approx(full)
    assert finalize(combined)["mean_kl"] == pytest.approx(finalize(full)["mean_kl"])


def test_separate_teacher_targets_are_supported_without_injecting_into_inputs():
    prediction, data = example([[[.8, .1, .1]]], [[[.8, .1, .1]]], [[10]], [[True]])
    targets = [data.pop(key) for key in ("teacher_probs", "teacher_top1", "teacher_selected")]
    assert finalize(metrics_batch(prediction, data, *targets))["exact_transition_agreement"] == 1


def test_teacher_selection_outside_eligibility_is_not_silently_ignored():
    prediction, data = example([[[.8, .1, .1]]], [[[.8, .1, .1]]], [[10]], [[True]], eligible=[[False]])
    with pytest.raises(ValueError, match="outside"):
        metrics_batch(prediction, data)

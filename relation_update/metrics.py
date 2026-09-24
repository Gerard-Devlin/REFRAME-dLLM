"""Token-weighted, additive diagnostics for one native conditional update.

These metrics compare against the frozen teacher's next call, not task answers.
OTHER represents many vocabulary items: it may carry probability mass but is
never a token candidate and never participates in the candidate argmax.
"""

import torch


def _selection(log_probs, candidates, eligible, threshold):
    """Native threshold + forced best position, with full-vocabulary mass."""
    candidate_log_probs, indices = log_probs[..., :-1].max(-1)
    prediction = candidates.gather(-1, indices.unsqueeze(-1)).squeeze(-1)
    confidence = candidate_log_probs.exp()
    scores = torch.where(eligible, confidence, -torch.inf)
    selected = (scores > threshold) & eligible
    active = eligible.any(-1)
    rows = active.nonzero(as_tuple=True)[0]
    selected[rows, scores.argmax(-1)[rows]] = True
    return prediction, selected


def _kl(teacher_probs, log_probs):
    # A true zero in the teacher has zero contribution, even if prediction is
    # also zero. Do not smooth away an impossible prediction of a nonzero bin.
    positive = teacher_probs > 0
    terms = torch.where(
        positive,
        teacher_probs * (teacher_probs.clamp_min(1e-30).log() - log_probs),
        torch.zeros_like(teacher_probs),
    )
    return terms.sum(-1)


@torch.no_grad()
def metrics_batch(log_probs, batch, teacher_probs=None, teacher_top1=None,
                  teacher_selected=None, threshold=0.9, stop_id=151645):
    """Return sums/counts suitable for elementwise distributed SUM reduction.

    ``batch`` contains only current-state inputs; targets may be passed as
    separate arguments. For convenience, omitted targets are read from batch.
    Candidates are chosen in the current state, with its true argmax first.
    No target is inserted into that candidate set, including uncovered targets.

    Probabilities have K actual candidate bins plus one aggregate OTHER bin.
    Only positions in ``eligible`` contribute token metrics. Empty transitions
    are counted separately and do not earn vacuous exact-match credit.
    """
    candidates = batch["candidates"]
    base_log_probs = batch["base_log_probs"].float()
    eligible = batch["eligible"].bool()
    teacher_probs = batch["teacher_probs"] if teacher_probs is None else teacher_probs
    teacher_top1 = batch["teacher_top1"] if teacher_top1 is None else teacher_top1
    teacher_selected = batch["teacher_selected"] if teacher_selected is None else teacher_selected
    teacher_probs, teacher_selected = teacher_probs.float(), teacher_selected.bool()
    log_probs = log_probs.float()
    if candidates.ndim != 3 or candidates.shape[-1] < 1:
        raise ValueError("Candidates must have shape [batch, positions, K] with K >= 1")
    position_shape = candidates.shape[:2]
    probability_shape = (*position_shape, candidates.shape[-1] + 1)
    if (log_probs.shape != probability_shape or base_log_probs.shape != probability_shape
            or teacher_probs.shape != probability_shape):
        raise ValueError("Predicted, base and teacher distributions must contain K + OTHER bins")
    if any(value.shape != position_shape for value in (eligible, teacher_top1, teacher_selected)):
        raise ValueError("Eligibility and teacher token/commit targets must match [batch, positions]")
    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1")
    if bool((teacher_selected & ~eligible).any()):
        raise ValueError("Teacher selected a position outside the recorded next native sub-block")

    current_top1 = candidates[..., 0]
    changed = eligible & (teacher_top1 != current_top1)
    covered = (candidates == teacher_top1.unsqueeze(-1)).any(-1) & eligible
    active = eligible.any(-1)
    prediction, proposed = _selection(log_probs, candidates, eligible, threshold)
    stale_prediction, stale_proposed = _selection(base_log_probs, candidates, eligible, threshold)
    correct = (prediction == teacher_top1) & eligible
    stale_correct = (current_top1 == teacher_top1) & eligible
    kl, base_kl = _kl(teacher_probs, log_probs), _kl(teacher_probs, base_log_probs)
    hit = proposed & teacher_selected & correct
    stale_hit = stale_proposed & teacher_selected & (stale_prediction == teacher_top1)
    exact = (proposed == teacher_selected).all(-1) & ((prediction == teacher_top1) | ~teacher_selected).all(-1) & active
    stale_exact = ((stale_proposed == teacher_selected).all(-1)
                   & ((stale_prediction == teacher_top1) | ~teacher_selected).all(-1) & active)
    teacher_eos = teacher_selected & (teacher_top1 == stop_id)
    proposed_eos = proposed & (prediction == stop_id)
    teacher_eos_rows = teacher_eos.any(-1) & active

    totals = {
        "transitions": torch.tensor(position_shape[0], device=eligible.device),
        "active_transitions": active.sum(),
        "empty_transitions": (~active).sum(),
        "eligible_tokens": eligible.sum(),
        "changed_tokens": changed.sum(),
        "covered_tokens": covered.sum(),
        "changed_covered_tokens": (changed & covered).sum(),
        "top1_correct_tokens": correct.sum(),
        "changed_top1_correct_tokens": (correct & changed).sum(),
        "stale_top1_correct_tokens": stale_correct.sum(),
        "kl_sum": kl[eligible].sum(),
        "base_kl_sum": base_kl[eligible].sum(),
        "changed_kl_sum": kl[changed].sum(),
        "changed_base_kl_sum": base_kl[changed].sum(),
        "teacher_topk_mass_sum": teacher_probs[..., :-1].sum(-1)[eligible].sum(),
        "predicted_other_mass_sum": log_probs[..., -1].exp()[eligible].sum(),
        "teacher_selected_tokens": teacher_selected.sum(),
        "proposed_tokens": proposed.sum(),
        "correct_proposed_tokens": hit.sum(),
        "proposed_position_matches": (proposed & teacher_selected).sum(),
        "exact_transitions": exact.sum(),
        "stale_proposed_tokens": stale_proposed.sum(),
        "stale_correct_proposed_tokens": stale_hit.sum(),
        "stale_exact_transitions": stale_exact.sum(),
        "teacher_eos_tokens": teacher_eos.sum(),
        "proposed_eos_tokens": proposed_eos.sum(),
        "correct_proposed_eos_tokens": (proposed_eos & teacher_eos).sum(),
        "teacher_eos_transitions": teacher_eos_rows.sum(),
        "exact_eos_transitions": (exact & teacher_eos_rows).sum(),
    }
    # One device-to-host transfer for the whole batch of scalar diagnostics.
    values = torch.stack([value.double() for value in totals.values()]).cpu().tolist()
    return dict(zip(totals, values))


def finalize(totals):
    """Convert globally reduced sums to ratios; undefined ratios become null."""
    totals = {key: float(value) for key, value in totals.items()}

    def ratio(numerator, denominator):
        count = totals.get(denominator, 0)
        return totals.get(numerator, 0) / count if count else None

    report = {
        "mean_kl": ratio("kl_sum", "eligible_tokens"),
        "base_mean_kl": ratio("base_kl_sum", "eligible_tokens"),
        "changed_mean_kl": ratio("changed_kl_sum", "changed_tokens"),
        "changed_base_mean_kl": ratio("changed_base_kl_sum", "changed_tokens"),
        "teacher_changed_fraction": ratio("changed_tokens", "eligible_tokens"),
        "teacher_top1_coverage": ratio("covered_tokens", "eligible_tokens"),
        "changed_teacher_top1_coverage": ratio("changed_covered_tokens", "changed_tokens"),
        "next_top1_agreement": ratio("top1_correct_tokens", "eligible_tokens"),
        "changed_top1_agreement": ratio("changed_top1_correct_tokens", "changed_tokens"),
        "changed_covered_top1_agreement": ratio("changed_top1_correct_tokens", "changed_covered_tokens"),
        "stale_top1_agreement": ratio("stale_top1_correct_tokens", "eligible_tokens"),
        "teacher_topk_mass": ratio("teacher_topk_mass_sum", "eligible_tokens"),
        "predicted_other_mass": ratio("predicted_other_mass_sum", "eligible_tokens"),
        "proposal_precision": ratio("correct_proposed_tokens", "proposed_tokens"),
        "commit_recall": ratio("correct_proposed_tokens", "teacher_selected_tokens"),
        "proposal_position_precision": ratio("proposed_position_matches", "proposed_tokens"),
        "commit_position_recall": ratio("proposed_position_matches", "teacher_selected_tokens"),
        "exact_transition_agreement": ratio("exact_transitions", "active_transitions"),
        "stale_proposal_precision": ratio("stale_correct_proposed_tokens", "stale_proposed_tokens"),
        "stale_commit_recall": ratio("stale_correct_proposed_tokens", "teacher_selected_tokens"),
        "stale_exact_transition_agreement": ratio("stale_exact_transitions", "active_transitions"),
        "eos_proposal_precision": ratio("correct_proposed_eos_tokens", "proposed_eos_tokens"),
        "eos_commit_recall": ratio("correct_proposed_eos_tokens", "teacher_eos_tokens"),
        "eos_exact_transition_agreement": ratio("exact_eos_transitions", "teacher_eos_transitions"),
        "counts": totals,
        "scope": "Held-out native next-call prediction only; not task accuracy or end-to-end speedup. "
                 "Coverage and changed-position agreement include teacher targets outside current top-K as failures. "
                 "Commit agreement includes token identity; OTHER cannot be committed. EOS targets are included.",
    }
    return report

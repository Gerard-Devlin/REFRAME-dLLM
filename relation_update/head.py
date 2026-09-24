"""Small conditional heads over a frozen current top-K + OTHER partition.

Only the current state and the actual newly committed tokens are inputs. Teacher
probabilities belong to the loss, never to the forward pass. OTHER retains the
probability mass outside the current candidates; it is not an extra vocabulary
token and this head cannot invent a candidate absent from that partition.
"""
from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class HeadConfig:
    hidden_size: int
    feature_size: int
    width: int = 64
    mode: str = "relation"

    def __post_init__(self):
        if min(self.hidden_size, self.feature_size, self.width) < 1:
            raise ValueError("Head dimensions must be positive")
        if self.mode not in {"relation", "blind", "calibration"}:
            raise ValueError(f"Unknown head mode: {self.mode}")


def _frozen_float(value):
    return value.detach().float()


def _normalize_features(value):
    # Frozen embeddings and residual-stream states have very different scales.
    # No trainable normalization parameters are needed to compare their content.
    return F.layer_norm(_frozen_float(value), (value.shape[-1],))


class ResidualHead(nn.Module):
    """Predict conditional log-probability changes without a vocabulary-sized head.

    ``relation`` and ``blind`` have identical parameters and computation graphs.
    The blind control zeros *both* token features and commit locations, so it
    cannot infer the intervention from a presence mask or relative positions.
    ``calibration`` is a smaller control that only rescales existing scores and
    adjusts the probability assigned to OTHER, without seeing the intervention.
    """

    def __init__(self, config: HeadConfig):
        super().__init__()
        self.config = config
        w = config.width
        self.hidden_projection = nn.Linear(config.hidden_size, w)
        if config.mode == "calibration":
            self.calibration = nn.Sequential(nn.SiLU(), nn.Linear(w, 2))
            nn.init.zeros_(self.calibration[-1].weight)
            nn.init.zeros_(self.calibration[-1].bias)
            return

        self.candidate_projection = nn.Linear(config.feature_size, w, bias=False)
        self.commit_key = nn.Linear(config.feature_size, w, bias=False)
        self.commit_value = nn.Linear(config.feature_size, w, bias=False)
        self.commit_query = nn.Linear(w, w, bias=False)
        self.relative_bias = nn.Linear(2, 1, bias=False)
        self.relative_value = nn.Linear(2, w, bias=False)
        self.message_projection = nn.Linear(w + 1, w)
        self.score_projection = nn.Linear(1, w)
        self.candidate_output = nn.Sequential(nn.SiLU(), nn.Linear(w, 1))
        self.tail_output = nn.Sequential(nn.SiLU(), nn.Linear(w, 1))
        # Zero residual gives the original current partition, including its tail.
        for output in (self.candidate_output[-1], self.tail_output[-1]):
            nn.init.zeros_(output.weight)
            nn.init.zeros_(output.bias)

    def _inputs(self, batch):
        hidden = batch["hidden"]
        features = batch["candidate_features"]
        base = batch["base_log_probs"]
        commits = batch["commit_features"]
        committed = batch["committed"]
        if hidden.ndim != 3 or hidden.shape[-1] != self.config.hidden_size:
            raise ValueError("hidden must have shape [B,L,hidden_size]")
        b, length, _ = hidden.shape
        if (features.ndim != 4 or features.shape[:2] != (b, length)
                or features.shape[-1] != self.config.feature_size):
            raise ValueError("candidate_features must have shape [B,L,K,feature_size]")
        k = features.shape[2]
        if k < 1 or base.shape != (b, length, k + 1):
            raise ValueError("base_log_probs must contain K candidates and an OTHER bucket")
        if commits.shape != (b, length, self.config.feature_size):
            raise ValueError("commit_features must have shape [B,L,feature_size]")
        if committed.shape != (b, length) or committed.dtype != torch.bool:
            raise ValueError("committed must be a boolean [B,L] tensor")
        if "eligible" in batch and (batch["eligible"].shape != (b, length)
                                     or batch["eligible"].dtype != torch.bool):
            raise ValueError("eligible must be a boolean [B,L] tensor")
        if "candidates" in batch and batch["candidates"].shape != (b, length, k):
            raise ValueError("candidates must have shape [B,L,K]")
        # Clipping -inf for exact zero-probability buckets keeps KL/gradients
        # finite. The added mass is at most FP32's smallest normal magnitude.
        base = _frozen_float(base).clamp_min(math.log(torch.finfo(torch.float32).tiny))
        return hidden, features, base, commits, committed

    def _commit_message(self, hidden, commits, committed):
        b, length, _ = hidden.shape
        features = _normalize_features(commits)
        mask = committed.detach()
        if self.config.mode == "blind":
            features = torch.zeros_like(features)
            mask = torch.zeros_like(mask)

        positions = torch.arange(length, device=hidden.device, dtype=torch.float32)
        displacement = (positions[None, :] - positions[:, None]) / max(length - 1, 1)
        relative = torch.stack((displacement, displacement.abs()), dim=-1)
        keys, values = self.commit_key(features), self.commit_value(features)
        query = self.commit_query(hidden)
        scores = torch.einsum("blw,bjw->blj", query, keys) / math.sqrt(self.config.width)
        scores = scores + self.relative_bias(relative).squeeze(-1)[None]
        # An all-false mask must produce exactly zero, not an all--inf softmax.
        scores = scores.masked_fill(~mask[:, None, :], -1e4)
        weights = scores.softmax(-1) * mask[:, None, :].float()
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-12)
        message = torch.einsum("blj,bjw->blw", weights, values)
        message = message + torch.einsum("blj,ljw->blw", weights, self.relative_value(relative))
        density = mask.float().mean(-1)[:, None, None].expand(b, length, 1)
        return self.message_projection(torch.cat((message, density), dim=-1))

    def forward(self, batch):
        hidden, features, base, commits, committed = self._inputs(batch)
        hidden = self.hidden_projection(_normalize_features(hidden))
        # Keep score arithmetic FP32 even when callers use mixed precision.
        score_features = base.clamp(-30.0, 0.0)
        if self.config.mode == "calibration":
            parameters = self.calibration(hidden).float()
            inverse_temperature_delta = parameters[..., :1].clamp(-5, 5).expm1()
            residual = inverse_temperature_delta * base
            tail_bias = F.pad(parameters[..., 1:], (base.shape[-1] - 1, 0))
            return F.log_softmax(base + residual + tail_bias, dim=-1)

        context = hidden + self._commit_message(hidden, commits, committed)
        candidates = self.candidate_projection(_normalize_features(features))
        candidate_state = (context[:, :, None, :] + candidates
                           + self.score_projection(score_features[..., :-1, None]))
        candidate_residual = self.candidate_output(candidate_state).squeeze(-1)
        tail_state = context + self.score_projection(score_features[..., -1:])
        tail_residual = self.tail_output(tail_state)
        residual = torch.cat((candidate_residual, tail_residual), dim=-1).float()
        return F.log_softmax(base + residual, dim=-1)


def distillation_loss(log_probs, teacher_probs, eligible):
    """Return (summed teacher-to-head KL, eligible token count), including OTHER.

    The caller divides globally reduced sums/counts to obtain a token-weighted
    objective. A batch without eligible tokens returns a differentiable zero.
    """
    if log_probs.ndim != 3 or teacher_probs.shape != log_probs.shape:
        raise ValueError("Teacher and head distributions must have shape [B,L,K+1]")
    if eligible.shape != log_probs.shape[:2] or eligible.dtype != torch.bool:
        raise ValueError("eligible must be a boolean [B,L] tensor")
    prediction = log_probs.float()[eligible]
    teacher = _frozen_float(teacher_probs)[eligible]
    teacher_log = teacher.clamp_min(torch.finfo(torch.float32).tiny).log()
    terms = teacher * (teacher_log - prediction)
    return terms.sum(), eligible.sum()

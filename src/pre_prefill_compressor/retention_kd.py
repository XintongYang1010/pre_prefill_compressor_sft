"""Generic retention-distillation objectives for compressor-only training."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class RetentionKDConfig:
    """Weights for a task anchor plus representation-retention objectives."""

    task_weight: float = 1.0
    response_jsd_weight: float = 1.0
    vision_semantic_weight: float = 0.1
    vision_language_affinity_weight: float = 0.1
    feature_weight: float = 0.1
    jsd_beta: float = 0.5
    temperature: float = 1.0
    class_balance_beta: float = 0.99
    maximum_class_weight: float = 4.0

    def __post_init__(self) -> None:
        weights = (
            self.task_weight,
            self.response_jsd_weight,
            self.vision_semantic_weight,
            self.vision_language_affinity_weight,
            self.feature_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in weights):
            raise ValueError("all objective weights must be finite and positive")
        if not 0.0 <= self.jsd_beta <= 1.0:
            raise ValueError("jsd_beta must be in [0, 1]")
        if self.temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= self.class_balance_beta < 1.0:
            raise ValueError("class_balance_beta must be in [0, 1)")
        if self.maximum_class_weight < 1.0:
            raise ValueError("maximum_class_weight must be at least one")


@dataclass(frozen=True)
class EffectiveNumberWeights:
    sample_weights: torch.Tensor
    class_weights: torch.Tensor
    class_counts: torch.Tensor


@dataclass(frozen=True)
class FeatureDistillationLoss:
    loss: torch.Tensor
    normalized_mse: torch.Tensor
    cosine_distance: torch.Tensor


def effective_number_weights(
    labels: torch.Tensor,
    *,
    beta: float = 0.99,
    maximum_weight: float = 4.0,
    num_classes: int | None = None,
) -> EffectiveNumberWeights:
    """Return mean-one sample weights based on each class's effective count.

    The implementation supports any non-negative integer class labels.  A
    water-filling scale preserves a sample-weight mean of one while respecting
    ``maximum_weight`` for rare classes.
    """

    labels = labels.reshape(-1).long()
    if labels.numel() == 0 or bool((labels < 0).any()):
        raise ValueError("labels must be a non-empty tensor of non-negative integers")
    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    if maximum_weight < 1.0:
        raise ValueError("maximum_weight must be at least one")
    inferred_classes = int(labels.max().item()) + 1
    if num_classes is None:
        num_classes = inferred_classes
    if num_classes < inferred_classes or num_classes <= 0:
        raise ValueError("num_classes does not cover all labels")

    counts = torch.bincount(labels, minlength=num_classes).to(dtype=torch.float64)
    observed = counts > 0
    raw = torch.zeros_like(counts)
    if beta == 0.0:
        raw[observed] = 1.0
    else:
        beta_tensor = torch.tensor(beta, dtype=counts.dtype)
        raw[observed] = (1.0 - beta_tensor) / (1.0 - beta_tensor.pow(counts[observed]))

    total = float(counts.sum().item())

    def weighted_mass(scale: float) -> float:
        scaled = torch.minimum(
            raw * scale,
            torch.full_like(raw, float(maximum_weight)),
        )
        return float((counts * scaled).sum().item())

    lower, upper = 0.0, 1.0
    while weighted_mass(upper) < total:
        upper *= 2.0
        if not math.isfinite(upper):
            raise FloatingPointError("unable to normalize class weights")
    for _ in range(80):
        middle = (lower + upper) / 2.0
        if weighted_mass(middle) < total:
            lower = middle
        else:
            upper = middle
    class_weights = torch.minimum(
        raw * upper,
        torch.full_like(raw, float(maximum_weight)),
    ).to(device=labels.device, dtype=torch.float32)
    sample_weights = class_weights.index_select(0, labels)
    if not torch.isclose(
        sample_weights.mean(), sample_weights.new_tensor(1.0), atol=1e-6
    ):
        raise RuntimeError("sample weights are not mean-normalized")
    return EffectiveNumberWeights(
        sample_weights=sample_weights,
        class_weights=class_weights,
        class_counts=counts.to(device=labels.device, dtype=torch.long),
    )


def generalized_jsd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    labels: torch.Tensor | None = None,
    beta: float = 0.5,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generalized JSD with forward/reverse-KL endpoint conventions.

    ``labels == -100`` masks positions.  Teacher tensors are always detached.
    """

    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if student_logits.ndim < 2:
        raise ValueError("logits must include a vocabulary/class dimension")
    if not 0.0 <= beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    student_log_probs = F.log_softmax(student_logits.float() / temperature, dim=-1)
    teacher_log_probs = F.log_softmax(
        teacher_logits.detach().float() / temperature,
        dim=-1,
    )
    if beta == 0.0:
        token_divergence = F.kl_div(
            student_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
    elif beta == 1.0:
        token_divergence = F.kl_div(
            teacher_log_probs,
            student_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
    else:
        beta_tensor = student_log_probs.new_tensor(beta)
        mixture_log_probs = torch.logsumexp(
            torch.stack(
                (
                    student_log_probs + torch.log1p(-beta_tensor),
                    teacher_log_probs + torch.log(beta_tensor),
                )
            ),
            dim=0,
        )
        teacher_kl = F.kl_div(
            mixture_log_probs,
            teacher_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        student_kl = F.kl_div(
            mixture_log_probs,
            student_log_probs,
            reduction="none",
            log_target=True,
        ).sum(dim=-1)
        token_divergence = beta_tensor * teacher_kl + (1.0 - beta_tensor) * student_kl
    if labels is not None:
        if labels.shape != student_logits.shape[:-1]:
            raise ValueError("labels must match all non-class logits dimensions")
        valid = labels != -100
        if not bool(valid.any()):
            return token_divergence.sum() * 0.0
        token_divergence = token_divergence[valid]
    return token_divergence.mean()


def group_mean(
    teacher_values: torch.Tensor,
    source_to_compressed: torch.Tensor,
    *,
    compressed_tokens: int,
) -> torch.Tensor:
    """Pool source-token teacher values using an explicit compressor mapping."""

    if teacher_values.ndim < 2:
        raise ValueError("teacher_values must have shape [tokens, ...]")
    mapping = source_to_compressed.reshape(-1).long().to(teacher_values.device)
    if mapping.numel() != teacher_values.shape[0]:
        raise ValueError("source_to_compressed length does not match teacher values")
    if compressed_tokens <= 0:
        raise ValueError("compressed_tokens must be positive")
    if int(mapping.min().item()) < 0 or int(mapping.max().item()) >= compressed_tokens:
        raise ValueError("source_to_compressed contains an out-of-range group")
    flat = teacher_values.reshape(teacher_values.shape[0], -1)
    pooled = flat.new_zeros((compressed_tokens, flat.shape[1]))
    pooled.index_add_(0, mapping, flat)
    counts = torch.bincount(mapping, minlength=compressed_tokens).to(flat.dtype)
    if bool((counts == 0).any()):
        raise ValueError("every compressed token must receive a source token")
    pooled = pooled / counts.unsqueeze(-1)
    return pooled.reshape(compressed_tokens, *teacher_values.shape[1:])


def normalized_feature_distillation_loss(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> FeatureDistillationLoss:
    """Match feature direction while ignoring response magnitude."""

    if student.shape != teacher.shape:
        raise ValueError("student and teacher features must have the same shape")
    student_normalized = F.normalize(student.float(), dim=-1, eps=eps)
    teacher_normalized = F.normalize(teacher.detach().float(), dim=-1, eps=eps)
    squared_distance = (student_normalized - teacher_normalized).square().sum(dim=-1)
    cosine_distance = 1.0 - (student_normalized * teacher_normalized).sum(dim=-1)
    normalized_mse = squared_distance.mean()
    cosine = cosine_distance.mean()
    return FeatureDistillationLoss(
        loss=normalized_mse,
        normalized_mse=normalized_mse,
        cosine_distance=cosine,
    )


def vision_semantic_distillation_loss(
    student_vision_logits: torch.Tensor,
    teacher_vision_logits: torch.Tensor,
    *,
    source_to_compressed: torch.Tensor | None = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Reverse-KL semantic retention after deterministic spatial alignment."""

    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    if source_to_compressed is not None:
        teacher_vision_logits = group_mean(
            teacher_vision_logits,
            source_to_compressed,
            compressed_tokens=student_vision_logits.shape[0],
        )
    if student_vision_logits.shape != teacher_vision_logits.shape:
        raise ValueError("aligned student and teacher vision logits must match")
    student_log_probs = F.log_softmax(
        student_vision_logits.float() / temperature, dim=-1
    )
    teacher_log_probs = F.log_softmax(
        teacher_vision_logits.detach().float() / temperature,
        dim=-1,
    )
    student_probs = student_log_probs.exp()
    return (student_probs * (student_log_probs - teacher_log_probs)).sum(dim=-1).mean()


def vision_language_affinity_distillation_loss(
    student_vision: torch.Tensor,
    teacher_vision: torch.Tensor,
    student_language: torch.Tensor,
    teacher_language: torch.Tensor,
    *,
    source_to_compressed: torch.Tensor | None = None,
) -> torch.Tensor:
    """Preserve the visual-token/language-token cosine-affinity matrix."""

    if source_to_compressed is not None:
        teacher_vision = group_mean(
            teacher_vision,
            source_to_compressed,
            compressed_tokens=student_vision.shape[0],
        )
    if student_vision.shape != teacher_vision.shape:
        raise ValueError("aligned student and teacher vision features must match")
    if student_language.shape != teacher_language.shape:
        raise ValueError("student and teacher language features must match")
    if student_vision.shape[-1] != student_language.shape[-1]:
        raise ValueError("vision and language features must share a hidden dimension")
    student_affinity = F.normalize(student_vision.float(), dim=-1) @ F.normalize(
        student_language.float(), dim=-1
    ).transpose(0, 1)
    teacher_affinity = F.normalize(
        teacher_vision.detach().float(), dim=-1
    ) @ F.normalize(teacher_language.detach().float(), dim=-1).transpose(0, 1)
    return F.smooth_l1_loss(student_affinity, teacher_affinity)


def weighted_classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    sample_weights: torch.Tensor | None = None,
    *,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Hard-label CE for either classification or token-level SFT.

    ``logits`` may be ``[batch, classes]`` or ``[batch, ..., classes]``;
    ``labels`` must match every dimension except the last.  ``sample_weights``
    contains one task/class-balance weight per batch member and is broadcast
    over its valid target tokens.  Ignored tokens never enter the denominator.
    """

    if logits.ndim < 2 or labels.shape != logits.shape[:-1]:
        raise ValueError("labels must match all logits dimensions except classes")
    if logits.shape[0] == 0:
        raise ValueError("the batch dimension must be non-empty")
    valid = labels != ignore_index
    if not bool(valid.any()):
        return logits.sum() * 0.0
    losses = F.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        labels.long().reshape(-1),
        reduction="none",
        ignore_index=ignore_index,
    ).reshape_as(labels)
    if sample_weights is None:
        return losses[valid].mean()
    if sample_weights.shape != labels.shape[:1]:
        raise ValueError("sample_weights must contain one value per batch member")
    broadcast_shape = (labels.shape[0],) + (1,) * (labels.ndim - 1)
    broadcast_weights = (
        sample_weights.to(losses).reshape(broadcast_shape).expand_as(losses)
    )
    valid_weights = broadcast_weights[valid]
    return (losses[valid] * valid_weights).sum() / valid_weights.sum().clamp_min(1e-12)

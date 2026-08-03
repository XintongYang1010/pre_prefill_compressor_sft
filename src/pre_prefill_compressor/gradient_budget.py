"""Bounded gradient budgeting for multi-objective retention training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
from torch import nn


@dataclass(frozen=True)
class GradientBudgetConfig:
    ema_beta: float = 0.9
    minimum_weight: float = 0.25
    maximum_weight: float = 4.0
    activation_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        if not 0.0 <= self.ema_beta < 1.0:
            raise ValueError("ema_beta must be in [0, 1)")
        if self.minimum_weight <= 0.0:
            raise ValueError("minimum_weight must be positive")
        if self.maximum_weight < self.minimum_weight:
            raise ValueError("maximum_weight must be at least minimum_weight")
        if not math.isfinite(self.activation_epsilon) or self.activation_epsilon <= 0.0:
            raise ValueError("activation_epsilon must be finite and positive")


@dataclass(frozen=True)
class GradientCapAudit:
    target_key: str
    reference_key: str
    maximum_ratio: float
    reference_weighted_norm: float
    target_uncapped_weighted_norm: float
    target_capped_weighted_norm: float
    actual_ratio: float
    cap_applied: bool


@dataclass(frozen=True)
class ObjectiveGradientSet:
    """Per-objective compressor gradients after optional DP world averaging."""

    gradients: dict[str, tuple[torch.Tensor, ...]]
    norms: dict[str, float]
    pairwise_cosines: dict[str, float]
    data_parallel_world_size: int


@dataclass(frozen=True)
class GradientUpdateAudit:
    """Audit record for one explicit multi-objective optimizer update."""

    objective_norms: dict[str, float]
    pairwise_cosines: dict[str, float]
    weights_before_cap: dict[str, float]
    weights_after_cap: dict[str, float]
    cap: GradientCapAudit | None
    joint_norm_before_clip: float
    joint_norm_after_clip: float
    global_clip_scale: float
    data_parallel_world_size: int
    joint_gradient_active: bool


class EMABoundedGradientController:
    """Balance active objectives by inverse EMA gradient norm within bounds."""

    def __init__(
        self,
        objective_keys: Iterable[str],
        config: GradientBudgetConfig | None = None,
    ) -> None:
        keys = tuple(objective_keys)
        if not keys or len(keys) != len(set(keys)):
            raise ValueError("objective_keys must be non-empty and unique")
        self.objective_keys = keys
        self.config = config or GradientBudgetConfig()
        self.ema_norms: dict[str, float] = {}
        self.weights: dict[str, float] = {key: 1.0 for key in keys}

    def _validate_keys(self, values: Mapping[str, object]) -> None:
        if set(values) != set(self.objective_keys):
            missing = sorted(set(self.objective_keys) - set(values))
            extra = sorted(set(values) - set(self.objective_keys))
            raise ValueError(
                f"objective key mismatch: missing={missing}, extra={extra}"
            )

    def measure_gradient_norms(
        self,
        objectives: Mapping[str, torch.Tensor],
        parameters: Iterable[nn.Parameter],
    ) -> dict[str, float]:
        """Measure each objective independently on the current autograd graph."""

        self._validate_keys(objectives)
        trainable = tuple(
            parameter for parameter in parameters if parameter.requires_grad
        )
        if not trainable:
            raise ValueError("no trainable parameters were supplied")
        norms: dict[str, float] = {}
        for key in self.objective_keys:
            gradients = torch.autograd.grad(
                objectives[key],
                trainable,
                retain_graph=True,
                allow_unused=True,
            )
            squared_norm = sum(
                float(gradient.detach().float().square().sum().item())
                for gradient in gradients
                if gradient is not None
            )
            norm = math.sqrt(squared_norm)
            if not math.isfinite(norm):
                raise FloatingPointError(f"non-finite gradient norm for {key}")
            norms[key] = norm
        return norms

    def update_from_norms(self, norms: Mapping[str, float]) -> dict[str, float]:
        """Update EMA statistics and return mean-one bounded active weights."""

        self._validate_keys(norms)
        active: list[str] = []
        for key in self.objective_keys:
            norm = float(norms[key])
            if not math.isfinite(norm) or norm < 0.0:
                raise FloatingPointError(f"invalid gradient norm for {key}")
            if norm <= self.config.activation_epsilon:
                self.weights[key] = 1.0
                continue
            active.append(key)
            previous = self.ema_norms.get(key)
            self.ema_norms[key] = (
                norm
                if previous is None
                else self.config.ema_beta * previous
                + (1.0 - self.config.ema_beta) * norm
            )
        if not active:
            return dict(self.weights)

        target_norm = sum(self.ema_norms[key] for key in active) / len(active)
        preliminary = {
            key: min(
                self.config.maximum_weight,
                max(
                    self.config.minimum_weight,
                    target_norm
                    / max(self.ema_norms[key], self.config.activation_epsilon),
                ),
            )
            for key in active
        }
        mean_preliminary = sum(preliminary.values()) / len(preliminary)
        for key in active:
            self.weights[key] = min(
                self.config.maximum_weight,
                max(
                    self.config.minimum_weight,
                    preliminary[key] / mean_preliminary,
                ),
            )
        return dict(self.weights)

    def audit_and_update(
        self,
        objectives: Mapping[str, torch.Tensor],
        parameters: Iterable[nn.Parameter],
    ) -> dict[str, float]:
        norms = self.measure_gradient_norms(objectives, parameters)
        self.update_from_norms(norms)
        return norms

    def weighted_sum(
        self,
        objectives: Mapping[str, torch.Tensor],
        *,
        weights: Mapping[str, float] | None = None,
    ) -> torch.Tensor:
        self._validate_keys(objectives)
        selected = self.weights if weights is None else weights
        self._validate_keys(selected)
        return torch.stack(
            [objectives[key] * float(selected[key]) for key in self.objective_keys]
        ).sum()

    def state_dict(self) -> dict[str, object]:
        return {
            "objective_keys": self.objective_keys,
            "config": asdict(self.config),
            "ema_norms": dict(self.ema_norms),
            "weights": dict(self.weights),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        if set(state["objective_keys"]) != set(self.objective_keys):  # type: ignore[arg-type]
            raise ValueError("checkpoint objective keys do not match the controller")
        if GradientBudgetConfig(**dict(state["config"])) != self.config:  # type: ignore[arg-type]
            raise ValueError("checkpoint gradient-budget config does not match")
        ema_norms = {
            key: float(value) for key, value in dict(state["ema_norms"]).items()
        }  # type: ignore[arg-type]
        weights = {key: float(value) for key, value in dict(state["weights"]).items()}  # type: ignore[arg-type]
        self._validate_keys(weights)
        if any(key not in self.objective_keys for key in ema_norms):
            raise ValueError("checkpoint contains an unknown EMA objective")
        if any(not math.isfinite(value) or value < 0.0 for value in ema_norms.values()):
            raise FloatingPointError("checkpoint contains an invalid EMA norm")
        if any(not math.isfinite(value) or value <= 0.0 for value in weights.values()):
            raise FloatingPointError("checkpoint contains an invalid objective weight")
        self.ema_norms = ema_norms
        self.weights = weights


def _gradient_tuple_norm(gradients: tuple[torch.Tensor, ...]) -> float:
    squared_norm = sum(
        float(gradient.detach().float().square().sum().item()) for gradient in gradients
    )
    norm = math.sqrt(squared_norm)
    if not math.isfinite(norm):
        raise FloatingPointError("non-finite gradient norm")
    return norm


def _pairwise_gradient_cosines(
    gradients: Mapping[str, tuple[torch.Tensor, ...]],
    *,
    eps: float = 1e-12,
) -> dict[str, float]:
    keys = tuple(gradients)
    cosines: dict[str, float] = {}
    norms = {key: _gradient_tuple_norm(gradients[key]) for key in keys}
    for left_index, left_key in enumerate(keys):
        for right_key in keys[left_index + 1 :]:
            left = gradients[left_key]
            right = gradients[right_key]
            if len(left) != len(right):
                raise ValueError("objective gradient tuples have different lengths")
            dot = sum(
                float(
                    (left_gradient.detach().float() * right_gradient.detach().float())
                    .sum()
                    .item()
                )
                for left_gradient, right_gradient in zip(left, right)
            )
            denominator = norms[left_key] * norms[right_key]
            cosine = 0.0 if denominator <= eps else dot / denominator
            if not math.isfinite(cosine):
                raise FloatingPointError("non-finite pairwise gradient cosine")
            cosines[f"{left_key}|{right_key}"] = max(-1.0, min(1.0, cosine))
    return cosines


def collect_objective_gradients(
    objectives: Mapping[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
    *,
    average_across_data_parallel: bool = True,
    process_group: object | None = None,
) -> ObjectiveGradientSet:
    """Differentiate every objective and world-average each gradient tensor.

    The returned gradients are detached tensors. Unused parameters receive a
    zero tensor so every rank reduces the same ordered tensor set. If
    ``torch.distributed`` is not initialized, the effective world size is one.
    """

    if not objectives:
        raise ValueError("objectives must be non-empty")
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if not trainable:
        raise ValueError("no trainable parameters were supplied")
    if len(objectives) != len(set(objectives)):
        raise ValueError("objective keys must be unique")

    distributed = (
        average_across_data_parallel and dist.is_available() and dist.is_initialized()
    )
    world_size = int(dist.get_world_size(group=process_group)) if distributed else 1
    if world_size <= 0:
        raise RuntimeError("invalid data-parallel world size")

    collected: dict[str, tuple[torch.Tensor, ...]] = {}
    for key, objective in objectives.items():
        if not isinstance(objective, torch.Tensor) or objective.numel() != 1:
            raise ValueError(f"objective {key!r} must be a scalar tensor")
        if not objective.requires_grad:
            raise ValueError(f"objective {key!r} is detached from the trainable graph")
        raw_gradients = torch.autograd.grad(
            objective,
            trainable,
            retain_graph=True,
            allow_unused=True,
        )
        objective_gradients: list[torch.Tensor] = []
        for parameter, gradient in zip(trainable, raw_gradients):
            reduced = (
                torch.zeros_like(parameter)
                if gradient is None
                else gradient.detach().clone()
            )
            if distributed:
                dist.all_reduce(reduced, op=dist.ReduceOp.SUM, group=process_group)
                reduced.div_(world_size)
            if not bool(torch.isfinite(reduced).all()):
                raise FloatingPointError(f"non-finite gradient tensor for {key}")
            objective_gradients.append(reduced)
        collected[key] = tuple(objective_gradients)

    norms = {key: _gradient_tuple_norm(value) for key, value in collected.items()}
    return ObjectiveGradientSet(
        gradients=collected,
        norms=norms,
        pairwise_cosines=_pairwise_gradient_cosines(collected),
        data_parallel_world_size=world_size,
    )


def combine_objective_gradients(
    objective_gradients: Mapping[str, tuple[torch.Tensor, ...]],
    weights: Mapping[str, float],
    *,
    maximum_global_norm: float = 1.0,
    eps: float = 1e-12,
) -> tuple[tuple[torch.Tensor, ...], float, float, float]:
    """Weight, sum, and globally clip explicit objective gradients."""

    if set(objective_gradients) != set(weights) or not objective_gradients:
        raise ValueError("objective gradients and weights must have matching keys")
    if not math.isfinite(maximum_global_norm) or maximum_global_norm <= 0.0:
        raise ValueError("maximum_global_norm must be finite and positive")
    normalized_weights = {key: float(value) for key, value in weights.items()}
    if any(
        not math.isfinite(value) or value < 0.0 for value in normalized_weights.values()
    ):
        raise FloatingPointError("weights must be finite and non-negative")

    keys = tuple(objective_gradients)
    parameter_count = len(objective_gradients[keys[0]])
    if parameter_count == 0 or any(
        len(objective_gradients[key]) != parameter_count for key in keys
    ):
        raise ValueError(
            "objective gradient tuples must have one common non-zero length"
        )
    combined: list[torch.Tensor] = []
    for parameter_index in range(parameter_count):
        reference = objective_gradients[keys[0]][parameter_index]
        if any(
            objective_gradients[key][parameter_index].shape != reference.shape
            for key in keys
        ):
            raise ValueError("objective gradient tensor shapes do not match")
        joint = torch.zeros_like(reference)
        for key in keys:
            joint.add_(
                objective_gradients[key][parameter_index],
                alpha=normalized_weights[key],
            )
        combined.append(joint)

    norm_before = _gradient_tuple_norm(tuple(combined))
    clip_scale = min(1.0, maximum_global_norm / max(norm_before, eps))
    if clip_scale < 1.0:
        combined = [gradient * clip_scale for gradient in combined]
    norm_after = _gradient_tuple_norm(tuple(combined))
    return tuple(combined), norm_before, norm_after, clip_scale


def install_parameter_gradients(
    parameters: Iterable[nn.Parameter],
    gradients: tuple[torch.Tensor, ...],
) -> None:
    """Install an explicit gradient tuple for the next optimizer step."""

    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    if len(trainable) != len(gradients):
        raise ValueError("gradient tuple does not match trainable parameters")
    for parameter, gradient in zip(trainable, gradients):
        if gradient.shape != parameter.shape:
            raise ValueError("gradient shape does not match its parameter")
        parameter.grad = (
            gradient.detach()
            .to(
                device=parameter.device,
                dtype=parameter.dtype,
            )
            .clone()
        )


def apply_ce_anchor_cap(
    weights: Mapping[str, float],
    gradient_norms: Mapping[str, float],
    *,
    target_key: str,
    ce_key: str,
    maximum_ratio: float = 1.0,
    eps: float = 1e-12,
) -> tuple[dict[str, float], GradientCapAudit]:
    """Cap one auxiliary weighted gradient against the hard-CE contribution.

    Unlike the EMA bounds, this safety cap is allowed to reduce the target
    weight below the controller's usual minimum.  This makes the task loss a
    hard anchor rather than merely another balanced objective.
    """

    if set(weights) != set(gradient_norms):
        raise ValueError("weights and gradient_norms must have matching key sets")
    if target_key not in weights or ce_key not in weights:
        raise KeyError("target_key and ce_key must exist")
    if target_key == ce_key:
        raise ValueError("target_key and ce_key must differ")
    if not math.isfinite(maximum_ratio) or maximum_ratio <= 0.0:
        raise ValueError("maximum_ratio must be finite and positive")
    if not math.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be finite and positive")
    normalized_weights = {key: float(value) for key, value in weights.items()}
    normalized_norms = {key: float(value) for key, value in gradient_norms.items()}
    if any(
        not math.isfinite(value) or value <= 0.0
        for value in normalized_weights.values()
    ):
        raise FloatingPointError("weights must be finite and positive")
    if any(
        not math.isfinite(value) or value < 0.0 for value in normalized_norms.values()
    ):
        raise FloatingPointError("gradient norms must be finite and non-negative")

    reference = normalized_weights[ce_key] * normalized_norms[ce_key]
    uncapped = normalized_weights[target_key] * normalized_norms[target_key]
    allowed = maximum_ratio * reference
    if normalized_norms[target_key] <= eps:
        capped_weight = normalized_weights[target_key]
    else:
        capped_weight = min(
            normalized_weights[target_key],
            allowed / normalized_norms[target_key],
        )
    normalized_weights[target_key] = capped_weight
    capped = capped_weight * normalized_norms[target_key]
    ratio = capped / max(reference, eps)
    if reference <= eps and capped <= eps:
        ratio = 0.0
    if ratio > maximum_ratio * (1.0 + 1e-6):
        raise RuntimeError("CE-anchor gradient cap was not enforced")
    return normalized_weights, GradientCapAudit(
        target_key=target_key,
        reference_key=ce_key,
        maximum_ratio=maximum_ratio,
        reference_weighted_norm=reference,
        target_uncapped_weighted_norm=uncapped,
        target_capped_weighted_norm=capped,
        actual_ratio=ratio,
        cap_applied=capped_weight < float(weights[target_key]),
    )


def build_budgeted_gradient_update(
    controller: EMABoundedGradientController,
    objectives: Mapping[str, torch.Tensor],
    parameters: Iterable[nn.Parameter],
    *,
    ce_key: str | None = None,
    capped_auxiliary_key: str | None = None,
    maximum_auxiliary_to_ce_ratio: float = 1.0,
    maximum_global_norm: float = 1.0,
    average_across_data_parallel: bool = True,
    process_group: object | None = None,
) -> GradientUpdateAudit:
    """Construct and install the explicit RetentionKD compressor update.

    This is the auditable path used when a weighted scalar ``backward`` would
    be insufficient: each objective is differentiated independently, each
    gradient tensor is optionally DP-world-averaged, the controller and
    optional CE anchor operate on those averaged gradients, and the clipped
    joint gradient is written directly to ``parameter.grad``.
    """

    controller._validate_keys(objectives)
    trainable = tuple(parameter for parameter in parameters if parameter.requires_grad)
    gradient_set = collect_objective_gradients(
        objectives,
        trainable,
        average_across_data_parallel=average_across_data_parallel,
        process_group=process_group,
    )
    weights_before_cap = controller.update_from_norms(gradient_set.norms)
    weights_after_cap = dict(weights_before_cap)
    cap_audit: GradientCapAudit | None = None
    if (ce_key is None) != (capped_auxiliary_key is None):
        raise ValueError(
            "ce_key and capped_auxiliary_key must either both be set or both be None"
        )
    if ce_key is not None and capped_auxiliary_key is not None:
        weights_after_cap, cap_audit = apply_ce_anchor_cap(
            weights_before_cap,
            gradient_set.norms,
            target_key=capped_auxiliary_key,
            ce_key=ce_key,
            maximum_ratio=maximum_auxiliary_to_ce_ratio,
        )
    combined, norm_before, norm_after, clip_scale = combine_objective_gradients(
        gradient_set.gradients,
        weights_after_cap,
        maximum_global_norm=maximum_global_norm,
    )
    joint_gradient_active = norm_before > controller.config.activation_epsilon
    if joint_gradient_active:
        install_parameter_gradients(trainable, combined)
    else:
        # AdamW applies decoupled weight decay to parameters whose gradient is
        # an explicit zero tensor. Restore None so a zero-joint step is a true
        # no-op even if the caller still invokes optimizer.step().
        for parameter in trainable:
            parameter.grad = None
    return GradientUpdateAudit(
        objective_norms=dict(gradient_set.norms),
        pairwise_cosines=dict(gradient_set.pairwise_cosines),
        weights_before_cap=dict(weights_before_cap),
        weights_after_cap=dict(weights_after_cap),
        cap=cap_audit,
        joint_norm_before_clip=norm_before,
        joint_norm_after_clip=norm_after,
        global_clip_scale=clip_scale,
        data_parallel_world_size=gradient_set.data_parallel_world_size,
        joint_gradient_active=joint_gradient_active,
    )

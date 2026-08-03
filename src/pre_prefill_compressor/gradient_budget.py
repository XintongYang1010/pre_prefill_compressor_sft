"""Bounded gradient budgeting for multi-objective retention training."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass

import torch
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

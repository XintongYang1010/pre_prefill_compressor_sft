"""Generic, resumable compressor-training checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import random
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .gradient_budget import EMABoundedGradientController

CHECKPOINT_SCHEMA = "pre_prefill_compressor_training"
CHECKPOINT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class CheckpointState:
    step: int
    config: dict[str, Any]
    config_digest: str
    objective_history: dict[str, list[float]]
    extra: dict[str, Any]


def stable_config_digest(config: Mapping[str, Any]) -> str:
    """Hash a JSON-compatible recipe independently of dictionary order."""

    payload = json.dumps(
        dict(config),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    gradient_controller: EMABoundedGradientController | None = None,
    config: Mapping[str, Any] | None = None,
    objective_history: Mapping[str, list[float]] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Atomically save all state needed for an exact same-recipe resume."""

    if step < 0:
        raise ValueError("step must be non-negative")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    normalized_config = dict(config or {})
    payload: dict[str, Any] = {
        "schema": CHECKPOINT_SCHEMA,
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": None if optimizer is None else optimizer.state_dict(),
        "scheduler": None if scheduler is None else scheduler.state_dict(),
        "gradient_controller": (
            None if gradient_controller is None else gradient_controller.state_dict()
        ),
        "config": normalized_config,
        "config_digest": stable_config_digest(normalized_config),
        "objective_history": {
            key: [float(value) for value in values]
            for key, values in dict(objective_history or {}).items()
        },
        "extra": dict(extra or {}),
        "rng": {
            "python": random.getstate(),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        },
    }
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=str(destination.parent),
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, destination)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    gradient_controller: EMABoundedGradientController | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool = True,
    restore_rng: bool = True,
) -> CheckpointState:
    """Load a checkpoint created by :func:`save_training_checkpoint`.

    PyTorch checkpoints use pickle internally and therefore must only be loaded
    from a trusted source.
    """

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    payload = torch.load(source, map_location=map_location, weights_only=False)
    if payload.get("schema") != CHECKPOINT_SCHEMA:
        raise ValueError("unrecognized checkpoint schema")
    if payload.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("unsupported checkpoint schema version")
    if payload.get("config_digest") != stable_config_digest(payload["config"]):
        raise ValueError("checkpoint config digest does not match its config")
    model.load_state_dict(payload["model"], strict=strict)
    if optimizer is not None:
        if payload["optimizer"] is None:
            raise ValueError("checkpoint has no optimizer state")
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None:
        if payload["scheduler"] is None:
            raise ValueError("checkpoint has no scheduler state")
        scheduler.load_state_dict(payload["scheduler"])
    if gradient_controller is not None:
        if payload["gradient_controller"] is None:
            raise ValueError("checkpoint has no gradient-controller state")
        gradient_controller.load_state_dict(payload["gradient_controller"])
    if restore_rng:
        random.setstate(payload["rng"]["python"])
        torch.set_rng_state(payload["rng"]["torch_cpu"].cpu())
        cuda_states = payload["rng"].get("torch_cuda")
        if cuda_states is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_states)
    return CheckpointState(
        step=int(payload["step"]),
        config=dict(payload["config"]),
        config_digest=str(payload["config_digest"]),
        objective_history={
            key: [float(value) for value in values]
            for key, values in dict(payload["objective_history"]).items()
        },
        extra=dict(payload["extra"]),
    )

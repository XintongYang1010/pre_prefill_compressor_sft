"""Spatial visual-token compression after a model's native visual merger.

This module intentionally depends only on tensor contracts.  It contains no
model-vendor wrapper, dataset schema, serving integration, or private path.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class VisionCompressorConfig:
    """Configuration for a learned spatial ``2 x 2 -> 1`` compressor.

    ``grid_thw`` values supplied to :func:`build_image_token_plan` describe the
    encoder grid *before* ``native_merge_size`` is applied.  The input features
    are expected to be the output of that native merger.
    """

    input_dim: int
    bottleneck_dim: int = 128
    native_merge_size: int = 2
    spatial_stride: int = 2

    def __post_init__(self) -> None:
        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if self.bottleneck_dim <= 0:
            raise ValueError("bottleneck_dim must be positive")
        if self.native_merge_size <= 0:
            raise ValueError("native_merge_size must be positive")
        if self.spatial_stride != 2:
            raise ValueError("the reference compressor implements exactly 2 x 2 -> 1")


@dataclass(frozen=True)
class ImageTokenPlan:
    """A deterministic, per-image mapping from source to compressed tokens.

    ``source_to_compressed`` assigns each real source token exactly once.
    ``group_indices`` has four entries per compressed token; for an odd final
    row or column, its final valid edge is repeated to keep the learned input
    width fixed.  ``group_valid_mask`` records that geometry for auditing.  The
    reference recipe deliberately includes replicated entries in the four-way
    teacher mean and reconstruction objective.
    """

    encoder_grid_thw: torch.Tensor
    llm_grid_thw: torch.Tensor
    native_shapes_thw: tuple[tuple[int, int, int], ...]
    compressed_shapes_thw: tuple[tuple[int, int, int], ...]
    source_tokens_per_image: tuple[int, ...]
    compressed_tokens_per_image: tuple[int, ...]
    source_to_compressed: torch.Tensor
    group_indices: torch.Tensor
    group_valid_mask: torch.Tensor

    @property
    def total_source_tokens(self) -> int:
        return sum(self.source_tokens_per_image)

    @property
    def total_compressed_tokens(self) -> int:
        return sum(self.compressed_tokens_per_image)

    @property
    def keep_ratio(self) -> float:
        return self.total_compressed_tokens / self.total_source_tokens


@dataclass
class FeatureCompressionOutput:
    """Compressor outputs and aligned, detached teacher-facing structure."""

    compressed: torch.Tensor
    pooled_teacher: torch.Tensor
    grouped_source: torch.Tensor
    reconstructed_source: torch.Tensor
    group_valid_mask: torch.Tensor
    plan: ImageTokenPlan


def _validated_grid(grid_thw: torch.Tensor) -> torch.Tensor:
    if not isinstance(grid_thw, torch.Tensor):
        raise TypeError("grid_thw must be a torch.Tensor")
    if grid_thw.ndim != 2 or grid_thw.shape[1] != 3 or grid_thw.shape[0] == 0:
        raise ValueError("grid_thw must have non-empty shape [images, 3]")
    if grid_thw.dtype.is_floating_point or grid_thw.dtype.is_complex:
        raise TypeError("grid_thw must have an integer dtype")
    grid = grid_thw.detach().to(device="cpu", dtype=torch.long)
    if bool((grid <= 0).any()):
        raise ValueError("all grid_thw entries must be positive")
    return grid


def build_image_token_plan(
    grid_thw: torch.Tensor,
    *,
    native_merge_size: int = 2,
    spatial_stride: int = 2,
) -> ImageTokenPlan:
    """Build a mapping that never mixes images or temporal frames."""

    if native_merge_size <= 0:
        raise ValueError("native_merge_size must be positive")
    if spatial_stride != 2:
        raise ValueError("the reference plan implements exactly spatial_stride=2")
    grid = _validated_grid(grid_thw)
    if bool((grid[:, 1:] % native_merge_size != 0).any()):
        raise ValueError("grid height and width must be divisible by native_merge_size")

    native_shapes: list[tuple[int, int, int]] = []
    compressed_shapes: list[tuple[int, int, int]] = []
    source_counts: list[int] = []
    compressed_counts: list[int] = []
    source_to_compressed: list[int] = []
    group_indices: list[list[int]] = []
    group_valid: list[list[bool]] = []
    llm_grid = grid.clone()
    source_offset = 0
    compressed_offset = 0

    for image_index, (t_value, h_value, w_value) in enumerate(grid.tolist()):
        native_t = int(t_value)
        native_h = int(h_value) // native_merge_size
        native_w = int(w_value) // native_merge_size
        compressed_h = (native_h + spatial_stride - 1) // spatial_stride
        compressed_w = (native_w + spatial_stride - 1) // spatial_stride
        native_shape = (native_t, native_h, native_w)
        compressed_shape = (native_t, compressed_h, compressed_w)
        native_shapes.append(native_shape)
        compressed_shapes.append(compressed_shape)
        source_counts.append(native_t * native_h * native_w)
        compressed_counts.append(native_t * compressed_h * compressed_w)
        llm_grid[image_index, 1] = compressed_h * native_merge_size
        llm_grid[image_index, 2] = compressed_w * native_merge_size

        for time_index in range(native_t):
            for row in range(native_h):
                for column in range(native_w):
                    local_group = (
                        time_index * compressed_h * compressed_w
                        + (row // spatial_stride) * compressed_w
                        + column // spatial_stride
                    )
                    source_to_compressed.append(compressed_offset + local_group)

        for time_index in range(native_t):
            for compressed_row in range(compressed_h):
                for compressed_column in range(compressed_w):
                    indices: list[int] = []
                    valid: list[bool] = []
                    for row_delta in range(spatial_stride):
                        for column_delta in range(spatial_stride):
                            row = compressed_row * spatial_stride + row_delta
                            column = compressed_column * spatial_stride + column_delta
                            in_bounds = row < native_h and column < native_w
                            row = min(row, native_h - 1)
                            column = min(column, native_w - 1)
                            local_index = (
                                time_index * native_h * native_w
                                + row * native_w
                                + column
                            )
                            indices.append(source_offset + local_index)
                            valid.append(in_bounds)
                    group_indices.append(indices)
                    group_valid.append(valid)

        source_offset += source_counts[-1]
        compressed_offset += compressed_counts[-1]

    plan = ImageTokenPlan(
        encoder_grid_thw=grid.clone(),
        llm_grid_thw=llm_grid,
        native_shapes_thw=tuple(native_shapes),
        compressed_shapes_thw=tuple(compressed_shapes),
        source_tokens_per_image=tuple(source_counts),
        compressed_tokens_per_image=tuple(compressed_counts),
        source_to_compressed=torch.tensor(source_to_compressed, dtype=torch.long),
        group_indices=torch.tensor(group_indices, dtype=torch.long),
        group_valid_mask=torch.tensor(group_valid, dtype=torch.bool),
    )
    if plan.source_to_compressed.numel() != plan.total_source_tokens:
        raise RuntimeError("source mapping is incomplete")
    if tuple(plan.group_indices.shape) != (plan.total_compressed_tokens, 4):
        raise RuntimeError("group mapping has an unexpected shape")
    return plan


class GridTokenCompressor(nn.Module):
    """Learn one token from every local post-merger 2 x 2 feature group."""

    def __init__(self, config: VisionCompressorConfig) -> None:
        super().__init__()
        self.config = config
        group_dim = config.input_dim * config.spatial_stride**2
        self.group_encoder = nn.Linear(group_dim, config.bottleneck_dim)
        self.token_decoder = nn.Linear(config.bottleneck_dim, config.input_dim)
        self.reconstruction_encoder = nn.Linear(config.input_dim, config.bottleneck_dim)
        self.reconstruction_decoder = nn.Linear(config.bottleneck_dim, group_dim)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.group_encoder,
            self.token_decoder,
            self.reconstruction_encoder,
            self.reconstruction_decoder,
        ):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    @property
    def trainable_parameter_count(self) -> int:
        return sum(
            parameter.numel()
            for parameter in self.parameters()
            if parameter.requires_grad
        )

    def compress_with_plan(
        self,
        features: torch.Tensor,
        plan: ImageTokenPlan,
    ) -> FeatureCompressionOutput:
        if features.ndim != 2 or features.shape[1] != self.config.input_dim:
            raise ValueError(
                f"features must have shape [tokens, {self.config.input_dim}]"
            )
        if features.shape[0] != plan.total_source_tokens:
            raise ValueError("feature length does not match the token plan")
        indices = plan.group_indices.to(device=features.device)
        valid_mask = plan.group_valid_mask.to(device=features.device)
        grouped = features.index_select(0, indices.reshape(-1)).reshape(
            plan.total_compressed_tokens,
            4,
            self.config.input_dim,
        )
        flat_groups = grouped.reshape(plan.total_compressed_tokens, -1)
        compressed = self.token_decoder(F.gelu(self.group_encoder(flat_groups)))
        reconstructed = self.reconstruction_decoder(
            F.gelu(self.reconstruction_encoder(compressed))
        ).reshape_as(grouped)
        pooled_teacher = grouped.mean(dim=1)
        return FeatureCompressionOutput(
            compressed=compressed,
            pooled_teacher=pooled_teacher,
            grouped_source=grouped,
            reconstructed_source=reconstructed,
            group_valid_mask=valid_mask,
            plan=plan,
        )

    def forward(
        self,
        features: torch.Tensor,
        grid_thw: torch.Tensor,
    ) -> FeatureCompressionOutput:
        plan = build_image_token_plan(
            grid_thw,
            native_merge_size=self.config.native_merge_size,
            spatial_stride=self.config.spatial_stride,
        )
        return self.compress_with_plan(features, plan)


def compress_feature_branches(
    compressor: GridTokenCompressor,
    branches: Mapping[str, torch.Tensor],
    grid_thw: torch.Tensor,
) -> dict[str, FeatureCompressionOutput]:
    """Apply exactly one spatial plan to main and optional side branches."""

    if not branches:
        raise ValueError("branches must be non-empty")
    plan = build_image_token_plan(
        grid_thw,
        native_merge_size=compressor.config.native_merge_size,
        spatial_stride=compressor.config.spatial_stride,
    )
    return {
        name: compressor.compress_with_plan(features, plan)
        for name, features in branches.items()
    }


def replicated_slot_reconstruction_loss(
    output: FeatureCompressionOutput,
) -> torch.Tensor:
    """Reconstruct four slots, including replicated odd-grid edge entries.

    This is the exact historical recipe. It is intentionally *not* a valid-slot
    masked loss: a replicated edge token receives repeated weight. A true-mask
    alternative remains an explicit architecture ablation.
    """

    return F.mse_loss(output.reconstructed_source, output.grouped_source.detach())


def masked_reconstruction_loss(output: FeatureCompressionOutput) -> torch.Tensor:
    """Backward-compatible alias for :func:`replicated_slot_reconstruction_loss`.

    The old name is retained for package compatibility; it must not be read as
    evidence that replicated odd-edge slots are excluded.
    """

    return replicated_slot_reconstruction_loss(output)

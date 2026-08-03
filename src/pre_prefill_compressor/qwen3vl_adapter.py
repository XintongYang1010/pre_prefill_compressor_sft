"""Minimal public tensor adapter for Qwen3-VL-style image placeholders.

The adapter deliberately stops at documented tensor contracts. It does not
load a model, tokenizer, prompt template, or private example.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .compressor import ImageTokenPlan


@dataclass(frozen=True)
class CompactedPrompt:
    """Prompt tensors after replacing each image-token run by its new length."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    llm_grid_thw: torch.Tensor
    active_tokens_before: tuple[int, ...]
    active_tokens_after: tuple[int, ...]
    images_per_sample: tuple[int, ...]


def _image_token_runs(
    values: torch.Tensor, image_token_id: int
) -> list[tuple[int, int]]:
    positions = (
        torch.nonzero(values == image_token_id, as_tuple=False).flatten().tolist()
    )
    if not positions:
        return []
    runs: list[tuple[int, int]] = []
    start = previous = positions[0]
    for position in positions[1:]:
        if position != previous + 1:
            runs.append((start, previous + 1))
            start = position
        previous = position
    runs.append((start, previous + 1))
    return runs


def compact_image_placeholders(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    plan: ImageTokenPlan,
    *,
    image_token_id: int,
    pad_token_id: int,
    padding_side: str = "left",
) -> CompactedPrompt:
    """Compact image-placeholder runs with the exact per-image token plan.

    Contiguous active occurrences of ``image_token_id`` are treated as one
    image region. Runs are consumed in batch order and must match
    ``plan.source_tokens_per_image`` exactly. Non-image tokens retain order.
    """

    if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
        raise ValueError(
            "input_ids and attention_mask must have identical [batch, seq] shape"
        )
    if input_ids.dtype.is_floating_point or input_ids.dtype.is_complex:
        raise TypeError("input_ids must use an integer dtype")
    if image_token_id == pad_token_id:
        raise ValueError("image_token_id and pad_token_id must differ")
    if padding_side not in {"left", "right"}:
        raise ValueError("padding_side must be 'left' or 'right'")

    compacted_rows: list[torch.Tensor] = []
    before_counts: list[int] = []
    after_counts: list[int] = []
    images_per_sample: list[int] = []
    image_index = 0

    for ids, mask in zip(input_ids, attention_mask):
        active = ids[mask.to(dtype=torch.bool)]
        before_counts.append(int(active.numel()))
        runs = _image_token_runs(active, image_token_id)
        images_per_sample.append(len(runs))
        pieces: list[torch.Tensor] = []
        cursor = 0
        for start, end in runs:
            if image_index >= len(plan.source_tokens_per_image):
                raise ValueError(
                    "prompt contains more image regions than the token plan"
                )
            expected_source = plan.source_tokens_per_image[image_index]
            if end - start != expected_source:
                raise ValueError(
                    "image placeholder length does not match the token plan: "
                    f"image={image_index}, prompt={end - start}, expected={expected_source}"
                )
            pieces.append(active[cursor:start])
            pieces.append(
                torch.full(
                    (plan.compressed_tokens_per_image[image_index],),
                    image_token_id,
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                )
            )
            cursor = end
            image_index += 1
        pieces.append(active[cursor:])
        compacted = torch.cat(pieces) if pieces else active.clone()
        compacted_rows.append(compacted)
        after_counts.append(int(compacted.numel()))

    if image_index != len(plan.source_tokens_per_image):
        raise ValueError(
            "token plan contains image regions that were not present in the prompt"
        )

    maximum_length = max(after_counts)
    compacted_ids = torch.full(
        (input_ids.shape[0], maximum_length),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    compacted_mask = torch.zeros(
        (input_ids.shape[0], maximum_length),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    for row_index, values in enumerate(compacted_rows):
        length = int(values.numel())
        start = maximum_length - length if padding_side == "left" else 0
        compacted_ids[row_index, start : start + length] = values
        compacted_mask[row_index, start : start + length] = 1

    expected_placeholders = plan.total_compressed_tokens
    actual_placeholders = int(
        ((compacted_ids == image_token_id) & compacted_mask.to(dtype=torch.bool))
        .sum()
        .item()
    )
    if actual_placeholders != expected_placeholders:
        raise RuntimeError("compacted prompt and visual-feature counts diverged")

    return CompactedPrompt(
        input_ids=compacted_ids,
        attention_mask=compacted_mask,
        llm_grid_thw=plan.llm_grid_thw.to(device=input_ids.device),
        active_tokens_before=tuple(before_counts),
        active_tokens_after=tuple(after_counts),
        images_per_sample=tuple(images_per_sample),
    )

import pytest
import torch

from pre_prefill_compressor import build_image_token_plan, compact_image_placeholders


def test_compacts_each_image_run_and_preserves_non_image_order() -> None:
    image_token = 999
    pad_token = 0
    plan = build_image_token_plan(torch.tensor([[1, 4, 4], [1, 8, 4]]))
    input_ids = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0, 101, 999, 999, 999, 999, 102],
            [0, 0, 103, 999, 999, 999, 999, 999, 999, 999, 999, 104],
        ]
    )
    attention_mask = input_ids.ne(pad_token).long()

    compacted = compact_image_placeholders(
        input_ids,
        attention_mask,
        plan,
        image_token_id=image_token,
        pad_token_id=pad_token,
        padding_side="left",
    )

    assert compacted.active_tokens_before == (6, 10)
    assert compacted.active_tokens_after == (3, 4)
    assert compacted.images_per_sample == (1, 1)
    assert compacted.input_ids.tolist() == [
        [0, 101, 999, 102],
        [103, 999, 999, 104],
    ]
    assert compacted.attention_mask.tolist() == [[0, 1, 1, 1], [1, 1, 1, 1]]
    assert compacted.llm_grid_thw.tolist() == [[1, 2, 2], [1, 4, 2]]


def test_rejects_placeholder_length_drift() -> None:
    plan = build_image_token_plan(torch.tensor([[1, 4, 4]]))
    input_ids = torch.tensor([[101, 999, 999, 102]])
    with pytest.raises(ValueError, match="placeholder length"):
        compact_image_placeholders(
            input_ids,
            torch.ones_like(input_ids),
            plan,
            image_token_id=999,
            pad_token_id=0,
        )

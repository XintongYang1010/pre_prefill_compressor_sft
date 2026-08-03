# Qwen3-VL Integration Contract and Sanitized Evidence

## Scope

This document closes the public *description* gap between the tensor-only
reference package and an internally exercised Qwen3-VL integration. It contains
sanitized contracts, pseudocode, anonymous tensor shapes, and aggregate test
evidence. It is not a copy of a production patch. Internal source code, paths,
model assets, prompts, samples, infrastructure identifiers, and receipts remain
restricted.

The current integration is **image-only**. Video placeholders fail closed, and
no claim of video compatibility is made.

## Exact trainable boundary

```text
pixel values
  -> frozen Qwen3-VL vision encoder
  -> frozen native main merger + three native DeepStack mergers
  -> one shared trainable GridTokenCompressor applied branch by branch
  -> frozen Qwen3-VL language model
```

Only the raw compressor parameters update. Gradients traverse the frozen
language model on the student path, but the vision encoder, native mergers, LM,
and behavior-teacher compressor remain frozen.

The behavior teacher is the frozen pre-retention b512 compressor followed by
the same frozen Qwen3-VL language model. The feature teacher is the
uncompressed native-merger main/DeepStack representation. These are distinct
teacher sources rather than two separately trained foundation models.

## Tensor contract

Let an encoder grid be `(t, h, w)` and Qwen's native spatial merge size be
`m=2`. The native-merger feature grid is
`(t, h/m, w/m)`. A further compressor stride of `(2, 2)` produces
`(t, ceil((h/m)/2), ceil((w/m)/2))`. The grid passed back to Qwen is therefore
`(t, new_h*m, new_w*m)`.

| Tensor | Before | After | Required invariant |
| --- | --- | --- | --- |
| Main visual features | `[N, 5120]` | `[N', 5120]` | Image/frame order is unchanged |
| Each of three DeepStack features | `[N, 5120]` | `[N', 5120]` | Uses the exact main-stream plan |
| Image placeholder run | `N` tokens | `N'` tokens | Equals compressed main length |
| M-RoPE position IDs | `[3, B, S]` | `[3, B, S']` | Recomputed from compacted prompt/grid |
| RoPE delta | `[B, 1]` | `[B, 1]` | Recomputed, not inherited from the old grid |

The vision encoder taps are blocks `8/16/24`. These labels identify vision-side
feature extraction points. They must not be misread as text-decoder injection
layers: the three resulting DeepStack tensors are consumed, in order, after
text-decoder layers `0/1/2` in the pinned Qwen3-VL implementation.

The pinned Qwen3-VL interface does not use an `mm_token_type_ids` argument, so
its absence is not evidence of a missing integration field.

## Sanitized forward pseudocode

```python
with frozen_vision():
    main, ds8, ds16, ds24 = vision(pixel_values, image_grid_thw)

plan = build_image_token_plan(
    image_grid_thw,
    native_merge_size=2,
    spatial_stride=2,
)

compressed = compress_feature_branches(
    compressor,
    {"main": main, "ds8": ds8, "ds16": ds16, "ds24": ds24},
    image_grid_thw,
)

prompt = compact_image_placeholders(
    input_ids,
    attention_mask,
    plan,
    image_token_id=IMAGE_TOKEN,
    pad_token_id=PAD_TOKEN,
)

assert len(compressed["main"].compressed) == count_image_placeholders(prompt)
assert all(
    len(compressed[name].compressed) == len(compressed["main"].compressed)
    for name in ("ds8", "ds16", "ds24")
)

image_mask = get_placeholder_mask(prompt.input_ids, prompt.attention_mask)
inputs_embeds[image_mask] = compressed["main"].compressed

# Recompute three-axis M-RoPE positions from the compacted prompt and grid.
position_ids, rope_deltas = get_rope_index(
    prompt.input_ids,
    image_grid_thw=prompt.llm_grid_thw,
    attention_mask=prompt.attention_mask,
)

student_outputs = frozen_language_model(
    input_ids=None,
    inputs_embeds=inputs_embeds,
    attention_mask=prompt.attention_mask,
    position_ids=position_ids,
    visual_pos_masks=image_mask,
    deepstack_visual_embeds=[
        compressed["ds8"].compressed,
        compressed["ds16"].compressed,
        compressed["ds24"].compressed,
    ],
)
```

The final Qwen calls above are contract-level pseudocode. The public package
implements the compressor, shared image-local plan, and prompt/grid compaction;
it intentionally does not vendor an internal model wrapper or serving patch.

For cached generation, decode position IDs must incorporate the recomputed
RoPE delta, while visual and DeepStack inputs are supplied only during prefill.
Generation/cache behavior is not yet reproduced by a public end-to-end test.

## Anonymous integration evidence

One multi-image integration case exercised main plus all three DeepStack
branches:

- each branch changed from `[1321, 5120]` to `[342, 5120]`;
- per-image lengths changed from `[529, 264, 264, 264]` to
  `[144, 66, 66, 66]`;
- total prompt tokens changed from `4020` to `3041`;
- the identity-mode maximum absolute tensor error was `0.0`.

A separate redacted service-integration check processed `8/8` requests and 29
images, reduced aggregate visual tokens from `40,955` to `10,385`, and passed
27 compacted-grid consistency checks plus `8/8` position checks. These results
support mechanism compatibility. They are not business-quality, latency,
throughput, or production-promotion evidence.

## Odd-grid edge semantics

When a final row or column is odd, the last valid feature is replicated to keep
the learned four-slot input width fixed. The historical teacher mean and local
reconstruction loss include those replicated slots. Therefore an odd-edge token
can receive repeated weight; this is not an unbiased valid-slot-masked loss.

The public API now calls this
`replicated_slot_reconstruction_loss`. The former
`masked_reconstruction_loss` name remains only as a compatibility alias. A
replicated-slot versus true-valid-mask comparison is an open ablation.

## What remains unestablished

- a distributable end-to-end Qwen3-VL model wrapper using public assets;
- public generation/cache and multi-image end-to-end tests;
- video support;
- a replicated-edge versus valid-slot-mask ablation;
- fixed-contract serving performance for the selected Full-u64 checkpoint.

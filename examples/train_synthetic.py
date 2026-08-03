"""CPU-only synthetic example of compressor-only RetentionKD training.

The tensors are generated locally and do not encode any application schema or
real sample.  The frozen heads stand in for a frozen vision/LLM stack.
"""

from __future__ import annotations

from collections import OrderedDict

import torch
from torch import nn

from pre_prefill_compressor import (
    EMABoundedGradientController,
    GridTokenCompressor,
    VisionCompressorConfig,
    apply_ce_anchor_cap,
    effective_number_weights,
    generalized_jsd_loss,
    masked_reconstruction_loss,
    normalized_feature_distillation_loss,
    vision_language_affinity_distillation_loss,
    vision_semantic_distillation_loss,
    weighted_classification_loss,
)


def run_training(steps: int = 3) -> dict[str, float]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    torch.manual_seed(7)
    feature_dim = 12
    compressor = GridTokenCompressor(
        VisionCompressorConfig(input_dim=feature_dim, bottleneck_dim=24)
    )
    frozen_task_head = nn.Linear(feature_dim, 2)
    frozen_semantic_head = nn.Linear(feature_dim, 5)
    for parameter in (
        *frozen_task_head.parameters(),
        *frozen_semantic_head.parameters(),
    ):
        parameter.requires_grad_(False)

    # Two synthetic images.  Each [1, 4, 4] encoder grid becomes a 2 x 2
    # post-merger feature grid, then one compressed token.
    grid_thw = torch.tensor([[1, 4, 4], [1, 4, 4]], dtype=torch.long)
    source_features = torch.randn(8, feature_dim)
    source_features[4:] += 0.75
    labels = torch.tensor([0, 1], dtype=torch.long)
    sample_weights = effective_number_weights(labels, beta=0.9).sample_weights
    language_anchors = torch.randn(3, feature_dim)

    objective_keys = (
        "task_ce",
        "response_jsd",
        "vision_semantic",
        "vision_language_affinity",
        "feature_protection",
    )
    controller = EMABoundedGradientController(objective_keys)
    optimizer = torch.optim.AdamW(compressor.parameters(), lr=5e-4)
    last_values: dict[str, float] = {}

    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        output = compressor(source_features, grid_thw)
        student_task_logits = frozen_task_head(output.compressed)
        teacher_task_logits = frozen_task_head(output.pooled_teacher.detach())
        student_semantic_logits = frozen_semantic_head(output.compressed)
        teacher_semantic_logits = frozen_semantic_head(output.pooled_teacher.detach())
        feature = normalized_feature_distillation_loss(
            output.compressed,
            output.pooled_teacher,
        )
        objectives = OrderedDict(
            (
                (
                    "task_ce",
                    weighted_classification_loss(
                        student_task_logits,
                        labels,
                        sample_weights,
                    ),
                ),
                (
                    "response_jsd",
                    generalized_jsd_loss(student_task_logits, teacher_task_logits),
                ),
                (
                    "vision_semantic",
                    vision_semantic_distillation_loss(
                        student_semantic_logits,
                        teacher_semantic_logits,
                    ),
                ),
                (
                    "vision_language_affinity",
                    vision_language_affinity_distillation_loss(
                        output.compressed,
                        output.pooled_teacher,
                        language_anchors,
                        language_anchors,
                    ),
                ),
                (
                    "feature_protection",
                    feature.loss + 0.1 * masked_reconstruction_loss(output),
                ),
            )
        )
        gradient_norms = controller.audit_and_update(
            objectives, compressor.parameters()
        )
        capped_weights, _ = apply_ce_anchor_cap(
            controller.weights,
            gradient_norms,
            target_key="vision_semantic",
            ce_key="task_ce",
            maximum_ratio=1.0,
        )
        total = controller.weighted_sum(objectives, weights=capped_weights)
        total.backward()
        torch.nn.utils.clip_grad_norm_(compressor.parameters(), max_norm=1.0)
        optimizer.step()
        last_values = {
            key: float(value.detach().item()) for key, value in objectives.items()
        }
        last_values["total"] = float(total.detach().item())

    return last_values


if __name__ == "__main__":
    for name, value in run_training().items():
        print(f"{name}: {value:.6f}")

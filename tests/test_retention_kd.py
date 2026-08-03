import torch
import torch.nn.functional as F

from pre_prefill_compressor import (
    effective_number_weights,
    generalized_jsd_loss,
    group_mean,
    normalized_feature_distillation_loss,
    vision_language_affinity_distillation_loss,
    vision_semantic_distillation_loss,
    weighted_classification_loss,
)


def test_effective_number_weights_are_capped_and_mean_one() -> None:
    labels = torch.tensor([0] * 9 + [1])
    result = effective_number_weights(labels, beta=0.99, maximum_weight=4.0)
    assert torch.allclose(result.sample_weights.mean(), torch.tensor(1.0), atol=1e-6)
    assert result.class_weights[1] > result.class_weights[0]
    assert float(result.class_weights.max()) <= 4.0
    assert result.class_counts.tolist() == [9, 1]


def test_jsd_mask_and_teacher_detach() -> None:
    student = torch.tensor([[[2.0, -1.0], [0.5, 0.5]]], requires_grad=True)
    teacher = student.detach().clone().requires_grad_(True)
    labels = torch.tensor([[1, -100]])
    assert torch.allclose(
        generalized_jsd_loss(student, teacher, labels=labels), torch.tensor(0.0)
    )
    changed = teacher.detach().clone()
    changed[:, 1] = torch.tensor([100.0, -100.0])
    loss = generalized_jsd_loss(student, changed, labels=labels)
    loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_group_mean_vsd_and_vlad_alignment() -> None:
    teacher_vision = torch.tensor([[1.0, 0.0], [3.0, 0.0], [0.0, 2.0], [0.0, 4.0]])
    mapping = torch.tensor([0, 0, 1, 1])
    pooled = group_mean(teacher_vision, mapping, compressed_tokens=2)
    assert torch.equal(pooled, torch.tensor([[2.0, 0.0], [0.0, 3.0]]))
    semantic_teacher = torch.tensor([[2.0, -1.0], [0.5, 1.5], [1.0, 0.0], [0.0, 2.0]])
    semantic_student = group_mean(semantic_teacher, mapping, compressed_tokens=2)
    assert (
        vision_semantic_distillation_loss(
            semantic_student, semantic_teacher, source_to_compressed=mapping
        ).abs()
        < 1e-6
    )
    language = torch.tensor([[1.0, 1.0], [-1.0, 1.0]])
    assert (
        vision_language_affinity_distillation_loss(
            pooled, teacher_vision, language, language, source_to_compressed=mapping
        ).abs()
        < 1e-6
    )


def test_feature_loss_is_directional_and_detaches_teacher() -> None:
    student = torch.tensor([[1.0, 2.0]], requires_grad=True)
    teacher = torch.tensor([[2.0, 4.0]], requires_grad=True)
    result = normalized_feature_distillation_loss(student, teacher)
    assert result.loss < 1e-7
    result.loss.backward()
    assert student.grad is not None
    assert teacher.grad is None


def test_weighted_token_ce_keeps_only_valid_targets_in_denominator() -> None:
    logits = torch.tensor(
        [
            [[3.0, -1.0], [-1.0, 3.0], [0.0, 0.0]],
            [[-1.0, 3.0], [3.0, -1.0], [0.0, 0.0]],
        ]
    )
    labels = torch.tensor([[0, 1, -100], [1, -100, -100]])
    sample_weights = torch.tensor([1.0, 3.0])
    per_token = F.cross_entropy(
        logits.reshape(-1, 2), labels.reshape(-1), reduction="none", ignore_index=-100
    ).reshape_as(labels)
    expected = (per_token[0, 0] + per_token[0, 1] + 3.0 * per_token[1, 0]) / 5.0
    actual = weighted_classification_loss(logits, labels, sample_weights)
    assert torch.allclose(actual, expected)

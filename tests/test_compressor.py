import torch
import torch.nn.functional as F
from torch import nn

from pre_prefill_compressor import (
    GridTokenCompressor,
    VisionCompressorConfig,
    build_image_token_plan,
    compress_feature_branches,
    masked_reconstruction_loss,
    replicated_slot_reconstruction_loss,
)


def test_multi_image_plan_never_crosses_boundaries() -> None:
    grid = torch.tensor([[1, 4, 4], [1, 6, 10]], dtype=torch.long)
    plan = build_image_token_plan(grid)
    assert plan.source_tokens_per_image == (4, 15)
    assert plan.compressed_tokens_per_image == (1, 6)
    assert plan.total_source_tokens == 19
    assert plan.total_compressed_tokens == 7
    assert plan.llm_grid_thw.tolist() == [[1, 2, 2], [1, 4, 6]]
    assert plan.source_to_compressed[:4].tolist() == [0, 0, 0, 0]
    assert int(plan.source_to_compressed[4:].min()) == 1
    assert int(plan.group_indices[0].max()) < 4
    assert bool((plan.group_indices[1:] >= 4).all())
    assert torch.bincount(plan.source_to_compressed, minlength=7).gt(0).all()


def test_odd_grid_uses_replicated_edge_for_four_way_teacher_mean() -> None:
    grid = torch.tensor([[1, 6, 10]], dtype=torch.long)  # post-merger 3 x 5
    features = torch.arange(15, dtype=torch.float32).unsqueeze(-1)
    compressor = GridTokenCompressor(
        VisionCompressorConfig(input_dim=1, bottleneck_dim=4)
    )
    output = compressor(features, grid)
    gathered = features[output.plan.group_indices]
    assert torch.equal(output.grouped_source, gathered)
    assert torch.allclose(output.pooled_teacher, gathered.mean(dim=1))
    assert not bool(output.plan.group_valid_mask[-1].all())
    assert output.plan.group_valid_mask[-1].sum().item() == 1
    assert gathered[-1, :, 0].tolist() == [14.0, 14.0, 14.0, 14.0]
    assert gathered[2, :, 0].tolist() == [4.0, 4.0, 9.0, 9.0]
    manual = F.mse_loss(output.reconstructed_source, output.grouped_source.detach())
    assert torch.allclose(replicated_slot_reconstruction_loss(output), manual)
    assert torch.allclose(masked_reconstruction_loss(output), manual)


def test_main_and_side_branches_share_one_plan() -> None:
    grid = torch.tensor([[2, 4, 4]], dtype=torch.long)
    compressor = GridTokenCompressor(
        VisionCompressorConfig(input_dim=3, bottleneck_dim=5)
    )
    branches = {"main": torch.randn(8, 3), "side": torch.randn(8, 3)}
    outputs = compress_feature_branches(compressor, branches, grid)
    assert outputs["main"].plan is outputs["side"].plan
    assert (
        outputs["main"].compressed.shape == outputs["side"].compressed.shape == (2, 3)
    )
    # Two time steps are distinct groups rather than one cross-frame group.
    assert outputs["main"].plan.source_to_compressed.tolist() == [0] * 4 + [1] * 4


def test_only_compressor_receives_gradients() -> None:
    compressor = GridTokenCompressor(
        VisionCompressorConfig(input_dim=4, bottleneck_dim=8)
    )
    frozen_head = nn.Linear(4, 2)
    frozen_head.requires_grad_(False)
    output = compressor(torch.randn(4, 4), torch.tensor([[1, 4, 4]]))
    frozen_head(output.compressed).square().mean().backward()
    assert any(parameter.grad is not None for parameter in compressor.parameters())
    assert all(parameter.grad is None for parameter in frozen_head.parameters())


def test_invalid_grid_fails_closed() -> None:
    with torch.no_grad():
        try:
            build_image_token_plan(torch.tensor([[1, 5, 4]]))
        except ValueError as error:
            assert "divisible" in str(error)
        else:
            raise AssertionError("non-divisible encoder grid was accepted")

import runpy
from pathlib import Path

import pytest
import torch

from pre_prefill_compressor import (
    EMABoundedGradientController,
    GridTokenCompressor,
    VisionCompressorConfig,
    exact_mcnemar_p_value,
    load_training_checkpoint,
    paired_binary_evaluation,
    save_training_checkpoint,
    stable_config_digest,
)


def test_paired_metrics_denominators_and_exact_mcnemar() -> None:
    result = paired_binary_evaluation(
        ["a", "b", "c", "d", "positive-invalid", "negative-invalid"],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, None, None],
        [0, 0, 1, 1, 1, None],
    )
    assert result.total_rows == 6
    assert (result.baseline_valid_rows, result.baseline_invalid_rows) == (4, 2)
    assert (result.candidate_valid_rows, result.candidate_invalid_rows) == (5, 1)
    assert result.baseline.denominator == result.candidate.denominator == 6
    assert result.baseline.false_negative == 2  # includes positive invalid
    assert result.baseline.unknown_negative == 1
    assert result.candidate.unknown_negative == 1
    assert result.baseline.f1 == pytest.approx(0.4)
    assert result.candidate.f1 == pytest.approx(1.0)
    assert result.baseline.accuracy == pytest.approx(2 / 6)
    assert result.candidate.accuracy == pytest.approx(5 / 6)
    assert result.baseline_wrong_candidate_correct == 3
    assert result.baseline_correct_candidate_wrong == 0
    assert result.both_wrong == 1  # negative invalid is wrong for both models
    assert result.mcnemar_exact_p_value == pytest.approx(0.25)
    assert exact_mcnemar_p_value(0, 0) == 1.0


def test_checkpoint_roundtrip_and_stable_config_digest(tmp_path: Path) -> None:
    assert stable_config_digest({"b": 2, "a": 1}) == stable_config_digest(
        {"a": 1, "b": 2}
    )
    model = GridTokenCompressor(VisionCompressorConfig(input_dim=2, bottleneck_dim=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    controller = EMABoundedGradientController(("ce", "vsd"))
    controller.update_from_norms({"vsd": 10.0, "ce": 1.0})
    loss = (
        model(torch.randn(4, 2), torch.tensor([[1, 4, 4]])).compressed.square().mean()
    )
    loss.backward()
    optimizer.step()
    expected = {
        key: value.detach().clone() for key, value in model.state_dict().items()
    }
    path = tmp_path / "training.pt"
    config = {"optimizer": {"lr": 1e-3}, "recipe": "synthetic"}
    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        gradient_controller=controller,
        step=7,
        config=config,
        objective_history={"ce": [1.0, 0.8]},
        extra={"note": "public synthetic test"},
    )
    for parameter in model.parameters():
        parameter.data.zero_()
    restored_controller = EMABoundedGradientController(("vsd", "ce"))
    state = load_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        gradient_controller=restored_controller,
        restore_rng=False,
    )
    assert state.step == 7
    assert state.config == config
    assert state.config_digest == stable_config_digest(config)
    assert state.objective_history == {"ce": [1.0, 0.8]}
    assert restored_controller.weights == controller.weights
    for key, value in model.state_dict().items():
        assert torch.equal(value, expected[key])


def test_cpu_synthetic_training_example_runs() -> None:
    namespace = runpy.run_path(
        str(Path(__file__).parents[1] / "examples" / "train_synthetic.py"),
        run_name="synthetic_example",
    )
    metrics = namespace["run_training"](steps=1)
    assert set(metrics) >= {"task_ce", "response_jsd", "vision_semantic", "total"}
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert metrics["joint_gradient_norm"] <= 1.0 + 1e-6

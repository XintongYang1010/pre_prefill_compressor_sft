from collections import OrderedDict

import pytest
import torch

from pre_prefill_compressor import (
    EMABoundedGradientController,
    GradientBudgetConfig,
    apply_ce_anchor_cap,
    build_budgeted_gradient_update,
    collect_objective_gradients,
)


def test_key_order_is_irrelevant_but_missing_keys_fail() -> None:
    controller = EMABoundedGradientController(("ce", "vsd", "feature"))
    reversed_norms = OrderedDict(
        (key, value) for key, value in (("feature", 0.0), ("vsd", 100.0), ("ce", 1.0))
    )
    weights = controller.update_from_norms(reversed_norms)
    assert weights["feature"] == 1.0
    assert all(0.25 <= weights[key] <= 4.0 for key in ("ce", "vsd"))
    objectives = OrderedDict(
        (key, torch.tensor(value))
        for key, value in (("feature", 3.0), ("ce", 1.0), ("vsd", 2.0))
    )
    assert torch.isfinite(controller.weighted_sum(objectives))
    with pytest.raises(ValueError, match="missing"):
        controller.update_from_norms({"ce": 1.0, "vsd": 2.0})


def test_ce_anchor_cap_is_key_order_independent() -> None:
    weights = OrderedDict((key, value) for key, value in (("vsd", 1.0), ("ce", 2.0)))
    norms = OrderedDict((key, value) for key, value in (("ce", 3.0), ("vsd", 100.0)))
    capped, audit = apply_ce_anchor_cap(
        weights, norms, target_key="vsd", ce_key="ce", maximum_ratio=1.0
    )
    assert audit.cap_applied
    assert capped["vsd"] == pytest.approx(0.06)
    assert audit.target_capped_weighted_norm <= audit.reference_weighted_norm


def test_gradient_measurement_and_state_roundtrip() -> None:
    parameter = torch.nn.Parameter(torch.tensor(2.0))
    controller = EMABoundedGradientController(
        ("small", "large"),
        GradientBudgetConfig(ema_beta=0.5, minimum_weight=0.25, maximum_weight=4.0),
    )
    objectives = {"large": 10.0 * parameter.square(), "small": parameter.square()}
    norms = controller.audit_and_update(objectives, [parameter])
    assert norms["large"] == pytest.approx(10.0 * norms["small"])
    restored = EMABoundedGradientController(
        ("large", "small"),
        GradientBudgetConfig(ema_beta=0.5, minimum_weight=0.25, maximum_weight=4.0),
    )
    restored.load_state_dict(controller.state_dict())
    assert restored.ema_norms == controller.ema_norms
    assert restored.weights == controller.weights


def test_collect_objective_gradients_reports_norms_and_conflict() -> None:
    parameter = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    objectives = OrderedDict(
        (
            ("ce", parameter.sum()),
            ("opposed", -2.0 * parameter.sum()),
        )
    )
    result = collect_objective_gradients(
        objectives,
        [parameter],
        average_across_data_parallel=False,
    )
    assert result.data_parallel_world_size == 1
    assert result.norms["opposed"] == pytest.approx(2.0 * result.norms["ce"])
    assert result.pairwise_cosines["ce|opposed"] == pytest.approx(-1.0)


def test_explicit_budgeted_update_installs_capped_clipped_joint_gradient() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    controller = EMABoundedGradientController(
        ("ce", "vsd"),
        GradientBudgetConfig(ema_beta=0.0, minimum_weight=1.0, maximum_weight=1.0),
    )
    objectives = OrderedDict(
        (
            ("ce", parameter.square().sum()),
            ("vsd", 100.0 * parameter.square().sum()),
        )
    )
    audit = build_budgeted_gradient_update(
        controller,
        objectives,
        [parameter],
        ce_key="ce",
        capped_auxiliary_key="vsd",
        maximum_auxiliary_to_ce_ratio=1.0,
        maximum_global_norm=1.0,
        average_across_data_parallel=False,
    )
    assert audit.cap is not None and audit.cap.cap_applied
    assert audit.cap.target_capped_weighted_norm == pytest.approx(
        audit.cap.reference_weighted_norm
    )
    assert audit.joint_norm_before_clip > 1.0
    assert audit.joint_norm_after_clip == pytest.approx(1.0)
    assert parameter.grad is not None
    assert float(parameter.grad.norm().item()) == pytest.approx(1.0)
    assert audit.pairwise_cosines["ce|vsd"] == pytest.approx(1.0)


def test_zero_ce_gradient_safely_disables_capped_auxiliary() -> None:
    parameter = torch.nn.Parameter(torch.tensor([2.0, -1.0]))
    controller = EMABoundedGradientController(
        ("ce", "vsd"),
        GradientBudgetConfig(ema_beta=0.0, minimum_weight=1.0, maximum_weight=1.0),
    )
    audit = build_budgeted_gradient_update(
        controller,
        OrderedDict(
            (
                ("ce", 0.0 * parameter.sum()),
                ("vsd", parameter.square().sum()),
            )
        ),
        [parameter],
        ce_key="ce",
        capped_auxiliary_key="vsd",
        maximum_auxiliary_to_ce_ratio=1.0,
        average_across_data_parallel=False,
    )
    assert audit.weights_after_cap["vsd"] == 0.0
    assert audit.cap is not None and audit.cap.cap_applied
    assert audit.joint_norm_before_clip == 0.0
    assert parameter.grad is not None
    assert torch.equal(parameter.grad, torch.zeros_like(parameter))

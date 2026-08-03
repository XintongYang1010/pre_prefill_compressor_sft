from collections import OrderedDict

import pytest
import torch

from pre_prefill_compressor import (
    EMABoundedGradientController,
    GradientBudgetConfig,
    apply_ce_anchor_cap,
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

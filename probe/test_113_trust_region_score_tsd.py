from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).with_name("113_trust_region_score_tsd.py")


def load_module():
    spec = importlib.util.spec_from_file_location("trust_region_tsd_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def trust():
    return load_module()


def test_local_trust_region_caps_each_spatial_update(trust):
    parameter = torch.nn.Parameter(torch.zeros(1, 2, 2, 2))
    optimizer = trust.MeanCorrectedTrustRegionSGD([parameter], lr=1.0)
    optimizer.trust_region_rms = 10.0
    optimizer.local_trust_region_rms = 0.25
    parameter.grad = torch.tensor(
        [[[[3.0, 0.01], [0.0, 0.0]], [[4.0, 0.01], [0.0, 0.0]]]]
    )
    optimizer.step()
    local_rms = parameter.detach().square().mean(dim=1).sqrt()
    assert float(local_rms.max()) == pytest.approx(0.25)
    assert float(local_rms[0, 0, 1]) < 0.25


def test_global_trust_region_still_applies_with_local_cap(trust):
    parameter = torch.nn.Parameter(torch.zeros(1, 2, 2, 2))
    optimizer = trust.MeanCorrectedTrustRegionSGD([parameter], lr=1.0)
    optimizer.trust_region_rms = 0.1
    optimizer.local_trust_region_rms = 0.5
    parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    global_rms = parameter.detach().square().mean().sqrt()
    assert float(global_rms) == pytest.approx(0.1)

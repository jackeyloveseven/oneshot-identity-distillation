from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).with_name("115_krea2_identity_tsd.py")


@pytest.fixture(scope="module")
def tsd():
    spec = importlib.util.spec_from_file_location("krea2_tsd_under_test", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_velocity_delta_to_score_delta_sign_and_scale(tsd):
    delta_velocity = torch.ones(1, 2, 3)
    actual = tsd.score_delta_from_velocity_delta(delta_velocity, 0.25)
    assert torch.allclose(actual, torch.full_like(actual, -3.0))


def test_score_direction_matches_teacher_minus_base_x0(tsd):
    base_velocity = torch.tensor([[[1.0, -1.0]]])
    teacher_velocity = torch.tensor([[[3.0, -4.0]]])
    sigma = 0.4
    delta_score = tsd.score_delta_from_velocity_delta(
        teacher_velocity - base_velocity, sigma
    )
    teacher_minus_base_x0 = -sigma * (teacher_velocity - base_velocity)
    cosine = torch.nn.functional.cosine_similarity(
        delta_score.flatten(), teacher_minus_base_x0.flatten(), dim=0
    )
    assert float(cosine) == pytest.approx(1.0)


def test_none_normalization_preserves_score_magnitude(tsd):
    value = torch.tensor([[[1.0, -3.0]]])
    assert torch.equal(tsd.normalize_field(value, "none"), value)


def test_packed_trust_region_caps_global_and_each_token(tsd):
    anchor = torch.zeros(1, 4, 8)
    candidate = torch.full_like(anchor, 10.0)
    projected, global_rms, max_token_rms = tsd.project_trust_region(
        candidate, anchor, global_radius=0.1, local_radius=0.06
    )
    assert projected.shape == anchor.shape
    assert global_rms <= 0.1 + 1e-6
    assert max_token_rms <= 0.06 + 1e-6


def test_adaptive_trust_matches_one_euler_step_and_caps(tsd):
    delta_velocity = torch.full((1, 4, 8), 2.0)
    global_radius, local_radius = tsd.adaptive_trust_radii(
        delta_velocity,
        delta_sigma=0.04,
        ratio=0.5,
        global_cap=0.1,
        local_cap=0.08,
    )
    assert global_radius == pytest.approx(0.04)
    assert local_radius == pytest.approx(0.032)

    capped_global, capped_local = tsd.adaptive_trust_radii(
        delta_velocity,
        delta_sigma=1.0,
        ratio=1.0,
        global_cap=0.1,
        local_cap=0.08,
    )
    assert capped_global == pytest.approx(0.1)
    assert capped_local == pytest.approx(0.08)


def test_packed_roi_gate_preserves_only_selected_tokens(tsd):
    gate = tsd.packed_roi_gate(
        4,
        4,
        (0.25, 0.25, 0.75, 0.75),
        0.0,
        device=torch.device("cpu"),
    )
    assert gate.shape == (1, 16, 1)
    image_gate = gate.reshape(4, 4)
    assert float(image_gate[1:3, 1:3].sum()) == pytest.approx(4.0)
    assert float(image_gate.sum()) == pytest.approx(4.0)


def test_ref_boost_bias_only_targets_reference_keys(tsd):
    bias = tsd.ref_boost_bias(
        text_length=2,
        source_length=3,
        target_length=4,
        ref_boost=4.0,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )
    assert bias.shape == (1, 1, 9, 9)
    expected = torch.zeros_like(bias)
    expected[:, :, 5:, 2:5] = torch.log(torch.tensor(4.0))
    assert torch.allclose(bias, expected)
    assert (
        tsd.ref_boost_bias(
            2,
            3,
            4,
            1.0,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        is None
    )


def test_identity_anchor_interpolates_edit_and_reference_velocities(tsd):
    teacher = torch.zeros(1, 2, 3)
    reference = torch.full_like(teacher, 4.0)
    actual = tsd.identity_anchored_velocity(teacher, reference, 0.25)
    assert torch.allclose(actual, torch.ones_like(actual))


def test_identity_stat_transport_changes_channels_not_token_layout(tsd):
    edit = torch.tensor([[[1.0, 2.0], [-1.0, 2.0]]])
    reference = torch.tensor([[[4.0, 1.0], [-4.0, 1.0]]])
    actual = tsd.identity_stat_transport(
        edit, reference, weight=1.0, gain_min=0.25, gain_max=4.0
    )
    expected = torch.tensor([[[4.0, 1.0], [-4.0, 1.0]]])
    assert torch.allclose(actual, expected)
    assert torch.equal(torch.sign(actual), torch.sign(edit))

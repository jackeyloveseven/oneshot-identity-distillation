from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch


SCRIPT = Path(__file__).with_name("112_true_score_tsd.py")


def load_module():
    spec = importlib.util.spec_from_file_location("true_score_tsd_under_test", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_omni_prompt_segments_match_one_condition_image_protocol(tsd):
    segments = tsd.build_omni_prompt_segments("draw the target", 1)
    assert segments == [
        "<|im_start|>user\n<|vision_start|>",
        (
            "<|vision_end|>draw the target<|im_end|>\n"
            "<|im_start|>assistant\n<|vision_start|>"
        ),
        "<|vision_end|><|im_end|>",
    ]


def test_omni_prompt_segments_insert_one_boundary_per_extra_image(tsd):
    segments = tsd.build_omni_prompt_segments("target", 3)
    assert len(segments) == 5
    assert segments[1:3] == [
        "<|vision_end|><|vision_start|>",
        "<|vision_end|><|vision_start|>",
    ]


def test_omni_prompt_segments_reject_negative_condition_count(tsd):
    with pytest.raises(ValueError, match="non-negative"):
        tsd.build_omni_prompt_segments("target", -1)


@pytest.fixture(scope="module")
def tsd():
    return load_module()


def test_visual_identity_subtract_is_difference_in_differences(tsd):
    positive = torch.tensor([1.0, 3.0, 5.0])
    negative = torch.tensor([0.5, 1.0, 2.0])
    result, beta = tsd.visual_identity_contrast(
        positive,
        negative,
        "visual_identity_subtract",
    )
    assert beta == 1.0
    torch.testing.assert_close(result, positive - negative)


def test_visual_identity_projection_removes_parallel_component(tsd):
    positive = torch.tensor([1.5, 2.0])
    negative = torch.tensor([1.0, 0.0])
    result, beta = tsd.visual_identity_contrast(
        positive,
        negative,
        "visual_identity_project",
    )
    assert beta == pytest.approx(1.5)
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0]))


def test_visual_identity_projection_preserves_orthogonal_signal(tsd):
    positive = torch.tensor([0.0, 4.0])
    negative = torch.tensor([2.0, 0.0])
    result, beta = tsd.visual_identity_contrast(
        positive,
        negative,
        "visual_identity_project",
    )
    assert beta == pytest.approx(0.0)
    torch.testing.assert_close(result, positive)


def test_visual_identity_projection_supports_shrinkage(tsd):
    positive = torch.tensor([1.5, 2.0])
    negative = torch.tensor([1.0, 0.0])
    result, beta = tsd.visual_identity_contrast(
        positive,
        negative,
        "visual_identity_project",
        projection_strength=0.5,
    )
    assert beta == pytest.approx(0.75)
    torch.testing.assert_close(result, torch.tensor([0.75, 2.0]))


def test_gated_identity_with_zero_shared_keep_matches_full_projection(tsd):
    positive = torch.zeros(2, 1, 4, 4)
    negative = torch.ones_like(positive)
    positive[..., 0, 0] = 2.0
    positive[..., 3, 3] = -2.0
    projected, beta = tsd.visual_identity_contrast(
        positive,
        negative,
        "visual_identity_project",
    )
    gated, gated_beta, support = tsd.gated_visual_identity_contrast(
        positive,
        negative,
        shared_keep=0.0,
        support_quantile=0.75,
        support_gate_power=1.0,
    )
    assert gated_beta == pytest.approx(beta)
    torch.testing.assert_close(gated, projected)
    assert support.shape == (1, 1, 4, 4)
    assert float(support.min()) >= 0.0
    assert float(support.max()) <= 1.0


def test_subject_support_gate_is_independent_of_teacher_energy(tsd):
    positive = torch.full((2, 1, 4, 4), 2.0)
    negative = torch.ones_like(positive)
    subject = torch.zeros_like(positive)
    subject[..., 1:3, 1:3] = 4.0

    result, beta, gate = tsd.subject_support_visual_identity_contrast(
        positive,
        negative,
        subject,
        shared_keep=1.0,
        outside_keep=0.0,
        support_quantile=0.75,
        support_gate_power=1.0,
    )

    assert beta == pytest.approx(2.0)
    assert gate.shape == (1, 1, 4, 4)
    torch.testing.assert_close(result, gate * positive)


def test_subject_support_gate_can_keep_projected_signal_outside(tsd):
    positive = torch.tensor([[[[2.0, 2.0], [2.0, 2.0]]]])
    negative = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]])
    subject = torch.tensor([[[[4.0, 0.0], [0.0, 0.0]]]])
    projected, _ = tsd.visual_identity_contrast(
        positive,
        negative,
        "visual_identity_project",
    )
    result, _, gate = tsd.subject_support_visual_identity_contrast(
        positive,
        negative,
        subject,
        shared_keep=0.0,
        outside_keep=1.0,
        support_quantile=0.75,
        support_gate_power=1.0,
    )
    torch.testing.assert_close(result, projected)
    assert float(gate.min()) >= 0.0
    assert float(gate.max()) <= 1.0


def test_token_support_gate_uses_supplied_attention_map(tsd):
    positive = torch.full((2, 1, 4, 4), 2.0)
    negative = torch.ones_like(positive)
    token_support = torch.zeros(1, 1, 4, 4)
    token_support[..., 1:3, 1:3] = 4.0
    result, beta, gate = tsd.token_support_visual_identity_contrast(
        positive,
        negative,
        token_support,
        shared_keep=1.0,
        outside_keep=0.0,
        support_quantile=0.75,
        support_gate_power=1.0,
    )
    assert beta == pytest.approx(2.0)
    torch.testing.assert_close(result, gate * positive)


def test_parse_layer_indices(tsd):
    assert tsd.parse_layer_indices("3,5-7") == {3, 5, 6, 7}


def test_low_rank_subspace_removes_shared_nuisance(tsd):
    positives = [
        torch.tensor([3.0, 2.0]),
        torch.tensor([3.0, 2.0]),
    ]
    negatives = [
        torch.tensor([3.0, 0.0]),
        torch.tensor([3.0, 0.0]),
    ]
    result, diagnostics = tsd.low_rank_nuisance_identity_contrast(
        positives,
        negatives,
        rank=1,
        ridge=0.0,
        projection_strength=1.0,
        consensus_rescue=0.0,
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0]))
    assert diagnostics["effective_rank"] == 1
    assert diagnostics["removed_fraction"] > 0.0


def test_low_rank_subspace_preserves_identity_orthogonal_to_view_variation(tsd):
    positives = [
        torch.tensor([1.0, 2.0]),
        torch.tensor([-1.0, 2.0]),
    ]
    negatives = [
        torch.tensor([1.0, 0.0]),
        torch.tensor([-1.0, 0.0]),
    ]
    result, _ = tsd.low_rank_nuisance_identity_contrast(
        positives,
        negatives,
        rank=1,
        ridge=0.0,
        projection_strength=1.0,
        consensus_rescue=0.0,
    )
    torch.testing.assert_close(result, torch.tensor([0.0, 2.0]))


def test_consensus_rescue_restores_discriminative_collinear_component(tsd):
    positives = [torch.tensor([3.0]), torch.tensor([3.0])]
    negatives = [torch.tensor([2.0]), torch.tensor([2.0])]
    without_rescue, _ = tsd.low_rank_nuisance_identity_contrast(
        positives,
        negatives,
        rank=1,
        ridge=0.0,
        projection_strength=1.0,
        consensus_rescue=0.0,
    )
    with_rescue, diagnostics = tsd.low_rank_nuisance_identity_contrast(
        positives,
        negatives,
        rank=1,
        ridge=0.0,
        projection_strength=1.0,
        consensus_rescue=1.0,
    )
    assert float(without_rescue) == pytest.approx(0.0, abs=1e-6)
    assert float(with_rescue) > float(without_rescue)
    assert diagnostics["rescue_fraction"] > 0.0


def test_kv_value_projection_keeps_positive_keys_and_filters_values(tsd):
    positive = [
        {
            3: (
                torch.tensor([[[[7.0, 8.0]]]]),
                torch.tensor([[[[3.0, 2.0]]]]),
            )
        },
        {
            3: (
                torch.tensor([[[[9.0, 10.0]]]]),
                torch.tensor([[[[3.0, 2.0]]]]),
            )
        },
    ]
    negative = [
        {
            3: (
                torch.tensor([[[[1.0, 1.0]]]]),
                torch.tensor([[[[3.0, 0.0]]]]),
            )
        },
        {
            3: (
                torch.tensor([[[[2.0, 2.0]]]]),
                torch.tensor([[[[3.0, 0.0]]]]),
            )
        },
    ]
    result, diagnostics = tsd.project_reference_value_stores(
        positive,
        negative,
        projection_strength=1.0,
    )
    keys, values = result[3]
    torch.testing.assert_close(keys, torch.tensor([[[[8.0, 9.0]]]]))
    torch.testing.assert_close(values, torch.tensor([[[[0.0, 2.0]]]]))
    assert diagnostics["beta_mean"] == pytest.approx(1.0)
    assert diagnostics["value_retained_rms_ratio_mean"] < 1.0


def test_prediction_delta_to_score_has_z_image_sign(tsd):
    delta_prediction = torch.ones(16, 1, 2, 2)
    result = tsd.score_delta_from_prediction_delta(
        delta_prediction,
        torch.tensor(0.5),
    )
    torch.testing.assert_close(result, torch.ones(1, 16, 2, 2))


def test_classifier_free_guidance_matches_z_image_pipeline_formula(tsd):
    conditional = torch.tensor([3.0, 5.0])
    unconditional = torch.tensor([1.0, 2.0])
    result = tsd.classifier_free_guided_prediction(
        conditional,
        unconditional,
        4.0,
    )
    torch.testing.assert_close(
        result,
        conditional + 4.0 * (conditional - unconditional),
    )


def test_project_simplex_returns_nonnegative_unit_sum(tsd):
    projected = tsd.project_simplex(torch.tensor([1.7, -0.2, 0.4]))
    assert float(projected.min()) >= 0.0
    assert float(projected.sum()) == pytest.approx(1.0)


def test_pareto_common_gradient_balances_conflicting_scales(tsd):
    left = torch.tensor([100.0, 0.0])
    right = torch.tensor([0.0, 1.0])
    common, weights = tsd.pareto_common_gradient(
        [left, right],
        iterations=128,
    )
    assert weights.tolist() == pytest.approx([0.5, 0.5], abs=1e-3)
    assert float(common[0]) == pytest.approx(float(common[1]), rel=1e-3)


def test_pareto_common_gradient_cancels_exact_opposition(tsd):
    common, weights = tsd.pareto_common_gradient(
        [torch.tensor([2.0, 0.0]), torch.tensor([-3.0, 0.0])],
        iterations=128,
    )
    tensor_rms = float(common.square().mean().sqrt())
    assert tensor_rms < 1e-4
    assert float(weights.sum()) == pytest.approx(1.0)


def test_identity_projection_makes_descent_nonincreasing_for_protected_loss(tsd):
    identity_gradient = torch.tensor([-1.0, 1.0])
    protected_gradient = torch.tensor([1.0, 0.0])
    projected, diagnostics = tsd.project_identity_gradient_to_protected_cone(
        identity_gradient,
        [protected_gradient],
    )
    assert float(projected @ protected_gradient) >= -1e-7
    assert diagnostics["constraint_dot_before"][0] < 0.0
    assert diagnostics["constraint_dot_after"][0] >= -1e-7


def test_positive_local_alignment_removes_orthogonal_score_components(tsd):
    source = torch.tensor([[[[2.0]], [[3.0]], [[4.0]]]])
    identity = torch.tensor([[[[1.0]], [[0.0]], [[0.0]]]])
    projected, diagnostics = tsd.positive_local_alignment_projection(
        source,
        identity,
        max_coefficient=3.0,
    )
    torch.testing.assert_close(
        projected,
        torch.tensor([[[[2.0]], [[0.0]], [[0.0]]]]),
    )
    assert diagnostics["positive_support_fraction"] == pytest.approx(1.0)
    assert diagnostics["retained_score_rms_ratio"] < 1.0


def test_positive_local_alignment_rejects_opposed_teacher_direction(tsd):
    source = torch.tensor([[[[-2.0]], [[0.0]]]])
    identity = torch.tensor([[[[1.0]], [[0.0]]]])
    projected, diagnostics = tsd.positive_local_alignment_projection(
        source,
        identity,
    )
    torch.testing.assert_close(projected, torch.zeros_like(source))
    assert diagnostics["positive_support_fraction"] == pytest.approx(0.0)


def test_decoded_identity_loss_has_useful_x0_gradient(tsd):
    class FakeVAE(torch.nn.Module):
        class Config:
            scaling_factor = 1.0
            shift_factor = 0.0

        config = Config()

        @property
        def dtype(self):
            return torch.float32

        def decode(self, value, return_dict=False):
            return (torch.tanh(value[:, :3]),)

    class FakeCCIP(torch.nn.Module):
        def forward(self, value):
            return value.mean(dim=(-2, -1))

    vae = FakeVAE()
    model = FakeCCIP()
    desired_pixels = torch.empty(4, 3, 8, 8)
    desired_pixels[:, 0].fill_(0.8)
    desired_pixels[:, 1:].fill_(-0.5)
    with torch.no_grad():
        target = tsd.ccip_embed(model, desired_pixels).mean(0, keepdim=True)
        target = torch.nn.functional.normalize(target, dim=-1)

    x0 = torch.zeros(1, 3, 8, 8, requires_grad=True)
    loss_before = tsd.decoded_identity_loss(
        model,
        vae,
        x0,
        target,
        fixed_batch_size=4,
    )
    gradient = torch.autograd.grad(loss_before, x0)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.square().mean().sqrt()) > 0.0

    with torch.no_grad():
        x1 = x0 - 0.05 * gradient / gradient.square().mean().sqrt()
    loss_after = tsd.decoded_identity_loss(
        model,
        vae,
        x1,
        target,
        fixed_batch_size=4,
    )
    assert float(loss_after.detach()) < float(loss_before.detach())


def test_normalized_identity_roi_crop_and_validation(tsd):
    roi = tsd.parse_normalized_box("0.25,0.25,0.75,0.75")
    pixels = torch.arange(3 * 8 * 8).reshape(1, 3, 8, 8)
    cropped = tsd.crop_normalized_roi(pixels, roi)
    assert cropped.shape == (1, 3, 4, 4)
    with pytest.raises(ValueError):
        tsd.parse_normalized_box("0.8,0.2,0.4,0.9")


def test_outside_roi_loss_ignores_changes_inside_roi(tsd):
    anchor = torch.zeros(1, 3, 8, 8)
    changed = anchor.clone()
    changed[..., 2:6, 2:6] = 1.0
    roi = (0.25, 0.25, 0.75, 0.75)
    assert float(tsd.outside_roi_l1_loss(changed, anchor, roi)) == pytest.approx(0.0)
    changed[..., 0, 0] = 1.0
    assert float(tsd.outside_roi_l1_loss(changed, anchor, roi)) > 0.0


def test_dino_roi_identity_loss_has_gradient(tsd):
    class FakeDINO(torch.nn.Module):
        @property
        def dtype(self):
            return torch.float32

        def forward(self, pixel_values):
            class Output:
                pooler_output = pixel_values.mean(dim=(-2, -1))
                last_hidden_state = pixel_values.flatten(2).transpose(1, 2)

            return Output()

    model = FakeDINO()
    pixels = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target_pixels = torch.zeros(1, 3, 8, 8)
    target_pixels[:, 0] = 0.8
    target_pixels[:, 1:] = -0.5
    with torch.no_grad():
        target = tsd.dino_embed(model, target_pixels)
    loss = tsd.dino_identity_loss_from_pixels(
        model,
        pixels,
        target,
        roi=(0.25, 0.25, 0.75, 0.75),
    )
    gradient = torch.autograd.grad(loss, pixels)[0]
    assert torch.isfinite(gradient).all()
    assert float(gradient.square().mean().sqrt()) > 0.0


def test_dino_patch_soft_chamfer_has_spatial_gradient(tsd):
    class FakeDINO(torch.nn.Module):
        @property
        def dtype(self):
            return torch.float32

        def forward(self, pixel_values):
            tokens = pixel_values.flatten(2).transpose(1, 2)

            class Output:
                pooler_output = tokens.mean(dim=1)
                last_hidden_state = torch.cat(
                    [pooler_output[:, None], tokens],
                    dim=1,
                )

            return Output()

    model = FakeDINO()
    pixels = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target_pixels = torch.zeros(1, 3, 8, 8)
    target_pixels[:, 0, :4] = 0.9
    target_pixels[:, 1, 4:] = 0.7
    with torch.no_grad():
        target = tsd.dino_patch_embed(model, target_pixels).reshape(-1, 3)
    loss = tsd.dino_patch_identity_loss_from_pixels(
        model,
        pixels,
        target,
        roi=(0.25, 0.25, 0.75, 0.75),
        temperature=0.05,
    )
    gradient = torch.autograd.grad(loss, pixels)[0]
    assert torch.isfinite(loss)
    assert torch.isfinite(gradient).all()
    assert float(gradient.square().mean().sqrt()) > 0.0


def test_soft_chamfer_is_lower_for_matching_patch_sets(tsd):
    target = torch.eye(4)
    matching = target.clone()
    mismatched = -target
    matched_loss = tsd.bidirectional_soft_chamfer_cosine(
        matching,
        target,
        temperature=0.05,
    )
    mismatched_loss = tsd.bidirectional_soft_chamfer_cosine(
        mismatched,
        target,
        temperature=0.05,
    )
    assert float(matched_loss) < float(mismatched_loss)


def test_lookahead_velocity_correction_is_descent_with_exact_rms(tsd):
    gradient = torch.tensor([[[[1.0, -2.0], [0.5, 3.0]]]])
    sigma = torch.tensor(0.7)
    next_sigma = torch.tensor(0.5)
    radius = 0.006
    correction = tsd.lookahead_identity_velocity_correction(
        gradient,
        sigma,
        next_sigma,
        next_latent_rms=radius,
    )
    next_latent_delta = (next_sigma - sigma) * correction
    assert float(next_latent_delta.square().mean().sqrt()) == pytest.approx(
        radius,
        rel=1e-5,
    )
    assert float((next_latent_delta * gradient).sum()) < 0.0


def test_multiscale_pseudo_target_loss_localizes_and_smooths_gradient(tsd):
    prediction = torch.zeros(1, 1, 8, 8, requires_grad=True)
    target = torch.zeros_like(prediction)
    target[..., 3:5, 3:5] = 1.0
    loss = tsd.multiscale_pseudo_target_loss(
        prediction,
        target,
        roi=(0.25, 0.25, 0.75, 0.75),
        outside_weight=0.0,
        lowpass_kernel=2,
        lowpass_weight=1.0,
    )
    gradient = torch.autograd.grad(loss, prediction)[0]
    assert float(loss.detach()) > 0.0
    assert torch.count_nonzero(gradient[..., :2, :]).item() == 0
    assert torch.count_nonzero(gradient[..., :, :2]).item() == 0
    assert torch.count_nonzero(gradient[..., 2:6, 2:6]).item() > 4


def test_multiscale_pseudo_target_native_mode_matches_half_mse(tsd):
    prediction = torch.tensor([[[[0.0, 1.0], [2.0, 3.0]]]])
    target = torch.zeros_like(prediction)
    actual = tsd.multiscale_pseudo_target_loss(prediction, target)
    expected = 0.5 * torch.nn.functional.mse_loss(prediction, target)
    assert float(actual) == pytest.approx(float(expected))

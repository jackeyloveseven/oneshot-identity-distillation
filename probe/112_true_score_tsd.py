#!/usr/bin/env python3
"""True inner-loop test-time score distillation for one identity-edit image.

Unlike ``110_identity_residual_tsd.py``, this script does not hand-add one
residual displacement. Every outer iteration first performs exactly one pure
T2I scheduler step, then optimizes that post-step latent at the next scheduler
sigma with AdamW against a detached x0 pseudo-target. By default the transient
teacher uses the transformer's native Omni multi-image interface; the legacy
ad-hoc reference K/V path remains available only as an explicit ablation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
BASE_IMPL = ROOT / "probe" / "110_identity_residual_tsd.py"
OUT_ROOT = ROOT / "out" / "true_score_tsd"


def load_impl():
    spec = importlib.util.spec_from_file_location("cherry_identity_residual_impl", BASE_IMPL)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_IMPL}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument(
        "--negative-references",
        default=None,
        help=(
            "comma-separated mismatched-ID references used to center the visual "
            "identity score; required by visual_identity_* contrasts"
        ),
    )
    parser.add_argument(
        "--identity-capsule-file",
        default=None,
        help="optional JSON capsule; omit it for the visual-reference-only ablation",
    )
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default=None)
    parser.add_argument(
        "--teacher-backend",
        choices=["native_omni", "legacy_kv"],
        default="legacy_kv",
        help=(
            "native_omni is a diagnostic that uses official vision-token "
            "prompt framing, but requires an actually reference-trained "
            "checkpoint; legacy_kv preserves the original experimental "
            "attention-cache injection path"
        ),
    )
    parser.add_argument(
        "--capture-prompt-mode",
        choices=["target", "generic", "without_identity"],
        default="target",
        help=(
            "caption context for the noisy target during reference K/V capture; "
            "target is state-matched, generic reproduces the legacy probe"
        ),
    )
    parser.add_argument(
        "--reference-caption-mode",
        choices=["identity", "generic"],
        default="identity",
        help=(
            "caption paired with the clean reference during capture; identity "
            "uses the capsule when available and otherwise falls back to generic"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument(
        "--inner-x0-mode",
        choices=["cfg", "conditional"],
        default="cfg",
        help=(
            "prediction used to form decoded inner-loop x0; cfg matches the "
            "actual scheduler trajectory, conditional reproduces the legacy bug"
        ),
    )
    parser.add_argument("--teacher-weight", type=float, default=4.0)
    parser.add_argument("--inner-iters", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--pseudo-step-size", type=float, default=0.05)
    parser.add_argument("--style-weight", type=float, default=2.0)
    parser.add_argument("--content-weight", type=float, default=0.2)
    parser.add_argument(
        "--distill-roi",
        default=None,
        help=(
            "optional normalized x0,y0,x1,y1 latent-space emphasis for the "
            "teacher pseudo-target loss"
        ),
    )
    parser.add_argument(
        "--distill-outside-weight",
        type=float,
        default=1.0,
        help="relative pseudo-target loss weight outside --distill-roi",
    )
    parser.add_argument(
        "--distill-lowpass-kernel",
        type=int,
        default=1,
        help="average-pooling kernel for the structural pseudo-target term",
    )
    parser.add_argument(
        "--distill-lowpass-weight",
        type=float,
        default=0.0,
        help="mixture weight of structural low-pass loss versus native loss",
    )
    parser.add_argument(
        "--identity-loss-weight",
        type=float,
        default=0.0,
        help=(
            "weight of a differentiable CCIP cosine loss on the decoded x0 "
            "estimate; zero preserves the original score-only experiment"
        ),
    )
    parser.add_argument(
        "--identity-model-path",
        default=(
            "out/trainlight/aux_models_cache/"
            "ccip_feat_simplified_bs4.onnx"
        ),
        help="fixed-batch simplified CCIP feature ONNX converted with onnx2torch",
    )
    parser.add_argument(
        "--identity-model-batch-size",
        type=int,
        default=4,
        help="fixed input batch size baked into --identity-model-path",
    )
    parser.add_argument(
        "--identity-roi",
        default=None,
        help=(
            "optional normalized generated-image ROI x0,y0,x1,y1 used before "
            "identity encoding; reference images remain full-frame"
        ),
    )
    parser.add_argument(
        "--identity-dino-weight",
        type=float,
        default=0.0,
        help="additive DINOv2 cosine-loss weight on the same generated ROI",
    )
    parser.add_argument(
        "--identity-dino-patch-weight",
        type=float,
        default=0.0,
        help=(
            "additive bidirectional soft-Chamfer loss on spatial DINOv2 patch "
            "tokens; unlike the CLS loss this penalizes missing local parts"
        ),
    )
    parser.add_argument(
        "--identity-dino-patch-temperature",
        type=float,
        default=0.05,
        help="soft-min temperature for spatial DINOv2 patch matching",
    )
    parser.add_argument(
        "--identity-gradient-mode",
        choices=[
            "weighted_sum",
            "pareto_constrained",
            "pareto_aligned_distill",
        ],
        default="weighted_sum",
        help=(
            "weighted_sum reproduces the original scalarized loss; "
            "pareto_constrained computes a minimum-norm common identity "
            "gradient and projects it so its descent step cannot increase "
            "the score/content trust objectives to first order; "
            "pareto_aligned_distill removes every local teacher-score "
            "component not positively aligned with the identity consensus"
        ),
    )
    parser.add_argument(
        "--identity-alignment-max-coefficient",
        type=float,
        default=2.0,
        help=(
            "maximum local least-squares coefficient when projecting the "
            "teacher-score gradient onto the Pareto identity direction"
        ),
    )
    parser.add_argument(
        "--identity-pareto-iters",
        type=int,
        default=64,
        help="projected-simplex iterations for the identity Pareto direction",
    )
    parser.add_argument(
        "--identity-constraint-iters",
        type=int,
        default=8,
        help="alternating half-space projections for protected gradients",
    )
    parser.add_argument(
        "--identity-dino-model",
        default="facebook/dinov2-small",
        help="local Hugging Face DINOv2 model used for independent ROI features",
    )
    parser.add_argument(
        "--identity-sigma-max",
        type=float,
        default=0.5,
        help="apply decoded-x0 identity supervision only at sigma <= this value",
    )
    parser.add_argument(
        "--identity-trajectory-rms",
        type=float,
        default=0.0,
        help=(
            "optional RMS radius of a look-ahead identity correction consumed "
            "by the current scheduler step; zero keeps post-step supervision"
        ),
    )
    parser.add_argument(
        "--identity-terminal-iters",
        type=int,
        default=0,
        help=(
            "optional clean-latent identity projection after the final scheduler "
            "step; the 113 wrapper applies the same global/local trust region"
        ),
    )
    parser.add_argument(
        "--identity-terminal-lr",
        type=float,
        default=None,
        help="terminal identity projection learning rate; defaults to --lr",
    )
    parser.add_argument(
        "--identity-terminal-pixel-weight",
        type=float,
        default=0.0,
        help=(
            "decoded-pixel L1 trust penalty to the pre-terminal clean image; "
            "suppresses CCIP shortcuts not controlled by latent RMS alone"
        ),
    )
    parser.add_argument(
        "--identity-terminal-outside-pixel-weight",
        type=float,
        default=0.0,
        help=(
            "extra decoded-pixel L1 weight outside --identity-roi, preserving "
            "the background while allowing identity changes inside the ROI"
        ),
    )
    parser.add_argument("--active-start", type=float, default=0.0)
    parser.add_argument("--active-end", type=float, default=0.06)
    parser.add_argument(
        "--projection-strength",
        type=float,
        default=1.0,
        help=(
            "visual_identity_project only: alpha in r_pos - alpha * beta_lsq * "
            "r_neg; 0 keeps the raw teacher residual and 1 fully projects"
        ),
    )
    parser.add_argument(
        "--shared-keep",
        type=float,
        default=0.5,
        help=(
            "visual_identity_gated only: fraction of the projected common "
            "reference component restored inside target-specific support"
        ),
    )
    parser.add_argument(
        "--support-quantile",
        type=float,
        default=0.75,
        help="visual_identity_gated only: energy quantile mapped to full support",
    )
    parser.add_argument(
        "--support-gate-power",
        type=float,
        default=1.0,
        help="visual_identity_gated only: exponent applied to the spatial support gate",
    )
    parser.add_argument(
        "--background-prompt",
        default=None,
        help=(
            "visual_identity_subject_gate only: matched scene prompt with the "
            "subject removed; its frozen-base contrast localizes target support"
        ),
    )
    parser.add_argument(
        "--outside-keep",
        type=float,
        default=0.0,
        help=(
            "visual_identity_subject_gate only: fraction of the projected "
            "identity residual retained outside base-localized subject support"
        ),
    )
    parser.add_argument(
        "--subject-token",
        default="character",
        help=(
            "visual_identity_token_gate only: prompt phrase whose frozen-base "
            "attention localizes the generated subject"
        ),
    )
    parser.add_argument(
        "--support-layers",
        default="16-25",
        help="visual_identity_token_gate only: inclusive attention layer range",
    )
    parser.add_argument(
        "--subspace-rank",
        type=int,
        default=4,
        help="visual_identity_subspace only: maximum nuisance-subspace rank",
    )
    parser.add_argument(
        "--subspace-ridge",
        type=float,
        default=0.05,
        help="visual_identity_subspace only: spectral shrinkage ridge",
    )
    parser.add_argument(
        "--subspace-strength",
        type=float,
        default=1.0,
        help="visual_identity_subspace only: nuisance projection removal strength",
    )
    parser.add_argument(
        "--consensus-rescue",
        type=float,
        default=0.5,
        help=(
            "visual_identity_subspace only: restore removed components where "
            "positive references agree and differ from negative references"
        ),
    )
    parser.add_argument(
        "--kv-value-projection-strength",
        type=float,
        default=1.0,
        help=(
            "visual_identity_kv_value_project only: remove the mismatched-reference "
            "component from captured teacher values while retaining positive keys"
        ),
    )
    parser.add_argument(
        "--residual-contrast",
        choices=[
            "teacher_base",
            "identity_interaction",
            "visual_identity_subtract",
            "visual_identity_project",
            "visual_identity_gated",
            "visual_identity_subject_gate",
            "visual_identity_token_gate",
            "visual_identity_subspace",
            "visual_identity_kv_value_project",
        ],
        default="identity_interaction",
    )
    parser.add_argument(
        "--distillation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="disable for a matched pure-T2I baseline",
    )
    parser.add_argument("--taper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--tag", default="trial")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def prediction_to_velocity(prediction: torch.Tensor) -> torch.Tensor:
    return (-prediction).unsqueeze(0).squeeze(2)


def classifier_free_guided_prediction(
    conditional: torch.Tensor,
    unconditional: torch.Tensor,
    guidance_scale: float,
) -> torch.Tensor:
    if conditional.shape != unconditional.shape:
        raise ValueError("conditional and unconditional predictions must share shape")
    return conditional + float(guidance_scale) * (
        conditional - unconditional
    )


def build_omni_prompt_segments(
    prompt: str,
    num_condition_images: int = 1,
) -> list[str]:
    """Build the exact text segments expected by ZImageOmniPipeline.

    Condition-image boundaries are represented in the language sequence, so
    independently encoding a reference caption and target caption is not an
    equivalent model input.
    """

    if num_condition_images < 0:
        raise ValueError("num_condition_images must be non-negative")
    if num_condition_images == 0:
        return [
            "<|im_start|>user\n"
            + prompt
            + "<|im_end|>\n<|im_start|>assistant\n"
        ]
    segments = ["<|im_start|>user\n<|vision_start|>"]
    segments.extend(
        ["<|vision_end|><|vision_start|>"] * (num_condition_images - 1)
    )
    segments.append(
        "<|vision_end|>"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n<|vision_start|>"
    )
    segments.append("<|vision_end|><|im_end|>")
    return segments


def encode_omni_prompt(
    pipe,
    prompt: str,
    *,
    device: torch.device | str,
    num_condition_images: int = 1,
    max_sequence_length: int = 512,
) -> list[torch.Tensor]:
    """Encode one Omni prompt into its per-image-boundary embedding segments."""

    segments = build_omni_prompt_segments(prompt, num_condition_images)
    text_inputs = pipe.tokenizer(
        segments,
        padding="max_length",
        max_length=max_sequence_length,
        truncation=True,
        return_tensors="pt",
    )
    input_ids = text_inputs.input_ids.to(device)
    attention_mask = text_inputs.attention_mask.to(device).bool()
    hidden = pipe.text_encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        output_hidden_states=True,
    ).hidden_states[-2]
    return [
        hidden[index][attention_mask[index]]
        for index in range(len(segments))
    ]


def score_delta_from_prediction_delta(
    prediction_delta: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-4,
) -> torch.Tensor:
    """Convert Z-Image p-teacher minus p-base into a same-state score delta."""

    sigma_safe = sigma.float().clamp_min(eps)
    coefficient = (1.0 - sigma.float()) / sigma_safe
    return coefficient * prediction_delta.float().unsqueeze(0).squeeze(2)


def rms_normalize_per_sample(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    flat = value.float().reshape(value.shape[0], -1)
    rms = flat.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    return value.float() / rms.reshape(value.shape[0], *([1] * (value.ndim - 1)))


def tensor_rms(value: torch.Tensor) -> float:
    return float(value.detach().float().square().mean().sqrt().item())


def tensor_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> float:
    left_flat = left.detach().float().reshape(-1)
    right_flat = right.detach().float().reshape(-1)
    denominator = left_flat.norm() * right_flat.norm()
    return float((left_flat @ right_flat / denominator.clamp_min(eps)).item())


def project_simplex(value: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of a vector onto the probability simplex."""

    if value.ndim != 1 or value.numel() < 1:
        raise ValueError("simplex projection expects a non-empty vector")
    sorted_value, _ = torch.sort(value.float(), descending=True)
    cumulative = torch.cumsum(sorted_value, dim=0) - 1.0
    indices = torch.arange(
        1,
        value.numel() + 1,
        device=value.device,
        dtype=torch.float32,
    )
    support = sorted_value - cumulative / indices > 0.0
    rho = int(support.nonzero(as_tuple=False)[-1].item())
    threshold = cumulative[rho] / float(rho + 1)
    return (value.float() - threshold).clamp_min(0.0).to(value.dtype)


def minimum_norm_convex_weights(
    gradients: list[torch.Tensor],
    *,
    iterations: int = 64,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Solve min_{w in simplex} ||sum_i w_i g_i||^2 for a small gradient set."""

    if not gradients:
        raise ValueError("at least one gradient is required")
    if iterations < 1:
        raise ValueError("Pareto iterations must be positive")
    shape = gradients[0].shape
    if any(gradient.shape != shape for gradient in gradients):
        raise ValueError("all gradients must share a shape")
    if len(gradients) == 1:
        return gradients[0].new_ones((1,), dtype=torch.float32)

    rows = torch.stack(
        [gradient.detach().float().reshape(-1) for gradient in gradients],
        dim=0,
    )
    gram = rows @ rows.transpose(0, 1)
    spectral_bound = gram.abs().sum(dim=1).max().clamp_min(eps)
    step_size = 1.0 / spectral_bound
    weights = rows.new_full((len(gradients),), 1.0 / len(gradients))
    for _ in range(iterations):
        weights = project_simplex(weights - step_size * (gram @ weights))
    return weights


def pareto_common_gradient(
    gradients: list[torch.Tensor],
    *,
    iterations: int = 64,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a scale-preserving minimum-norm common identity gradient.

    Each objective gradient is RMS-normalized before the simplex solve so a
    numerically large encoder cannot dominate merely because of feature scale.
    The result is restored to the mean input RMS for optimizer compatibility.
    """

    if not gradients:
        raise ValueError("at least one gradient is required")
    rms_values = torch.stack(
        [
            gradient.detach().float().square().mean().sqrt()
            for gradient in gradients
        ]
    )
    active = rms_values > eps
    if not bool(active.any()):
        return torch.zeros_like(gradients[0]), rms_values.new_zeros(len(gradients))
    active_gradients = [
        gradient
        for gradient, keep in zip(gradients, active.tolist())
        if keep
    ]
    active_rms = rms_values[active]
    normalized = [
        gradient.float() / rms
        for gradient, rms in zip(active_gradients, active_rms)
    ]
    active_weights = minimum_norm_convex_weights(
        normalized,
        iterations=iterations,
        eps=eps,
    )
    common = sum(
        weight * gradient
        for weight, gradient in zip(active_weights, normalized)
    )
    common = common * active_rms.mean()
    weights = rms_values.new_zeros(len(gradients))
    weights[active] = active_weights
    return common.to(gradients[0].dtype), weights


def project_identity_gradient_to_protected_cone(
    identity_gradient: torch.Tensor,
    protected_gradients: list[torch.Tensor],
    *,
    iterations: int = 8,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Project an identity gradient so identity descent preserves constraints.

    An optimizer takes ``d = -g_id``. For every protected loss gradient h,
    first-order non-increase requires ``h^T d <= 0``, equivalently
    ``h^T g_id >= 0``. Alternating orthogonal projections enforce these
    half-space constraints without scalar loss weights.
    """

    if iterations < 1:
        raise ValueError("constraint projection iterations must be positive")
    valid = [
        gradient.detach().float()
        for gradient in protected_gradients
        if float(gradient.detach().float().square().sum().item()) > eps
    ]
    projected = identity_gradient.float()
    before = [
        float((projected.reshape(-1) @ gradient.reshape(-1)).item())
        for gradient in valid
    ]
    for _ in range(iterations):
        changed = False
        for gradient in valid:
            dot = projected.reshape(-1) @ gradient.reshape(-1)
            if float(dot.item()) < 0.0:
                projected = projected - (
                    dot / gradient.square().sum().clamp_min(eps)
                ) * gradient
                changed = True
        if not changed:
            break
    after = [
        float((projected.reshape(-1) @ gradient.reshape(-1)).item())
        for gradient in valid
    ]
    return projected.to(identity_gradient.dtype), {
        "constraint_dot_before": before,
        "constraint_dot_after": after,
        "retained_rms_ratio": (
            tensor_rms(projected) / max(tensor_rms(identity_gradient), eps)
        ),
    }


def positive_local_alignment_projection(
    source_gradient: torch.Tensor,
    identity_gradient: torch.Tensor,
    *,
    max_coefficient: float = 2.0,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Keep only locally identity-supported components of a score gradient.

    For BCHW latents, each spatial C-vector of the teacher-score gradient is
    least-squares projected onto the corresponding Pareto identity gradient.
    Negative coefficients are removed because the two objectives disagree;
    orthogonal score components are removed because they carry no validated
    identity evidence.
    """

    if source_gradient.shape != identity_gradient.shape:
        raise ValueError("source and identity gradients must share a shape")
    if source_gradient.ndim != 4:
        raise ValueError("local alignment projection expects BCHW gradients")
    if max_coefficient <= 0.0:
        raise ValueError("alignment coefficient cap must be positive")
    source = source_gradient.float()
    identity = identity_gradient.float()
    numerator = (source * identity).sum(dim=1, keepdim=True)
    denominator = identity.square().sum(dim=1, keepdim=True).clamp_min(eps)
    raw_coefficient = numerator / denominator
    coefficient = raw_coefficient.clamp(0.0, float(max_coefficient))
    projected = coefficient * identity
    source_rms = source.square().mean().sqrt()
    projected_rms = projected.square().mean().sqrt()
    return projected.to(source_gradient.dtype), {
        "positive_support_fraction": float(
            (raw_coefficient > 0.0).float().mean().item()
        ),
        "coefficient_mean": float(coefficient.mean().item()),
        "coefficient_max": float(coefficient.max().item()),
        "retained_score_rms_ratio": float(
            (projected_rms / source_rms.clamp_min(eps)).item()
        ),
        "global_cosine": tensor_cosine(source, identity),
    }


def parse_normalized_box(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    fields = [item.strip() for item in value.split(",")]
    if len(fields) != 4:
        raise ValueError("--identity-roi must contain x0,y0,x1,y1")
    box = tuple(float(item) for item in fields)
    x0, y0, x1, y1 = box
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError(
            "--identity-roi coordinates must satisfy "
            "0 <= x0 < x1 <= 1 and 0 <= y0 < y1 <= 1"
        )
    return box


def crop_normalized_roi(
    pixels: torch.Tensor,
    roi: tuple[float, float, float, float] | None,
) -> torch.Tensor:
    if roi is None:
        return pixels
    _, _, height, width = pixels.shape
    x0, y0, x1, y1 = roi
    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    return pixels[..., top:bottom, left:right]


def outside_roi_l1_loss(
    pixels: torch.Tensor,
    anchor: torch.Tensor,
    roi: tuple[float, float, float, float] | None,
) -> torch.Tensor:
    if roi is None:
        return pixels.new_zeros(())
    _, _, height, width = pixels.shape
    x0, y0, x1, y1 = roi
    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    mask = pixels.new_ones((1, 1, height, width))
    mask[..., top:bottom, left:right] = 0.0
    absolute_error = (pixels.float() - anchor.float()).abs()
    denominator = (mask.sum() * pixels.shape[1]).clamp_min(1.0)
    return (absolute_error * mask).sum() / denominator


def roi_weight_map(
    value: torch.Tensor,
    roi: tuple[float, float, float, float] | None,
    *,
    outside_weight: float,
) -> torch.Tensor:
    """Create a BCHW-broadcastable map with unit weight inside the ROI."""

    if not 0.0 <= outside_weight <= 1.0:
        raise ValueError("outside weight must be in [0, 1]")
    _, _, height, width = value.shape
    weight = value.new_full((1, 1, height, width), float(outside_weight))
    if roi is None:
        weight.fill_(1.0)
        return weight
    x0, y0, x1, y1 = roi
    left = max(0, min(width - 1, int(round(x0 * width))))
    top = max(0, min(height - 1, int(round(y0 * height))))
    right = max(left + 1, min(width, int(round(x1 * width))))
    bottom = max(top + 1, min(height, int(round(y1 * height))))
    weight[..., top:bottom, left:right] = 1.0
    return weight


def multiscale_pseudo_target_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    roi: tuple[float, float, float, float] | None = None,
    outside_weight: float = 1.0,
    lowpass_kernel: int = 1,
    lowpass_weight: float = 0.0,
) -> torch.Tensor:
    """ROI-weighted native/structural MSE for detached score pseudo-targets."""

    if prediction.shape != target.shape or prediction.ndim != 4:
        raise ValueError("prediction and target must share BCHW shape")
    if lowpass_kernel < 1:
        raise ValueError("lowpass kernel must be positive")
    if not 0.0 <= lowpass_weight <= 1.0:
        raise ValueError("lowpass weight must be in [0, 1]")

    def weighted_half_mse(
        left: torch.Tensor,
        right: torch.Tensor,
        weight: torch.Tensor,
    ) -> torch.Tensor:
        squared_error = (left.float() - right.float()).square()
        denominator = (
            weight.sum() * left.shape[0] * left.shape[1]
        ).clamp_min(1.0)
        return 0.5 * (squared_error * weight).sum() / denominator

    native_weight = roi_weight_map(
        prediction,
        roi,
        outside_weight=outside_weight,
    )
    native_loss = weighted_half_mse(prediction, target, native_weight)
    if lowpass_weight == 0.0 or lowpass_kernel == 1:
        return native_loss

    pooled_prediction = F.avg_pool2d(
        prediction.float(),
        kernel_size=lowpass_kernel,
        stride=lowpass_kernel,
    )
    pooled_target = F.avg_pool2d(
        target.float(),
        kernel_size=lowpass_kernel,
        stride=lowpass_kernel,
    )
    pooled_weight = F.interpolate(
        native_weight,
        size=pooled_prediction.shape[-2:],
        mode="area",
    )
    structural_loss = weighted_half_mse(
        pooled_prediction,
        pooled_target,
        pooled_weight,
    )
    return (
        (1.0 - lowpass_weight) * native_loss
        + lowpass_weight * structural_loss
    )


def ccip_preprocess_pixels(pixels_pm1: torch.Tensor) -> torch.Tensor:
    """Differentiable CCIP preprocessing for pixels in ``[-1, 1]``."""

    mean = pixels_pm1.new_tensor(
        [0.48145466, 0.4578275, 0.40821073]
    ).view(1, 3, 1, 1)
    std = pixels_pm1.new_tensor(
        [0.26862954, 0.26130258, 0.27577711]
    ).view(1, 3, 1, 1)
    pixels_01 = ((pixels_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)
    resized = F.interpolate(
        pixels_01,
        size=(384, 384),
        mode="bilinear",
        align_corners=False,
    )
    return ((resized - mean) / std).float()


def ccip_embed(ccip_model: torch.nn.Module, pixels_pm1: torch.Tensor) -> torch.Tensor:
    output = ccip_model(ccip_preprocess_pixels(pixels_pm1))
    features = output[0] if isinstance(output, (tuple, list)) else output
    return F.normalize(features.float(), dim=-1)


def dino_features(
    dino_model: torch.nn.Module,
    pixels_pm1: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable ImageNet preprocessing and DINOv2 CLS/patch features."""

    mean = pixels_pm1.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = pixels_pm1.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    pixels_01 = ((pixels_pm1 + 1.0) / 2.0).clamp(0.0, 1.0)
    resized = F.interpolate(
        pixels_01,
        size=(224, 224),
        mode="bicubic",
        align_corners=False,
    )
    normalized = (resized - mean) / std
    output = dino_model(pixel_values=normalized.to(dino_model.dtype))
    cls_features = (
        output.pooler_output
        if getattr(output, "pooler_output", None) is not None
        else output.last_hidden_state[:, 0]
    )
    patch_features = output.last_hidden_state[:, 1:]
    return (
        F.normalize(cls_features.float(), dim=-1),
        F.normalize(patch_features.float(), dim=-1),
    )


def dino_embed(dino_model: torch.nn.Module, pixels_pm1: torch.Tensor) -> torch.Tensor:
    """Differentiable DINOv2 CLS embedding retained for the global objective."""

    return dino_features(dino_model, pixels_pm1)[0]


def dino_patch_embed(
    dino_model: torch.nn.Module,
    pixels_pm1: torch.Tensor,
) -> torch.Tensor:
    """Return normalized spatial DINOv2 patch tokens."""

    return dino_features(dino_model, pixels_pm1)[1]


def load_reference_pixels(
    reference_paths: list[Path],
    *,
    size: int,
    device: str,
) -> torch.Tensor:
    values = []
    for path in reference_paths:
        image = Image.open(path).convert("RGB").resize(
            (size, size),
            Image.Resampling.LANCZOS,
        )
        array = np.asarray(image).copy()
        values.append(
            torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0
        )
    return torch.stack(values, dim=0).to(device)


def load_dino_reference_target(
    dino_model: torch.nn.Module,
    reference_paths: list[Path],
    *,
    device: str,
) -> torch.Tensor:
    reference_pixels = load_reference_pixels(
        reference_paths,
        size=384,
        device=device,
    )
    with torch.no_grad():
        target = dino_embed(dino_model, reference_pixels).mean(dim=0, keepdim=True)
        return F.normalize(target, dim=-1)


def load_dino_reference_patch_target(
    dino_model: torch.nn.Module,
    reference_paths: list[Path],
    *,
    device: str,
) -> torch.Tensor:
    """Build a detached patch-token bank from all identity references."""

    reference_pixels = load_reference_pixels(
        reference_paths,
        size=384,
        device=device,
    )
    with torch.no_grad():
        patches = dino_patch_embed(dino_model, reference_pixels)
        return patches.reshape(-1, patches.shape[-1]).detach()


def load_ccip_reference_target(
    ccip_model: torch.nn.Module,
    reference_paths: list[Path],
    *,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    """Average reference identity after filling the model's fixed batch."""

    if batch_size < 1:
        raise ValueError("identity model batch size must be positive")
    reference_pixels = []
    for index in range(batch_size):
        path = reference_paths[index % len(reference_paths)]
        image = Image.open(path).convert("RGB").resize(
            (384, 384),
            Image.Resampling.LANCZOS,
        )
        array = np.asarray(image).copy()
        pixels = torch.from_numpy(array).permute(2, 0, 1).float() / 127.5 - 1.0
        reference_pixels.append(pixels)
    batch = torch.stack(reference_pixels, dim=0).to(device)
    with torch.no_grad():
        target = ccip_embed(ccip_model, batch).mean(dim=0, keepdim=True)
        return F.normalize(target, dim=-1)


def decode_x0_to_pixels(vae, x0_prediction: torch.Tensor) -> torch.Tensor:
    """Decode an x0 latent without detaching its gradient path."""

    value = x0_prediction.to(vae.dtype)
    value = value / vae.config.scaling_factor + vae.config.shift_factor
    return vae.decode(value, return_dict=False)[0].float().clamp(-1.0, 1.0)


def decoded_identity_loss(
    ccip_model: torch.nn.Module,
    vae,
    x0_prediction: torch.Tensor,
    reference_target: torch.Tensor,
    *,
    fixed_batch_size: int,
    roi: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """CCIP cosine distance from decoded x0 to the mean reference identity."""

    pixels = decode_x0_to_pixels(vae, x0_prediction)
    return identity_loss_from_pixels(
        ccip_model,
        pixels,
        reference_target,
        fixed_batch_size=fixed_batch_size,
        roi=roi,
    )


def identity_loss_from_pixels(
    ccip_model: torch.nn.Module,
    pixels: torch.Tensor,
    reference_target: torch.Tensor,
    *,
    fixed_batch_size: int,
    roi: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    """CCIP cosine distance for an already decoded generated image."""

    if pixels.shape[0] != 1:
        raise ValueError(
            "identity-aware TSD currently expects one generated sample, found "
            f"batch={pixels.shape[0]}"
        )
    identity_pixels = crop_normalized_roi(pixels, roi)
    repeated = identity_pixels.repeat(fixed_batch_size, 1, 1, 1)
    prediction = ccip_embed(ccip_model, repeated)
    target = reference_target.expand_as(prediction)
    return (1.0 - (prediction * target).sum(dim=-1)).mean()


def dino_identity_loss_from_pixels(
    dino_model: torch.nn.Module,
    pixels: torch.Tensor,
    reference_target: torch.Tensor,
    *,
    roi: tuple[float, float, float, float] | None = None,
) -> torch.Tensor:
    identity_pixels = crop_normalized_roi(pixels, roi)
    prediction = dino_embed(dino_model, identity_pixels)
    target = reference_target.expand_as(prediction)
    return (1.0 - (prediction * target).sum(dim=-1)).mean()


def bidirectional_soft_chamfer_cosine(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    """Spatially match two token sets with a smooth bidirectional nearest-neighbor loss.

    The log-mean-exp normalization keeps the soft minimum on the same scale when
    the reference bank contains more images.
    """

    if temperature <= 0.0:
        raise ValueError("patch temperature must be positive")
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("patch features must have shape [tokens, channels]")
    prediction = F.normalize(prediction.float(), dim=-1)
    target = F.normalize(target.float(), dim=-1)
    distances = 1.0 - prediction @ target.transpose(0, 1)

    def soft_min(values: torch.Tensor, dim: int) -> torch.Tensor:
        count = values.shape[dim]
        return -temperature * (
            torch.logsumexp(-values / temperature, dim=dim)
            - math.log(count)
        )

    prediction_to_target = soft_min(distances, dim=1).mean()
    target_to_prediction = soft_min(distances, dim=0).mean()
    return 0.5 * (prediction_to_target + target_to_prediction)


def dino_patch_identity_loss_from_pixels(
    dino_model: torch.nn.Module,
    pixels: torch.Tensor,
    reference_patch_target: torch.Tensor,
    *,
    roi: tuple[float, float, float, float] | None = None,
    temperature: float = 0.05,
) -> torch.Tensor:
    """DINOv2 spatial patch loss for an already decoded generated image."""

    identity_pixels = crop_normalized_roi(pixels, roi)
    prediction = dino_patch_embed(dino_model, identity_pixels)
    prediction = prediction.reshape(-1, prediction.shape[-1])
    return bidirectional_soft_chamfer_cosine(
        prediction,
        reference_patch_target,
        temperature=temperature,
    )


def lookahead_identity_velocity_correction(
    identity_x0_gradient: torch.Tensor,
    sigma: torch.Tensor | float,
    next_sigma: torch.Tensor | float,
    *,
    next_latent_rms: float,
) -> torch.Tensor:
    """Convert an x0 loss gradient into an exact-RMS next-state correction."""

    if next_latent_rms < 0.0:
        raise ValueError("next latent RMS must be non-negative")
    sigma_value = torch.as_tensor(
        sigma,
        device=identity_x0_gradient.device,
        dtype=torch.float32,
    )
    next_sigma_value = torch.as_tensor(
        next_sigma,
        device=identity_x0_gradient.device,
        dtype=torch.float32,
    )
    sigma_span = sigma_value - next_sigma_value
    if float(sigma_span.detach().item()) <= 0.0:
        raise ValueError("look-ahead correction requires sigma > next_sigma")
    return (
        float(next_latent_rms)
        / sigma_span
        * rms_normalize_per_sample(identity_x0_gradient)
    )


def visual_identity_contrast(
    positive_residual: torch.Tensor,
    negative_residual: torch.Tensor,
    mode: str,
    projection_strength: float = 1.0,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float]:
    """Remove the generic reference response from a target-ID response.

    ``subtract`` implements the exact difference-in-differences estimator
    ``(p_pos - p_base) - (p_neg - p_base)``. ``project`` estimates the generic
    reference coefficient by least squares and removes only the parallel
    component, leaving the orthogonal target-specific residual.
    """

    if mode == "visual_identity_subtract":
        return positive_residual - negative_residual, 1.0
    if mode != "visual_identity_project":
        raise ValueError(f"not a visual identity contrast: {mode}")
    positive_flat = positive_residual.detach().float().reshape(-1)
    negative_flat = negative_residual.detach().float().reshape(-1)
    coefficient = (positive_flat @ negative_flat) / negative_flat.square().sum().clamp_min(eps)
    coefficient = coefficient.clamp(0.0, 2.0)
    removed_coefficient = float(projection_strength) * coefficient
    return (
        positive_residual - removed_coefficient * negative_residual,
        float(removed_coefficient.item()),
    )


def gated_visual_identity_contrast(
    positive_residual: torch.Tensor,
    negative_residual: torch.Tensor,
    *,
    shared_keep: float,
    support_quantile: float,
    support_gate_power: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Restore shared reference structure only on target-specific support.

    The fully projected residual supplies a spatial evidence map. A fraction
    of the removed parallel component is restored only where that evidence is
    present, preventing global reference layout from re-entering the update.
    """

    positive_flat = positive_residual.detach().float().reshape(-1)
    negative_flat = negative_residual.detach().float().reshape(-1)
    beta = (positive_flat @ negative_flat) / negative_flat.square().sum().clamp_min(eps)
    beta = beta.clamp(0.0, 2.0)
    specific = positive_residual - beta * negative_residual
    reduce_dims = tuple(range(specific.ndim - 2))
    support = specific.detach().float().square().mean(dim=reduce_dims, keepdim=True).sqrt()
    if support.ndim == 4:
        support = torch.nn.functional.avg_pool2d(
            support,
            kernel_size=5,
            stride=1,
            padding=2,
        )
    threshold = torch.quantile(
        support.reshape(-1),
        float(support_quantile),
    ).clamp_min(eps)
    gate = (support / threshold).clamp(0.0, 1.0).pow(float(support_gate_power))
    restored_common = float(shared_keep) * gate * beta * negative_residual
    return specific + restored_common, float(beta.item()), gate


def subject_support_visual_identity_contrast(
    positive_residual: torch.Tensor,
    negative_residual: torch.Tensor,
    subject_residual: torch.Tensor,
    *,
    shared_keep: float,
    outside_keep: float,
    support_quantile: float,
    support_gate_power: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Gate identity transfer with reference-independent base semantics.

    ``subject_residual`` is the frozen base prediction difference between the
    full prompt and a matched background-only prompt. It therefore localizes
    the subject on the current trajectory without seeing any reference image.
    Inside that support we interpolate from the projected identity residual
    toward the raw positive teacher residual. Outside it we retain only a
    controlled fraction of the projected residual:

        d_in = d_projected + shared_keep * (r_pos - d_projected)
        d = gate * d_in + outside_keep * (1 - gate) * d_projected
    """

    projected, beta = visual_identity_contrast(
        positive_residual,
        negative_residual,
        "visual_identity_project",
        eps=eps,
    )
    reduce_dims = tuple(range(subject_residual.ndim - 2))
    support = (
        subject_residual.detach().float().square().mean(dim=reduce_dims, keepdim=True).sqrt()
    )
    if support.ndim == 4:
        support = torch.nn.functional.avg_pool2d(
            support,
            kernel_size=5,
            stride=1,
            padding=2,
        )
    threshold = torch.quantile(
        support.reshape(-1),
        float(support_quantile),
    ).clamp_min(eps)
    gate = (support / threshold).clamp(0.0, 1.0).pow(float(support_gate_power))
    inside_direction = projected + float(shared_keep) * (
        positive_residual - projected
    )
    direction = gate * inside_direction + float(outside_keep) * (1.0 - gate) * projected
    return direction, beta, gate


def token_support_visual_identity_contrast(
    positive_residual: torch.Tensor,
    negative_residual: torch.Tensor,
    token_support: torch.Tensor,
    *,
    shared_keep: float,
    outside_keep: float,
    support_quantile: float,
    support_gate_power: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, float, torch.Tensor]:
    """Gate identity transfer with base attention to a subject prompt token."""

    projected, beta = visual_identity_contrast(
        positive_residual,
        negative_residual,
        "visual_identity_project",
        eps=eps,
    )
    support = token_support.detach().float().clamp_min(0.0)
    if support.ndim == 4:
        support = torch.nn.functional.avg_pool2d(
            support,
            kernel_size=5,
            stride=1,
            padding=2,
        )
    threshold = torch.quantile(
        support.reshape(-1),
        float(support_quantile),
    ).clamp_min(eps)
    gate = (support / threshold).clamp(0.0, 1.0).pow(float(support_gate_power))
    inside_direction = projected + float(shared_keep) * (
        positive_residual - projected
    )
    direction = gate * inside_direction + float(outside_keep) * (1.0 - gate) * projected
    return direction, beta, gate


def low_rank_nuisance_identity_contrast(
    positive_residuals: list[torch.Tensor],
    negative_residuals: list[torch.Tensor],
    *,
    rank: int,
    ridge: float,
    projection_strength: float,
    consensus_rescue: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Separate target identity from a low-rank reference nuisance subspace.

    The nuisance samples contain the generic mismatched-reference mean,
    variation across mismatched references, and variation across positive
    references. The target positive mean is spectrally projected away from
    that subspace. Removed components are restored only where positive
    references agree after negative centering and are discriminative relative
    to the positive/negative response magnitudes.
    """

    if len(positive_residuals) < 2:
        raise ValueError("visual_identity_subspace requires at least two positive references")
    if len(negative_residuals) < 2:
        raise ValueError("visual_identity_subspace requires at least two negative references")
    if rank < 1:
        raise ValueError("subspace rank must be positive")
    if ridge < 0.0:
        raise ValueError("subspace ridge must be non-negative")

    positives = torch.stack(
        [value.detach().float() for value in positive_residuals],
        dim=0,
    )
    negatives = torch.stack(
        [value.detach().float() for value in negative_residuals],
        dim=0,
    )
    positive_mean = positives.mean(dim=0)
    negative_mean = negatives.mean(dim=0)

    nuisance_candidates = [negative_mean]
    nuisance_candidates.extend(value for value in negatives - negative_mean)
    nuisance_candidates.extend(value for value in positives - positive_mean)

    normalized_rows = []
    element_count = positive_mean.numel()
    for candidate in nuisance_candidates:
        candidate_flat = candidate.reshape(-1)
        candidate_rms = candidate_flat.square().mean().sqrt()
        if float(candidate_rms.item()) <= eps:
            continue
        normalized_rows.append(candidate_flat / candidate_rms / (element_count ** 0.5))
    if not normalized_rows:
        return positive_mean, {
            "effective_rank": 0,
            "removed_fraction": 0.0,
            "consensus_gate_mean": 0.0,
            "rescue_fraction": 0.0,
            "eigenvalues": [],
        }

    nuisance_rows = torch.stack(normalized_rows, dim=0)
    gram = nuisance_rows @ nuisance_rows.transpose(0, 1)
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    effective_rank = min(
        int(rank),
        int((eigenvalues > eps).sum().item()),
    )
    if effective_rank == 0:
        return positive_mean, {
            "effective_rank": 0,
            "removed_fraction": 0.0,
            "consensus_gate_mean": 0.0,
            "rescue_fraction": 0.0,
            "eigenvalues": [],
        }

    selected_values = eigenvalues[:effective_rank]
    selected_vectors = eigenvectors[:, :effective_rank]
    basis = (
        selected_vectors.transpose(0, 1) @ nuisance_rows
    ) / selected_values.sqrt().unsqueeze(1).clamp_min(eps)
    positive_flat = positive_mean.reshape(-1)
    coordinates = basis @ positive_flat
    spectral_weights = selected_values / (selected_values + float(ridge))
    removed_flat = basis.transpose(0, 1) @ (spectral_weights * coordinates)
    removed = removed_flat.reshape_as(positive_mean) * float(projection_strength)

    centered_positives = positives - negative_mean.unsqueeze(0)
    centered_mean = centered_positives.mean(dim=0)
    consensus = centered_mean.abs() / (
        centered_positives.square().mean(dim=0).sqrt().clamp_min(eps)
    )
    discriminative = centered_mean.abs() / (
        positive_mean.abs() + negative_mean.abs() + eps
    )
    rescue_gate = consensus.clamp(0.0, 1.0) * discriminative.clamp(0.0, 1.0)
    rescued = float(consensus_rescue) * rescue_gate * removed
    direction = positive_mean - removed + rescued

    positive_rms = positive_mean.square().mean().sqrt().clamp_min(eps)
    removed_rms = removed.square().mean().sqrt()
    rescued_rms = rescued.square().mean().sqrt()
    return direction, {
        "effective_rank": effective_rank,
        "removed_fraction": float((removed_rms / positive_rms).item()),
        "consensus_gate_mean": float(rescue_gate.mean().item()),
        "rescue_fraction": float((rescued_rms / positive_rms).item()),
        "eigenvalues": [
            float(value.item()) for value in selected_values
        ],
    }


def project_reference_value_stores(
    positive_stores: list[dict[int, tuple[torch.Tensor, torch.Tensor]]],
    negative_stores: list[dict[int, tuple[torch.Tensor, torch.Tensor]]],
    *,
    projection_strength: float,
    eps: float = 1e-8,
) -> tuple[
    dict[int, tuple[torch.Tensor, torch.Tensor]],
    dict[str, float],
]:
    """Project mismatched-reference content out of transient teacher values.

    Positive keys are retained for target-reference content addressing. Only
    values are projected, before the frozen DiT mixes reference and target
    tokens into an output score.
    """

    if not positive_stores or not negative_stores:
        raise ValueError("positive and negative K/V stores are required")
    layer_ids = set(positive_stores[0])
    if not layer_ids:
        raise ValueError("captured positive K/V store is empty")
    for store in [*positive_stores, *negative_stores]:
        if set(store) != layer_ids:
            raise ValueError("all captured K/V stores must contain the same layers")

    projected_store: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    betas = []
    retained_ratios = []
    for layer_id in sorted(layer_ids):
        positive_keys = torch.stack(
            [store[layer_id][0].detach().float() for store in positive_stores],
            dim=0,
        ).mean(dim=0)
        positive_values = torch.stack(
            [store[layer_id][1].detach().float() for store in positive_stores],
            dim=0,
        ).mean(dim=0)
        negative_values = torch.stack(
            [store[layer_id][1].detach().float() for store in negative_stores],
            dim=0,
        ).mean(dim=0)
        positive_flat = positive_values.reshape(-1)
        negative_flat = negative_values.reshape(-1)
        beta = (
            (positive_flat @ negative_flat)
            / negative_flat.square().sum().clamp_min(eps)
        ).clamp(0.0, 2.0)
        filtered_values = (
            positive_values
            - float(projection_strength) * beta * negative_values
        )
        positive_rms = positive_values.square().mean().sqrt().clamp_min(eps)
        filtered_rms = filtered_values.square().mean().sqrt()
        projected_store[layer_id] = (
            positive_keys.to(positive_stores[0][layer_id][0].dtype),
            filtered_values.to(positive_stores[0][layer_id][1].dtype),
        )
        betas.append(float(beta.item()))
        retained_ratios.append(float((filtered_rms / positive_rms).item()))
    return projected_store, {
        "beta_mean": sum(betas) / len(betas),
        "beta_max": max(betas),
        "value_retained_rms_ratio_mean": sum(retained_ratios) / len(retained_ratios),
    }


def parse_layer_indices(specification: str) -> set[int]:
    indices: set[int] = set()
    for item in specification.split(","):
        item = item.strip()
        if not item:
            continue
        if "-" in item:
            first, last = item.split("-", 1)
            start, end = int(first), int(last)
            if end < start:
                raise ValueError(f"invalid layer range: {item}")
            indices.update(range(start, end + 1))
        else:
            indices.add(int(item))
    if not indices:
        raise ValueError("--support-layers must select at least one layer")
    return indices


def resolve_prompt_token_indices(tokenizer, prompt: str, phrase: str) -> list[int]:
    wrapped = (
        "<|im_start|>user\n"
        + prompt
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    start = wrapped.lower().find(phrase.lower())
    if start < 0:
        raise ValueError(f"--subject-token {phrase!r} is not present in --prompt")
    end = start + len(phrase)
    tokenized = tokenizer(
        [wrapped],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    offsets = tokenized.offset_mapping[0].tolist()
    mask = tokenized.attention_mask[0].bool().tolist()
    indices = [
        index
        for index, ((token_start, token_end), keep) in enumerate(zip(offsets, mask))
        if keep and token_start < end and token_end > start
    ]
    if not indices:
        raise ValueError(f"could not map --subject-token {phrase!r} to prompt tokens")
    return indices


def main() -> int:
    args = parse_args()
    if args.inner_iters < 1:
        raise ValueError("--inner-iters must be positive")
    if args.identity_loss_weight < 0.0:
        raise ValueError("--identity-loss-weight must be non-negative")
    if args.identity_model_batch_size < 1:
        raise ValueError("--identity-model-batch-size must be positive")
    identity_roi = parse_normalized_box(args.identity_roi)
    distill_roi = parse_normalized_box(args.distill_roi)
    if not 0.0 <= args.distill_outside_weight <= 1.0:
        raise ValueError("--distill-outside-weight must be in [0, 1]")
    if args.distill_lowpass_kernel < 1:
        raise ValueError("--distill-lowpass-kernel must be positive")
    if not 0.0 <= args.distill_lowpass_weight <= 1.0:
        raise ValueError("--distill-lowpass-weight must be in [0, 1]")
    if args.identity_dino_weight < 0.0:
        raise ValueError("--identity-dino-weight must be non-negative")
    if args.identity_dino_patch_weight < 0.0:
        raise ValueError("--identity-dino-patch-weight must be non-negative")
    if args.identity_dino_patch_temperature <= 0.0:
        raise ValueError("--identity-dino-patch-temperature must be positive")
    if args.identity_pareto_iters < 1:
        raise ValueError("--identity-pareto-iters must be positive")
    if args.identity_constraint_iters < 1:
        raise ValueError("--identity-constraint-iters must be positive")
    if args.identity_alignment_max_coefficient <= 0.0:
        raise ValueError("--identity-alignment-max-coefficient must be positive")
    if not 0.0 < args.identity_sigma_max <= 1.0:
        raise ValueError("--identity-sigma-max must be in (0, 1]")
    if args.identity_trajectory_rms < 0.0:
        raise ValueError("--identity-trajectory-rms must be non-negative")
    if args.identity_terminal_iters < 0:
        raise ValueError("--identity-terminal-iters must be non-negative")
    if args.identity_terminal_lr is not None and args.identity_terminal_lr <= 0.0:
        raise ValueError("--identity-terminal-lr must be positive")
    if args.identity_terminal_pixel_weight < 0.0:
        raise ValueError("--identity-terminal-pixel-weight must be non-negative")
    if args.identity_terminal_outside_pixel_weight < 0.0:
        raise ValueError(
            "--identity-terminal-outside-pixel-weight must be non-negative"
        )
    if args.identity_terminal_outside_pixel_weight > 0.0 and identity_roi is None:
        raise ValueError(
            "--identity-terminal-outside-pixel-weight requires --identity-roi"
        )
    if args.identity_terminal_iters > 0 and args.identity_loss_weight <= 0.0:
        raise ValueError(
            "--identity-terminal-iters requires a positive --identity-loss-weight"
        )
    if not 0.0 <= args.projection_strength <= 1.0:
        raise ValueError("--projection-strength must be in [0, 1]")
    if not 0.0 <= args.shared_keep <= 1.0:
        raise ValueError("--shared-keep must be in [0, 1]")
    if not 0.0 <= args.support_quantile <= 1.0:
        raise ValueError("--support-quantile must be in [0, 1]")
    if args.support_gate_power <= 0.0:
        raise ValueError("--support-gate-power must be positive")
    if not 0.0 <= args.outside_keep <= 1.0:
        raise ValueError("--outside-keep must be in [0, 1]")
    if args.subspace_rank < 1:
        raise ValueError("--subspace-rank must be positive")
    if args.subspace_ridge < 0.0:
        raise ValueError("--subspace-ridge must be non-negative")
    if not 0.0 <= args.subspace_strength <= 1.0:
        raise ValueError("--subspace-strength must be in [0, 1]")
    if not 0.0 <= args.consensus_rescue <= 1.0:
        raise ValueError("--consensus-rescue must be in [0, 1]")
    if not 0.0 <= args.kv_value_projection_strength <= 1.0:
        raise ValueError("--kv-value-projection-strength must be in [0, 1]")
    if args.residual_contrast == "visual_identity_subject_gate" and not args.background_prompt:
        raise ValueError(
            "--residual-contrast visual_identity_subject_gate requires --background-prompt"
        )
    if (
        args.teacher_backend == "native_omni"
        and args.residual_contrast == "visual_identity_kv_value_project"
    ):
        raise ValueError(
            "visual_identity_kv_value_project requires --teacher-backend legacy_kv"
        )
    support_layers = parse_layer_indices(args.support_layers)

    os.environ.setdefault("ZBASE", "models/Z-Image")
    os.environ.setdefault("REF", "/home/ubuntu/smoke_anime/revref_30k/img")
    impl = load_impl()
    kv = impl.load_base_probe()

    reference_paths = [
        Path(item.strip()).expanduser().resolve()
        for item in args.references.split(",")
        if item.strip()
    ]
    if not reference_paths:
        raise ValueError("at least one reference is required")
    for path in reference_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    negative_reference_paths = [
        Path(item.strip()).expanduser().resolve()
        for item in (args.negative_references or "").split(",")
        if item.strip()
    ]
    visual_identity_mode = args.residual_contrast.startswith("visual_identity_")
    if visual_identity_mode and not negative_reference_paths:
        raise ValueError(
            f"--residual-contrast {args.residual_contrast} requires --negative-references"
        )
    for path in negative_reference_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    capsule_path = (
        Path(args.identity_capsule_file).expanduser().resolve()
        if args.identity_capsule_file
        else None
    )
    capsule = None
    if capsule_path is not None:
        capsule_payload = json.loads(capsule_path.read_text(encoding="utf-8"))
        capsule = str(capsule_payload["capsule"]).strip()
    prompt = f"{args.prompt}, subject identity: {capsule}" if capsule else args.prompt
    prompt_without_identity = args.prompt
    if args.residual_contrast == "identity_interaction" and not capsule:
        raise ValueError(
            "--residual-contrast identity_interaction requires --identity-capsule-file; "
            "use teacher_base for the no-capsule ablation"
        )
    negative = args.negative or kv.NEG

    with torch.no_grad():
        reference_latents = [
            impl.encode_reference_image(
                kv,
                impl.Image.open(path).convert("RGB").resize(
                    (args.ref_size, args.ref_size),
                    impl.Image.Resampling.LANCZOS,
                ),
            )
            for path in reference_paths
        ]
        negative_reference_latents = [
            impl.encode_reference_image(
                kv,
                impl.Image.open(path).convert("RGB").resize(
                    (args.ref_size, args.ref_size),
                    impl.Image.Resampling.LANCZOS,
                ),
            )
            for path in negative_reference_paths
        ]
        cap_pos = kv.enc(prompt)
        cap_without_identity = (
            kv.enc(prompt_without_identity)
            if args.residual_contrast == "identity_interaction"
            else None
        )
        cap_neg = kv.enc(negative)
        omni_cap_pos = (
            encode_omni_prompt(kv.pipe, prompt, device=kv.DEV)
            if args.teacher_backend == "native_omni"
            else None
        )
        omni_cap_without_identity = (
            encode_omni_prompt(
                kv.pipe,
                prompt_without_identity,
                device=kv.DEV,
            )
            if args.teacher_backend == "native_omni"
            and args.residual_contrast == "identity_interaction"
            else None
        )
        omni_cap_neg = (
            encode_omni_prompt(kv.pipe, negative, device=kv.DEV)
            if args.teacher_backend == "native_omni"
            else None
        )
        if args.capture_prompt_mode == "target":
            cap_capture = cap_pos
        elif args.capture_prompt_mode == "without_identity":
            cap_capture = kv.enc(prompt_without_identity)
        else:
            cap_capture = kv.enc(kv.PROMPT)
        reference_caption = "anime illustration of a character"
        if args.reference_caption_mode == "identity" and capsule:
            reference_caption = (
                f"anime illustration of a character, subject identity: {capsule}"
            )
        cap_ref = kv.enc(reference_caption)
        cap_background = (
            kv.enc(args.background_prompt)
            if args.residual_contrast == "visual_identity_subject_gate"
            else None
        )
        subject_token_indices = (
            resolve_prompt_token_indices(kv.pipe.tokenizer, prompt, args.subject_token)
            if args.residual_contrast == "visual_identity_token_gate"
            else []
        )

    identity_enabled = args.identity_loss_weight > 0.0
    identity_model_path = Path(args.identity_model_path).expanduser().resolve()
    ccip_model = None
    ccip_reference_target = None
    dino_model = None
    dino_reference_target = None
    dino_reference_patch_target = None
    kv.pipe.text_encoder.to("cpu")
    if identity_enabled:
        if not identity_model_path.is_file():
            raise FileNotFoundError(identity_model_path)
        from onnx2torch import convert

        ccip_model = convert(str(identity_model_path)).to(kv.DEV).eval()
        ccip_model.requires_grad_(False)
        ccip_reference_target = load_ccip_reference_target(
            ccip_model,
            reference_paths,
            batch_size=args.identity_model_batch_size,
            device=kv.DEV,
        )
        if args.identity_dino_weight > 0.0 or args.identity_dino_patch_weight > 0.0:
            from transformers import AutoModel

            dino_model = AutoModel.from_pretrained(
                args.identity_dino_model,
                local_files_only=True,
            ).to(kv.DEV).eval()
            dino_model.requires_grad_(False)
            if args.identity_dino_weight > 0.0:
                dino_reference_target = load_dino_reference_target(
                    dino_model,
                    reference_paths,
                    device=kv.DEV,
                )
            if args.identity_dino_patch_weight > 0.0:
                dino_reference_patch_target = load_dino_reference_patch_target(
                    dino_model,
                    reference_paths,
                    device=kv.DEV,
                )
        kv.pipe.vae.to(kv.DEV).eval()
        kv.pipe.vae.requires_grad_(False)
    else:
        kv.pipe.vae.to("cpu")
    kv.tf.requires_grad_(False)
    torch.cuda.empty_cache()

    def identity_objectives_from_pixels(
        pixels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        ccip_loss = identity_loss_from_pixels(
            ccip_model,
            pixels,
            ccip_reference_target,
            fixed_batch_size=args.identity_model_batch_size,
            roi=identity_roi,
        )
        dino_cls_loss = pixels.new_zeros(())
        if dino_reference_target is not None:
            dino_cls_loss = dino_identity_loss_from_pixels(
                dino_model,
                pixels,
                dino_reference_target,
                roi=identity_roi,
            )
        dino_patch_loss = pixels.new_zeros(())
        if dino_reference_patch_target is not None:
            dino_patch_loss = dino_patch_identity_loss_from_pixels(
                dino_model,
                pixels,
                dino_reference_patch_target,
                roi=identity_roi,
                temperature=args.identity_dino_patch_temperature,
            )
        combined = (
            args.identity_loss_weight * ccip_loss
            + args.identity_dino_weight * dino_cls_loss
            + args.identity_dino_patch_weight * dino_patch_loss
        )
        return combined, ccip_loss, dino_cls_loss, dino_patch_loss

    generator = torch.Generator(kv.DEV).manual_seed(args.seed)
    latent_side = 2 * (args.size // 16)
    latents = torch.randn(
        (1, 16, latent_side, latent_side),
        generator=generator,
        device=kv.DEV,
        dtype=torch.float32,
    )

    scheduler = kv.pipe.scheduler
    image_seq_len = (latents.shape[2] // 2) * (latents.shape[3] // 2)
    mu = kv.calculate_shift(
        image_seq_len,
        scheduler.config.get("base_image_seq_len", 256),
        scheduler.config.get("max_image_seq_len", 4096),
        scheduler.config.get("base_shift", 0.5),
        scheduler.config.get("max_shift", 1.15),
    )
    scheduler.sigma_min = 0.0
    try:
        timesteps, _ = kv.retrieve_timesteps(scheduler, args.steps, kv.DEV, sigmas=None, mu=mu)
    except Exception:
        timesteps, _ = kv.retrieve_timesteps(scheduler, args.steps, kv.DEV, sigmas=None)

    kv.INJ.update(
        {
            "refpos": "strip",
            "route": "off",
            "vnorm": False,
            "ref_len": (args.ref_size // 16) ** 2,
            "tgt_len": (args.size // 16) ** 2,
            "w": float(args.teacher_weight),
            "measure": False,
            "ref_bias": 0.0,
            "support_capture": False,
            "support_token_indices": tuple(subject_token_indices),
            "support_layers": support_layers,
            "support_maps": [],
        }
    )

    def forward(cap, latent_value: torch.Tensor) -> torch.Tensor:
        x_image = latent_value.to(kv.DT).unsqueeze(2)[0]
        raw = kv.tf(
            x=[[x_image]],
            t=ts,
            cap_feats=[[cap]],
            siglip_feats=[None],
            image_noise_mask=[[1]],
            return_dict=False,
        )[0][0]
        return raw.float().clone()

    logs = []
    started = time.time()
    for outer_index, timestep in enumerate(timesteps):
        # The frozen base owns the reverse trajectory and advances exactly
        # once. Distillation is applied afterwards at the next scheduler state,
        # matching the source TSD ordering and preventing the same outer step
        # from immediately erasing the optimized latent.
        ts = (1000 - timestep.expand(latents.shape[0])) / 1000
        _, scheduler_sigma = impl.resolve_scheduler_sigma(scheduler, timestep, latents)
        with torch.no_grad():
            kv.INJ["mode"] = "off"
            kv.STORE.clear()
            cond = forward(cap_pos, latents)
            unc = forward(cap_neg, latents)
            scheduler_prediction = classifier_free_guided_prediction(
                cond,
                unc,
                args.cfg,
            )
            scheduler_velocity = prediction_to_velocity(scheduler_prediction)

        scheduler_index, scheduler_sigma = impl.resolve_scheduler_sigma(
            scheduler,
            timestep,
            latents,
        )
        current_progress = outer_index / max(1, len(timesteps) - 1)
        trajectory_identity_active = (
            identity_enabled
            and args.identity_trajectory_rms > 0.0
            and args.active_start <= current_progress <= args.active_end
            and float(scheduler_sigma.item()) <= args.identity_sigma_max
            and scheduler_index + 1 < len(scheduler.sigmas)
        )
        trajectory_identity_log = None
        if trajectory_identity_active:
            # First-order look-ahead: for an Euler flow step
            # z_next = z + (sigma_next - sigma) * v. Adding
            # rho / (sigma - sigma_next) * grad_x0 L to v therefore moves the
            # next scheduler state by exactly -rho * normalized_grad_x0 L.
            identity_x0 = (
                latents.detach()
                - scheduler_sigma.float() * scheduler_velocity.detach()
            ).requires_grad_(True)
            trajectory_pixels = decode_x0_to_pixels(kv.pipe.vae, identity_x0)
            (
                trajectory_loss,
                trajectory_ccip_loss,
                trajectory_dino_loss,
                trajectory_patch_loss,
            ) = identity_objectives_from_pixels(trajectory_pixels)
            trajectory_gradient = torch.autograd.grad(
                trajectory_loss,
                identity_x0,
            )[0]
            next_sigma = scheduler.sigmas[scheduler_index + 1].to(
                device=latents.device,
                dtype=torch.float32,
            )
            sigma_span = (scheduler_sigma - next_sigma).clamp_min(1e-6)
            velocity_correction = lookahead_identity_velocity_correction(
                trajectory_gradient,
                scheduler_sigma,
                next_sigma,
                next_latent_rms=args.identity_trajectory_rms,
            )
            scheduler_velocity = (
                scheduler_velocity + velocity_correction.detach()
            )
            trajectory_identity_log = {
                "loss": float(trajectory_loss.detach().item()),
                "identity_loss": float(trajectory_ccip_loss.detach().item()),
                "dino_identity_loss": (
                    float(trajectory_dino_loss.detach().item())
                    if dino_reference_target is not None
                    else None
                ),
                "dino_patch_identity_loss": (
                    float(trajectory_patch_loss.detach().item())
                    if dino_reference_patch_target is not None
                    else None
                ),
                "identity_x0_grad_rms": tensor_rms(trajectory_gradient),
                "velocity_correction_rms": tensor_rms(velocity_correction),
                "predicted_next_latent_correction_rms": (
                    args.identity_trajectory_rms
                ),
                "sigma_span": float(sigma_span.item()),
            }

        with torch.no_grad():
            latents = scheduler.step(
                scheduler_velocity.to(torch.float32),
                timestep,
                latents,
                return_dict=False,
            )[0]

        distill_index = outer_index + 1
        if distill_index < len(timesteps):
            distill_timestep = timesteps[distill_index]
            ts = (1000 - distill_timestep.expand(latents.shape[0])) / 1000
            _, sigma = impl.resolve_scheduler_sigma(
                scheduler,
                distill_timestep,
                latents,
            )
            progress = distill_index / max(1, len(timesteps) - 1)
            within_active_window = (
                args.active_start <= progress <= args.active_end
                and float(sigma.item()) > 1e-4
            )
            score_active = args.distillation and within_active_window
            identity_active = (
                identity_enabled
                and args.identity_trajectory_rms == 0.0
                and within_active_window
                and float(sigma.item()) <= args.identity_sigma_max
            )
            active = score_active or identity_active
        else:
            distill_timestep = None
            sigma = scheduler_sigma
            progress = 1.0
            score_active = False
            identity_active = False
            active = False

        inner_logs = []
        if active:
            content_anchor = latents.detach().clone()
            latents = latents.detach().requires_grad_(True)
            optimizer = torch.optim.AdamW([latents], lr=args.lr, weight_decay=0.0)
            local_progress = (progress - args.active_start) / max(
                args.active_end - args.active_start,
                1e-8,
            )
            taper_scale = max(0.0, 1.0 - local_progress) if args.taper else 1.0
            for inner_index in range(args.inner_iters):
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    kv.STORE.clear()
                    kv.INJ["mode"] = "off"
                    if args.residual_contrast == "visual_identity_token_gate":
                        kv.INJ["support_maps"].clear()
                        kv.INJ["support_capture"] = True
                    base_prediction = forward(cap_pos, latents)
                    base_unconditional_prediction = (
                        forward(cap_neg, latents)
                        if args.inner_x0_mode == "cfg"
                        or args.teacher_backend == "native_omni"
                        else None
                    )
                    base_guided_prediction = (
                        classifier_free_guided_prediction(
                            base_prediction,
                            base_unconditional_prediction,
                            args.cfg,
                        )
                        if base_unconditional_prediction is not None
                        else None
                    )
                    x0_base_prediction = (
                        base_guided_prediction
                        if args.inner_x0_mode == "cfg"
                        else base_prediction
                    )
                    kv.INJ["support_capture"] = False
                    token_support = None
                    if args.residual_contrast == "visual_identity_token_gate":
                        if not kv.INJ["support_maps"]:
                            raise RuntimeError("attention processor captured no subject support maps")
                        token_support_flat = torch.stack(
                            kv.INJ["support_maps"],
                            dim=0,
                        ).mean(dim=0)
                        token_side = int(round(token_support_flat.shape[-1] ** 0.5))
                        if token_side * token_side != token_support_flat.shape[-1]:
                            raise RuntimeError(
                                "target token count is not a square: "
                                f"{token_support_flat.shape[-1]}"
                            )
                        token_support = token_support_flat.reshape(
                            token_support_flat.shape[0],
                            1,
                            token_side,
                            token_side,
                        ).clamp_min(0.0)
                        token_support = torch.nn.functional.interpolate(
                            token_support,
                            size=base_prediction.shape[-2:],
                            mode="bilinear",
                            align_corners=False,
                        )
                    base_without_identity = (
                        forward(cap_without_identity, latents)
                        if cap_without_identity is not None
                        else None
                    )
                    base_without_identity_guided = (
                        classifier_free_guided_prediction(
                            base_without_identity,
                            base_unconditional_prediction,
                            args.cfg,
                        )
                        if base_without_identity is not None
                        and args.teacher_backend == "native_omni"
                        else base_without_identity
                    )
                    base_background = (
                        forward(cap_background, latents)
                        if cap_background is not None
                        else None
                    )
                    teacher_predictions = []
                    teacher_without_identity_predictions = []
                    x_inner = latents.to(kv.DT).unsqueeze(2)[0]

                    def query_reference_teachers(
                        query_reference_latents: list[torch.Tensor],
                        cap,
                        omni_cap=None,
                    ) -> list[torch.Tensor]:
                        predictions = []
                        for reference_latent in query_reference_latents:
                            if args.teacher_backend == "native_omni":
                                if omni_cap is None or omni_cap_neg is None:
                                    raise RuntimeError(
                                        "native Omni prompt embeddings were not initialized"
                                    )
                                kv.INJ["mode"] = "off"
                                positive_raw = kv.tf(
                                    x=[[reference_latent, x_inner]],
                                    t=ts,
                                    cap_feats=[omni_cap],
                                    siglip_feats=[None],
                                    image_noise_mask=[[0, 1]],
                                    return_dict=False,
                                )[0][0]
                                negative_raw = kv.tf(
                                    x=[[reference_latent, x_inner]],
                                    t=ts,
                                    cap_feats=[omni_cap_neg],
                                    siglip_feats=[None],
                                    image_noise_mask=[[0, 1]],
                                    return_dict=False,
                                )[0][0]
                                predictions.append(
                                    classifier_free_guided_prediction(
                                        positive_raw.float().clone(),
                                        negative_raw.float().clone(),
                                        args.cfg,
                                    )
                                )
                                continue
                            kv.STORE.clear()
                            kv.INJ["mode"] = "capture"
                            kv.tf(
                                x=[[reference_latent, x_inner]],
                                t=ts,
                                cap_feats=[[cap_ref, cap_capture]],
                                siglip_feats=[None],
                                image_noise_mask=[[0, 1]],
                                return_dict=False,
                            )
                            kv.INJ["mode"] = "inject"
                            predictions.append(forward(cap, latents))
                        return predictions

                    def capture_reference_store(
                        reference_latent: torch.Tensor,
                    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
                        kv.STORE.clear()
                        kv.INJ["mode"] = "capture"
                        kv.tf(
                            x=[[reference_latent, x_inner]],
                            t=ts,
                            cap_feats=[[cap_ref, cap_capture]],
                            siglip_feats=[None],
                            image_noise_mask=[[0, 1]],
                            return_dict=False,
                        )
                        return {
                            int(layer_id): (
                                key.detach().clone(),
                                value.detach().clone(),
                            )
                            for layer_id, (key, value) in kv.STORE.items()
                        }

                    def query_kv_value_projected_teacher(
                        cap,
                    ) -> tuple[torch.Tensor, dict[str, float]]:
                        positive_stores = [
                            capture_reference_store(reference_latent)
                            for reference_latent in reference_latents
                        ]
                        negative_stores = [
                            capture_reference_store(reference_latent)
                            for reference_latent in negative_reference_latents
                        ]
                        projected_store, diagnostics = project_reference_value_stores(
                            positive_stores,
                            negative_stores,
                            projection_strength=args.kv_value_projection_strength,
                        )
                        kv.STORE.clear()
                        kv.STORE.update(projected_store)
                        kv.INJ["mode"] = "inject"
                        prediction = forward(cap, latents)
                        return prediction, diagnostics

                    # Recapture at the current optimized latent. This keeps
                    # every teacher/base contrast on the same inner state.
                    kv_value_diagnostics = None
                    if args.residual_contrast == "visual_identity_kv_value_project":
                        if args.teacher_backend != "legacy_kv":
                            raise ValueError(
                                "visual_identity_kv_value_project is only defined "
                                "for --teacher-backend legacy_kv"
                            )
                        teacher_prediction, kv_value_diagnostics = (
                            query_kv_value_projected_teacher(cap_pos)
                        )
                        teacher_predictions = [teacher_prediction]
                    else:
                        teacher_predictions = query_reference_teachers(
                            reference_latents,
                            cap_pos,
                            omni_cap_pos,
                        )
                    if cap_without_identity is not None:
                        teacher_without_identity_predictions = query_reference_teachers(
                            reference_latents,
                            cap_without_identity,
                            omni_cap_without_identity,
                        )
                    negative_teacher_predictions = (
                        query_reference_teachers(
                            negative_reference_latents,
                            cap_pos,
                            omni_cap_pos,
                        )
                        if visual_identity_mode
                        and args.residual_contrast != "visual_identity_kv_value_project"
                        else []
                    )
                    kv.INJ["mode"] = "off"
                    kv.STORE.clear()
                    teacher_prediction = torch.stack(teacher_predictions, dim=0).mean(dim=0)
                    residual_base_prediction = (
                        base_guided_prediction
                        if args.teacher_backend == "native_omni"
                        else base_prediction
                    )
                    positive_residuals = [
                        prediction - residual_base_prediction
                        for prediction in teacher_predictions
                    ]
                    positive_residual = torch.stack(
                        positive_residuals,
                        dim=0,
                    ).mean(dim=0)
                    negative_residual = None
                    centering_beta = 0.0
                    positive_negative_cosine = None
                    support_gate_mean = None
                    subspace_effective_rank = None
                    subspace_removed_fraction = None
                    subspace_rescue_fraction = None
                    kv_value_beta_mean = None
                    kv_value_retained_rms_ratio = None
                    if visual_identity_mode:
                        if args.residual_contrast == "visual_identity_kv_value_project":
                            prediction_delta = positive_residual
                            kv_value_beta_mean = float(
                                kv_value_diagnostics["beta_mean"]
                            )
                            kv_value_retained_rms_ratio = float(
                                kv_value_diagnostics["value_retained_rms_ratio_mean"]
                            )
                            centering_beta = kv_value_beta_mean
                        else:
                            negative_teacher_prediction = torch.stack(
                                negative_teacher_predictions,
                                dim=0,
                            ).mean(dim=0)
                            negative_residual = (
                                negative_teacher_prediction
                                - residual_base_prediction
                            )
                            negative_residuals = [
                                prediction - residual_base_prediction
                                for prediction in negative_teacher_predictions
                            ]
                            positive_negative_cosine = tensor_cosine(
                                positive_residual,
                                negative_residual,
                            )
                        if args.residual_contrast == "visual_identity_kv_value_project":
                            pass
                        elif args.residual_contrast == "visual_identity_gated":
                            (
                                prediction_delta,
                                centering_beta,
                                support_gate,
                            ) = gated_visual_identity_contrast(
                                positive_residual,
                                negative_residual,
                                shared_keep=args.shared_keep,
                                support_quantile=args.support_quantile,
                                support_gate_power=args.support_gate_power,
                            )
                            support_gate_mean = float(support_gate.mean().item())
                        elif args.residual_contrast == "visual_identity_subject_gate":
                            (
                                prediction_delta,
                                centering_beta,
                                support_gate,
                            ) = subject_support_visual_identity_contrast(
                                positive_residual,
                                negative_residual,
                                base_prediction - base_background,
                                shared_keep=args.shared_keep,
                                outside_keep=args.outside_keep,
                                support_quantile=args.support_quantile,
                                support_gate_power=args.support_gate_power,
                            )
                            support_gate_mean = float(support_gate.mean().item())
                        elif args.residual_contrast == "visual_identity_token_gate":
                            (
                                prediction_delta,
                                centering_beta,
                                support_gate,
                            ) = token_support_visual_identity_contrast(
                                positive_residual,
                                negative_residual,
                                token_support,
                                shared_keep=args.shared_keep,
                                outside_keep=args.outside_keep,
                                support_quantile=args.support_quantile,
                                support_gate_power=args.support_gate_power,
                            )
                            support_gate_mean = float(support_gate.mean().item())
                        elif args.residual_contrast == "visual_identity_subspace":
                            prediction_delta, subspace_diagnostics = (
                                low_rank_nuisance_identity_contrast(
                                    positive_residuals,
                                    negative_residuals,
                                    rank=args.subspace_rank,
                                    ridge=args.subspace_ridge,
                                    projection_strength=args.subspace_strength,
                                    consensus_rescue=args.consensus_rescue,
                                )
                            )
                            subspace_effective_rank = int(
                                subspace_diagnostics["effective_rank"]
                            )
                            subspace_removed_fraction = float(
                                subspace_diagnostics["removed_fraction"]
                            )
                            subspace_rescue_fraction = float(
                                subspace_diagnostics["rescue_fraction"]
                            )
                            support_gate_mean = float(
                                subspace_diagnostics["consensus_gate_mean"]
                            )
                            centering_beta = subspace_removed_fraction
                        else:
                            prediction_delta, centering_beta = visual_identity_contrast(
                                positive_residual,
                                negative_residual,
                                args.residual_contrast,
                                projection_strength=args.projection_strength,
                            )
                    else:
                        prediction_delta = positive_residual
                    if cap_without_identity is not None:
                        teacher_without_identity = torch.stack(
                            teacher_without_identity_predictions,
                            dim=0,
                        ).mean(dim=0)
                        prediction_delta = prediction_delta - (
                            teacher_without_identity
                            - base_without_identity_guided
                        )
                    score_delta = score_delta_from_prediction_delta(prediction_delta, sigma)
                    pseudo_direction = rms_normalize_per_sample(score_delta)
                    base_velocity = prediction_to_velocity(x0_base_prediction)

                x0_prediction = latents - sigma.float() * base_velocity.detach()
                x0_target = (
                    x0_prediction.detach()
                    + float(args.pseudo_step_size) * taper_scale * pseudo_direction.detach()
                ).detach()
                style_loss = (
                    multiscale_pseudo_target_loss(
                        x0_prediction,
                        x0_target,
                        roi=distill_roi,
                        outside_weight=args.distill_outside_weight,
                        lowpass_kernel=args.distill_lowpass_kernel,
                        lowpass_weight=args.distill_lowpass_weight,
                    )
                    if score_active
                    else x0_prediction.new_zeros(())
                )
                content_decay = 1.0 - (inner_index / max(args.inner_iters, 1)) * 0.5
                content_loss = torch.nn.functional.l1_loss(
                    latents.float(),
                    content_anchor.float(),
                ) * content_decay
                identity_loss = x0_prediction.new_zeros(())
                dino_identity_loss = x0_prediction.new_zeros(())
                dino_patch_identity_loss = x0_prediction.new_zeros(())
                identity_grad_rms = None
                identity_score_cosine = None
                identity_pareto_weights = None
                identity_constraint_diagnostics = None
                identity_alignment_diagnostics = None
                weighted_identity_terms = []
                if identity_active:
                    identity_pixels = decode_x0_to_pixels(
                        kv.pipe.vae,
                        x0_prediction,
                    )
                    identity_loss = identity_loss_from_pixels(
                        ccip_model,
                        identity_pixels,
                        ccip_reference_target,
                        fixed_batch_size=args.identity_model_batch_size,
                        roi=identity_roi,
                    )
                    if dino_reference_target is not None:
                        dino_identity_loss = dino_identity_loss_from_pixels(
                            dino_model,
                            identity_pixels,
                            dino_reference_target,
                            roi=identity_roi,
                        )
                    if dino_reference_patch_target is not None:
                        dino_patch_identity_loss = (
                            dino_patch_identity_loss_from_pixels(
                                dino_model,
                                identity_pixels,
                                dino_reference_patch_target,
                                roi=identity_roi,
                                temperature=args.identity_dino_patch_temperature,
                            )
                        )
                    combined_identity_loss = (
                        args.identity_loss_weight * identity_loss
                        + args.identity_dino_weight * dino_identity_loss
                        + args.identity_dino_patch_weight
                        * dino_patch_identity_loss
                    )
                    weighted_identity_terms = [
                        args.identity_loss_weight * identity_loss,
                    ]
                    if dino_reference_target is not None:
                        weighted_identity_terms.append(
                            args.identity_dino_weight * dino_identity_loss
                        )
                    if dino_reference_patch_target is not None:
                        weighted_identity_terms.append(
                            args.identity_dino_patch_weight
                            * dino_patch_identity_loss
                        )
                    individual_identity_gradients = [
                        torch.autograd.grad(
                            term,
                            latents,
                            retain_graph=True,
                        )[0]
                        for term in weighted_identity_terms
                    ]
                    if args.identity_gradient_mode in {
                        "pareto_constrained",
                        "pareto_aligned_distill",
                    }:
                        identity_gradient, pareto_weights = pareto_common_gradient(
                            individual_identity_gradients,
                            iterations=args.identity_pareto_iters,
                        )
                        protected_gradients = []
                        if (
                            args.identity_gradient_mode == "pareto_constrained"
                            and score_active
                            and args.style_weight > 0.0
                        ):
                            protected_gradients.append(
                                torch.autograd.grad(
                                    args.style_weight * style_loss,
                                    latents,
                                    retain_graph=True,
                                )[0]
                            )
                        if args.content_weight > 0.0:
                            protected_gradients.append(
                                torch.autograd.grad(
                                    args.content_weight * content_loss,
                                    latents,
                                    retain_graph=True,
                                )[0]
                            )
                        (
                            identity_gradient,
                            identity_constraint_diagnostics,
                        ) = project_identity_gradient_to_protected_cone(
                            identity_gradient,
                            protected_gradients,
                            iterations=args.identity_constraint_iters,
                        )
                        if (
                            args.identity_gradient_mode
                            == "pareto_aligned_distill"
                            and score_active
                        ):
                            score_gradient = torch.autograd.grad(
                                args.style_weight * style_loss,
                                latents,
                                retain_graph=True,
                            )[0]
                            (
                                identity_gradient,
                                identity_alignment_diagnostics,
                            ) = positive_local_alignment_projection(
                                score_gradient,
                                identity_gradient,
                                max_coefficient=(
                                    args.identity_alignment_max_coefficient
                                ),
                            )
                        identity_pareto_weights = [
                            float(value)
                            for value in pareto_weights.detach().cpu().tolist()
                        ]
                    else:
                        identity_gradient = sum(individual_identity_gradients)
                    identity_grad_rms = tensor_rms(identity_gradient)
                    identity_score_cosine = tensor_cosine(
                        -identity_gradient,
                        pseudo_direction,
                    )
                loss = (
                    args.style_weight * style_loss
                    + args.content_weight * content_loss
                    + args.identity_loss_weight * identity_loss
                    + args.identity_dino_weight * dino_identity_loss
                    + args.identity_dino_patch_weight * dino_patch_identity_loss
                )
                if (
                    identity_active
                    and args.identity_gradient_mode
                    in {"pareto_constrained", "pareto_aligned_distill"}
                ):
                    protected_loss = args.content_weight * content_loss
                    if args.identity_gradient_mode == "pareto_constrained":
                        protected_loss = (
                            protected_loss + args.style_weight * style_loss
                        )
                    protected_gradient = torch.autograd.grad(
                        protected_loss,
                        latents,
                        retain_graph=False,
                    )[0]
                    latents.grad = protected_gradient + identity_gradient
                else:
                    loss.backward()
                total_grad_rms = tensor_rms(latents.grad)
                optimizer.step()
                inner_logs.append(
                    {
                        "inner": inner_index,
                        "loss": float(loss.detach().item()),
                        "style_loss": float(style_loss.detach().item()),
                        "content_loss": float(content_loss.detach().item()),
                        "identity_loss": (
                            float(identity_loss.detach().item())
                            if identity_active
                            else None
                        ),
                        "dino_identity_loss": (
                            float(dino_identity_loss.detach().item())
                            if identity_active and dino_reference_target is not None
                            else None
                        ),
                        "dino_patch_identity_loss": (
                            float(dino_patch_identity_loss.detach().item())
                            if identity_active
                            and dino_reference_patch_target is not None
                            else None
                        ),
                        "identity_grad_rms": identity_grad_rms,
                        "identity_score_cosine": identity_score_cosine,
                        "identity_pareto_weights": identity_pareto_weights,
                        "identity_constraint_diagnostics": (
                            identity_constraint_diagnostics
                        ),
                        "identity_alignment_diagnostics": (
                            identity_alignment_diagnostics
                        ),
                        "total_grad_rms": total_grad_rms,
                        "latent_grad_rms": tensor_rms(latents.grad),
                        "latent_update_rms": tensor_rms(latents.detach() - content_anchor),
                        "positive_residual_rms": tensor_rms(positive_residual),
                        "negative_residual_rms": (
                            tensor_rms(negative_residual)
                            if negative_residual is not None
                            else None
                        ),
                        "positive_negative_cosine": positive_negative_cosine,
                        "centering_beta": centering_beta,
                        "support_gate_mean": support_gate_mean,
                        "subspace_effective_rank": subspace_effective_rank,
                        "subspace_removed_fraction": subspace_removed_fraction,
                        "subspace_rescue_fraction": subspace_rescue_fraction,
                        "kv_value_beta_mean": kv_value_beta_mean,
                        "kv_value_retained_rms_ratio": kv_value_retained_rms_ratio,
                        "prediction_delta_rms": tensor_rms(prediction_delta),
                        "score_delta_rms": tensor_rms(score_delta),
                    }
                )
            latents = latents.detach()

        logs.append(
            {
                "outer": outer_index,
                "timestep": float(timestep.item()),
                "scheduler_sigma": float(scheduler_sigma.item()),
                "distill_timestep": (
                    float(distill_timestep.item())
                    if distill_timestep is not None
                    else None
                ),
                "sigma": float(sigma.item()),
                "progress": progress,
                "active": active,
                "score_active": score_active,
                "identity_active": identity_active,
                "trajectory_identity_active": trajectory_identity_active,
                "trajectory_identity": trajectory_identity_log,
                "inner": inner_logs,
            }
        )
        kv.STORE.clear()
        torch.cuda.empty_cache()
        if args.verbose:
            print(
                f"outer={outer_index:02d} scheduler_sigma={float(scheduler_sigma):.4f} "
                f"distill_sigma={float(sigma):.4f} "
                f"active={active} "
                f"loss={inner_logs[-1]['loss'] if inner_logs else 0.0:.6g} "
                f"alloc={torch.cuda.memory_allocated() / 2**30:.2f}GiB",
                flush=True,
            )

    terminal_identity_logs = []
    if identity_enabled and args.identity_terminal_iters > 0:
        terminal_anchor = latents.detach().clone()
        with torch.no_grad():
            terminal_pixel_anchor = decode_x0_to_pixels(
                kv.pipe.vae,
                terminal_anchor,
            ).detach()
        latents = latents.detach().requires_grad_(True)
        terminal_optimizer = torch.optim.AdamW(
            [latents],
            lr=(
                args.identity_terminal_lr
                if args.identity_terminal_lr is not None
                else args.lr
            ),
            weight_decay=0.0,
        )
        for terminal_index in range(args.identity_terminal_iters):
            terminal_optimizer.zero_grad(set_to_none=True)
            terminal_pixels = decode_x0_to_pixels(kv.pipe.vae, latents)
            terminal_identity_loss = identity_loss_from_pixels(
                ccip_model,
                terminal_pixels,
                ccip_reference_target,
                fixed_batch_size=args.identity_model_batch_size,
                roi=identity_roi,
            )
            terminal_dino_identity_loss = terminal_pixels.new_zeros(())
            if dino_reference_target is not None:
                terminal_dino_identity_loss = dino_identity_loss_from_pixels(
                    dino_model,
                    terminal_pixels,
                    dino_reference_target,
                    roi=identity_roi,
                )
            terminal_dino_patch_identity_loss = terminal_pixels.new_zeros(())
            if dino_reference_patch_target is not None:
                terminal_dino_patch_identity_loss = (
                    dino_patch_identity_loss_from_pixels(
                        dino_model,
                        terminal_pixels,
                        dino_reference_patch_target,
                        roi=identity_roi,
                        temperature=args.identity_dino_patch_temperature,
                    )
                )
            terminal_content_loss = F.l1_loss(
                latents.float(),
                terminal_anchor.float(),
            )
            terminal_pixel_loss = F.l1_loss(
                terminal_pixels.float(),
                terminal_pixel_anchor.float(),
            )
            terminal_outside_pixel_loss = outside_roi_l1_loss(
                terminal_pixels,
                terminal_pixel_anchor,
                identity_roi,
            )
            terminal_loss = (
                args.identity_loss_weight * terminal_identity_loss
                + args.identity_dino_weight * terminal_dino_identity_loss
                + args.identity_dino_patch_weight
                * terminal_dino_patch_identity_loss
                + args.content_weight * terminal_content_loss
                + args.identity_terminal_pixel_weight * terminal_pixel_loss
                + args.identity_terminal_outside_pixel_weight
                * terminal_outside_pixel_loss
            )
            terminal_pareto_weights = None
            terminal_constraint_diagnostics = None
            if args.identity_gradient_mode in {
                "pareto_constrained",
                "pareto_aligned_distill",
            }:
                identity_terms = [
                    args.identity_loss_weight * terminal_identity_loss,
                ]
                if dino_reference_target is not None:
                    identity_terms.append(
                        args.identity_dino_weight * terminal_dino_identity_loss
                    )
                if dino_reference_patch_target is not None:
                    identity_terms.append(
                        args.identity_dino_patch_weight
                        * terminal_dino_patch_identity_loss
                    )
                identity_gradients = [
                    torch.autograd.grad(
                        term,
                        latents,
                        retain_graph=True,
                    )[0]
                    for term in identity_terms
                ]
                terminal_identity_gradient, pareto_weights = pareto_common_gradient(
                    identity_gradients,
                    iterations=args.identity_pareto_iters,
                )
                protected_terms = []
                if args.content_weight > 0.0:
                    protected_terms.append(
                        args.content_weight * terminal_content_loss
                    )
                if args.identity_terminal_pixel_weight > 0.0:
                    protected_terms.append(
                        args.identity_terminal_pixel_weight * terminal_pixel_loss
                    )
                if args.identity_terminal_outside_pixel_weight > 0.0:
                    protected_terms.append(
                        args.identity_terminal_outside_pixel_weight
                        * terminal_outside_pixel_loss
                    )
                protected_gradients = [
                    torch.autograd.grad(
                        term,
                        latents,
                        retain_graph=True,
                    )[0]
                    for term in protected_terms
                ]
                (
                    terminal_identity_gradient,
                    terminal_constraint_diagnostics,
                ) = project_identity_gradient_to_protected_cone(
                    terminal_identity_gradient,
                    protected_gradients,
                    iterations=args.identity_constraint_iters,
                )
                if protected_terms:
                    terminal_protected_gradient = torch.autograd.grad(
                        sum(protected_terms),
                        latents,
                        retain_graph=False,
                    )[0]
                else:
                    terminal_protected_gradient = torch.zeros_like(latents)
                latents.grad = (
                    terminal_protected_gradient + terminal_identity_gradient
                )
                terminal_pareto_weights = [
                    float(value)
                    for value in pareto_weights.detach().cpu().tolist()
                ]
            else:
                terminal_loss.backward()
            terminal_grad_rms = tensor_rms(latents.grad)
            terminal_optimizer.step()
            terminal_identity_logs.append(
                {
                    "inner": terminal_index,
                    "loss": float(terminal_loss.detach().item()),
                    "identity_loss": float(
                        terminal_identity_loss.detach().item()
                    ),
                    "dino_identity_loss": (
                        float(terminal_dino_identity_loss.detach().item())
                        if dino_reference_target is not None
                        else None
                    ),
                    "dino_patch_identity_loss": (
                        float(terminal_dino_patch_identity_loss.detach().item())
                        if dino_reference_patch_target is not None
                        else None
                    ),
                    "content_loss": float(
                        terminal_content_loss.detach().item()
                    ),
                    "pixel_trust_loss": float(
                        terminal_pixel_loss.detach().item()
                    ),
                    "outside_pixel_trust_loss": float(
                        terminal_outside_pixel_loss.detach().item()
                    ),
                    "latent_grad_rms": terminal_grad_rms,
                    "identity_pareto_weights": terminal_pareto_weights,
                    "identity_constraint_diagnostics": (
                        terminal_constraint_diagnostics
                    ),
                    "latent_update_rms": tensor_rms(
                        latents.detach() - terminal_anchor
                    ),
                }
            )
        latents = latents.detach()

    with torch.no_grad():
        image = impl.decode_latent(kv, latents)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"u{args.uid}_{args.tag}_s{args.seed}_n{args.steps}_i{args.inner_iters}"
    image_path = OUT_ROOT / f"{stem}.png"
    json_path = OUT_ROOT / f"{stem}.json"
    image.save(image_path)
    metadata = {
        "method": "true_inner_loop_score_tsd",
        "uid": args.uid,
        "references": [str(path) for path in reference_paths],
        "negative_references": [str(path) for path in negative_reference_paths],
        "identity_capsule_file": str(capsule_path) if capsule_path is not None else None,
        "identity_capsule": capsule,
        "prompt": prompt,
        "capture_prompt_mode": args.capture_prompt_mode,
        "reference_caption_mode": args.reference_caption_mode,
        "reference_caption": reference_caption,
        "negative": negative,
        "teacher_backend": args.teacher_backend,
        "seed": args.seed,
        "size": args.size,
        "ref_size": args.ref_size,
        "steps": args.steps,
        "cfg": args.cfg,
        "inner_x0_mode": args.inner_x0_mode,
        "teacher_weight": args.teacher_weight,
        "inner_iters": args.inner_iters,
        "lr": args.lr,
        "pseudo_step_size": args.pseudo_step_size,
        "style_weight": args.style_weight,
        "content_weight": args.content_weight,
        "distill_roi": list(distill_roi) if distill_roi is not None else None,
        "distill_outside_weight": args.distill_outside_weight,
        "distill_lowpass_kernel": args.distill_lowpass_kernel,
        "distill_lowpass_weight": args.distill_lowpass_weight,
        "identity_loss_weight": args.identity_loss_weight,
        "identity_sigma_max": args.identity_sigma_max,
        "identity_trajectory_rms": args.identity_trajectory_rms,
        "identity_terminal_iters": args.identity_terminal_iters,
        "identity_terminal_lr": (
            args.identity_terminal_lr
            if args.identity_terminal_lr is not None
            else args.lr
        ),
        "identity_terminal_pixel_weight": args.identity_terminal_pixel_weight,
        "identity_terminal_outside_pixel_weight": (
            args.identity_terminal_outside_pixel_weight
        ),
        "identity_roi": list(identity_roi) if identity_roi is not None else None,
        "identity_dino_weight": args.identity_dino_weight,
        "identity_dino_patch_weight": args.identity_dino_patch_weight,
        "identity_dino_patch_temperature": args.identity_dino_patch_temperature,
        "identity_gradient_mode": args.identity_gradient_mode,
        "identity_pareto_iters": args.identity_pareto_iters,
        "identity_constraint_iters": args.identity_constraint_iters,
        "identity_alignment_max_coefficient": (
            args.identity_alignment_max_coefficient
        ),
        "identity_dino_model": (
            args.identity_dino_model
            if identity_enabled
            and (
                args.identity_dino_weight > 0.0
                or args.identity_dino_patch_weight > 0.0
            )
            else None
        ),
        "identity_model_path": (
            str(identity_model_path) if identity_enabled else None
        ),
        "identity_model_batch_size": (
            args.identity_model_batch_size if identity_enabled else None
        ),
        "identity_objective": (
            "decoded_x0_roi_ccip_dino_cls_soft_chamfer_to_reference"
            if identity_enabled and args.identity_dino_patch_weight > 0.0
            else "decoded_x0_roi_ccip_dino_cosine_to_mean_reference"
            if identity_enabled and args.identity_dino_weight > 0.0
            else "decoded_x0_roi_ccip_cosine_to_mean_reference"
            if identity_enabled and identity_roi is not None
            else "decoded_x0_ccip_cosine_to_mean_reference"
            if identity_enabled
            else None
        ),
        "active_window": [args.active_start, args.active_end],
        "taper": args.taper,
        "residual_contrast": args.residual_contrast,
        "projection_strength": args.projection_strength,
        "shared_keep": args.shared_keep,
        "support_quantile": args.support_quantile,
        "support_gate_power": args.support_gate_power,
        "background_prompt": args.background_prompt,
        "outside_keep": args.outside_keep,
        "subject_token": args.subject_token,
        "subject_token_indices": subject_token_indices,
        "support_layers": sorted(support_layers),
        "subspace_rank": args.subspace_rank,
        "subspace_ridge": args.subspace_ridge,
        "subspace_strength": args.subspace_strength,
        "consensus_rescue": args.consensus_rescue,
        "kv_value_projection_strength": args.kv_value_projection_strength,
        "support_source": (
            "frozen_base_full_minus_background"
            if args.residual_contrast == "visual_identity_subject_gate"
            else "frozen_base_image_query_to_subject_token_attention"
            if args.residual_contrast == "visual_identity_token_gate"
            else "teacher_projected_residual"
            if args.residual_contrast == "visual_identity_gated"
            else "multi_reference_low_rank_nuisance_subspace"
            if args.residual_contrast == "visual_identity_subspace"
            else "positive_keys_negative_projected_values"
            if args.residual_contrast == "visual_identity_kv_value_project"
            else None
        ),
        "distillation": args.distillation,
        "inner_order": (
            "pre_scheduler_identity_velocity_and_post_scheduler_score"
            if args.identity_trajectory_rms > 0.0
            else "post_scheduler_next_sigma"
        ),
        "teacher_only_reference_injection": True,
        "teacher_interface": (
            "native_omni_clean_reference_plus_noisy_target_with_official_prompt_framing"
            if args.teacher_backend == "native_omni"
            else "legacy_attention_kv_capture_and_reinjection"
        ),
        "scheduler_consumes_reference_prediction": False,
        "scheduler_consumes_identity_loss_correction": (
            args.identity_trajectory_rms > 0.0
        ),
        "wall_time_s": time.time() - started,
        "image_path": str(image_path),
        "logs": logs,
        "terminal_identity_logs": terminal_identity_logs,
    }
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {image_path}")
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

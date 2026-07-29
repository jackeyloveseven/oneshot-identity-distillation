#!/usr/bin/env python3
"""Few-shot identity residual distillation on the frozen Z-Image T2I base.

The existing strip reference path is used only to query transient teacher
velocities. The scheduler never consumes a reference-injected prediction.
Instead, matched teacher-minus-base residuals from several reference views are
aggregated into a detached x0 pseudo-target, the current latent is updated, and
the pure T2I base prediction is recomputed before the ordinary reverse step.

This is the minimal first-order experiment. Higher-order curvature gates are
deliberately optional and disabled by default.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance


ROOT = Path(__file__).resolve().parents[1]
BASE_PROBE = ROOT / "probe" / "30_kv_inject.py"
OUT_ROOT = ROOT / "out" / "identity_residual_tsd"


@dataclass(frozen=True)
class ReferenceView:
    name: str
    reference_index: int
    transform: str
    role: str
    latent: torch.Tensor


def load_base_probe():
    spec = importlib.util.spec_from_file_location("cherry_kv_teacher", BASE_PROBE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {BASE_PROBE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=["residual_tsd", "prompt_only", "direct_teacher"],
        default="residual_tsd",
    )
    parser.add_argument("--uid", required=True)
    parser.add_argument("--references", default=None, help="comma-separated paths; defaults to REF/u<uid>_ref.png")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--identity-capsule", default=None, help="concise stable subject description appended to the prompt")
    parser.add_argument("--identity-capsule-file", default=None, help="JSON produced by probe/111_identity_capsule.py")
    parser.add_argument("--negative", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ref-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--teacher-weight", type=float, default=4.0)
    parser.add_argument("--views-per-reference", type=int, default=2, choices=[1, 2, 4])
    parser.add_argument(
        "--aggregation",
        choices=["mean", "svd", "semantic"],
        default="mean",
        help=(
            "semantic uses only true-reference anchors for identity consensus; "
            "within-reference color/framing perturbations estimate a nuisance subspace"
        ),
    )
    parser.add_argument(
        "--residual-filter",
        choices=["none", "channel_center", "highpass", "highpass_channel"],
        default="none",
        help="remove simple color/composition nuisance before cross-view aggregation",
    )
    parser.add_argument(
        "--residual-contrast",
        choices=["teacher_base", "identity_interaction"],
        default="teacher_base",
        help=(
            "identity_interaction subtracts the reference effect without the identity capsule, "
            "retaining only reference-by-identity semantic interaction"
        ),
    )
    parser.add_argument("--tsd-step", type=float, default=0.02)
    parser.add_argument("--active-start", type=float, default=0.0)
    parser.add_argument("--active-end", type=float, default=0.65)
    parser.add_argument("--active-stride", type=int, default=1)
    parser.add_argument("--taper", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--min-consensus", type=float, default=0.0)
    parser.add_argument(
        "--semantic-nuisance-views",
        default="desaturate,center_crop,contrast",
        help="comma-separated identity-preserving perturbations used only by semantic aggregation",
    )
    parser.add_argument("--semantic-local-gate-power", type=float, default=1.0)
    parser.add_argument("--semantic-nuisance-ridge", type=float, default=0.05)
    parser.add_argument(
        "--semantic-max-removal",
        type=float,
        default=0.45,
        help="maximum nuisance-projection norm as a fraction of the anchor consensus norm",
    )
    parser.add_argument(
        "--hidden-support-layers",
        default="",
        help="optional inclusive layer range such as 21-25; records hidden differences but never injects them",
    )
    parser.add_argument(
        "--hidden-support-floor",
        type=float,
        default=0.10,
        help="minimum spatial gate retained when hidden donor-difference support is enabled",
    )
    parser.add_argument(
        "--cpu-offload-static",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="move the text encoder and idle VAE to CPU during denoising",
    )
    parser.add_argument(
        "--aggressive-memory-cleanup",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="release transient teacher tensors between frozen-model queries",
    )
    parser.add_argument("--verbose-steps", action="store_true")
    parser.add_argument(
        "--max-lowfreq-ratio",
        type=float,
        default=None,
        help="optional first-order structure gate; disabled by default",
    )
    parser.add_argument("--tag", default="smoke")
    return parser.parse_args()


def reference_paths(args: argparse.Namespace, kv) -> list[Path]:
    if args.references:
        paths = [Path(item).expanduser().resolve() for item in args.references.split(",") if item.strip()]
    else:
        paths = [Path(kv.REF) / f"u{args.uid}_ref.png"]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return paths


def load_identity_capsule(args: argparse.Namespace) -> tuple[str | None, str | None]:
    if args.identity_capsule and args.identity_capsule_file:
        raise ValueError("use only one of --identity-capsule and --identity-capsule-file")
    if args.identity_capsule:
        return args.identity_capsule.strip(), "cli"
    if not args.identity_capsule_file:
        return None, None
    path = Path(args.identity_capsule_file).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    capsule = payload.get("capsule")
    if not isinstance(capsule, str) or not capsule.strip():
        raise ValueError(f"identity capsule JSON has no non-empty 'capsule': {path}")
    return capsule.strip(), str(path)


def make_views(path: Path, count: int, size: int) -> list[tuple[str, Image.Image]]:
    image = Image.open(path).convert("RGB")
    views = [("identity", image)]
    if count >= 2:
        views.append(("mirror", image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)))
    if count >= 4:
        width, height = image.size
        margin_x = max(1, int(round(width * 0.04)))
        margin_y = max(1, int(round(height * 0.04)))
        crop = image.crop((margin_x, margin_y, width - margin_x, height - margin_y)).resize(image.size, Image.Resampling.LANCZOS)
        views.append(("center_crop", crop))
        views.append(("contrast", ImageEnhance.Contrast(image).enhance(1.06)))
    return [(name, view.resize((size, size), Image.Resampling.LANCZOS)) for name, view in views[:count]]


def make_semantic_views(
    path: Path,
    size: int,
    nuisance_names: list[str],
) -> list[tuple[str, str, Image.Image]]:
    """Build one identity anchor plus controlled nuisance perturbations.

    Perturbed views are never treated as additional identity references. Their
    teacher-response differences from the anchor estimate directions caused by
    color and framing changes that should not define subject identity.
    """

    image = Image.open(path).convert("RGB")
    views: list[tuple[str, str, Image.Image]] = [("identity", "anchor", image)]
    valid = {"desaturate", "center_crop", "contrast"}
    unknown = sorted(set(nuisance_names) - valid)
    if unknown:
        raise ValueError(f"unknown semantic nuisance views: {', '.join(unknown)}")
    for name in nuisance_names:
        if name == "desaturate":
            view = ImageEnhance.Color(image).enhance(0.15)
        elif name == "center_crop":
            width, height = image.size
            margin_x = max(1, int(round(width * 0.06)))
            margin_y = max(1, int(round(height * 0.06)))
            view = image.crop((margin_x, margin_y, width - margin_x, height - margin_y)).resize(
                image.size,
                Image.Resampling.LANCZOS,
            )
        else:  # contrast
            view = ImageEnhance.Contrast(image).enhance(1.12)
        views.append((name, "nuisance", view))
    return [
        (name, role, view.resize((size, size), Image.Resampling.LANCZOS))
        for name, role, view in views
    ]


@torch.inference_mode()
def encode_reference_image(kv, image: Image.Image) -> torch.Tensor:
    array = torch.from_numpy(np.asarray(image).copy()).float().permute(2, 0, 1) / 127.5 - 1.0
    array = array.unsqueeze(0).to(kv.DEV)
    latent = kv.pipe.vae.encode(array.to(torch.bfloat16)).latent_dist.mode()[0]
    latent = (latent - kv.pipe.vae.config.shift_factor) * kv.pipe.vae.config.scaling_factor
    return latent.unsqueeze(1).to(kv.DT)


def rms_normalize(value: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    rms = value.float().square().mean().sqrt().clamp_min(eps)
    return (value.float() / rms).to(value.dtype)


def cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-8) -> float:
    a = left.float().reshape(-1)
    b = right.float().reshape(-1)
    return float((a @ b / (a.norm().clamp_min(eps) * b.norm().clamp_min(eps))).item())


def filter_residual(value: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "none":
        return value
    filtered = value.float()
    if mode in {"channel_center", "highpass_channel"}:
        filtered = filtered - filtered.mean(dim=(-2, -1), keepdim=True)
    if mode in {"highpass", "highpass_channel"}:
        spatial = filtered
        kernel = min(8, spatial.shape[-2], spatial.shape[-1])
        pooled = F.avg_pool2d(spatial, kernel_size=kernel, stride=kernel)
        low = F.interpolate(pooled, size=spatial.shape[-2:], mode="bilinear", align_corners=False)
        filtered = filtered - low
    return filtered.to(value.dtype)


def aggregate_residuals(
    residuals: list[torch.Tensor],
    mode: str,
    residual_filter: str = "none",
) -> tuple[torch.Tensor, list[float], dict[str, object]]:
    if not residuals:
        raise ValueError("at least one residual is required")
    filtered_residuals = [filter_residual(item, residual_filter) for item in residuals]
    normalized = torch.stack([rms_normalize(item) for item in filtered_residuals], dim=0)
    mean = normalized.mean(dim=0)
    mean_norm = rms_normalize(mean)
    similarities = [cosine(item, mean_norm) for item in normalized]
    weights = torch.tensor([max(value, 0.0) for value in similarities], device=normalized.device, dtype=torch.float32)
    if float(weights.sum().item()) <= 1e-8:
        weights = torch.ones_like(weights)

    extra: dict[str, object] = {"view_cosine_to_mean": similarities}
    weight_shape = (len(residuals),) + (1,) * residuals[0].ndim
    if mode == "mean" or len(residuals) == 1:
        direction = (weights.reshape(weight_shape) * normalized.float()).sum(dim=0) / weights.sum()
    else:
        flat = normalized.float().reshape(len(residuals), -1)
        gram = flat @ flat.T / flat.shape[1]
        eigvals, eigvecs = torch.linalg.eigh(gram)
        coeff = eigvecs[:, -1]
        if float(coeff.sum().item()) < 0:
            coeff = -coeff
        direction = (coeff.reshape(weight_shape) * normalized.float()).sum(dim=0)
        extra.update({
            "gram_eigenvalues": [float(value) for value in eigvals.detach().cpu()],
            "top_eigenvector": [float(value) for value in coeff.detach().cpu()],
        })
    return rms_normalize(direction), similarities, extra


def local_reference_agreement(
    residuals: list[torch.Tensor],
    *,
    power: float = 1.0,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return a target-grid gate from local cross-reference channel agreement."""

    if not residuals:
        raise ValueError("at least one residual is required")
    if len(residuals) == 1:
        gate = torch.ones(residuals[0].shape[-2:], device=residuals[0].device, dtype=torch.float32)
        return gate, {"local_agreement_mean": 1.0, "local_agreement_min": 1.0}
    stack = torch.stack([rms_normalize(item).float() for item in residuals], dim=0)
    height, width = stack.shape[-2:]
    local = stack.reshape(stack.shape[0], -1, height, width)
    local_mean = local.mean(dim=0)
    numerator = (local * local_mean.unsqueeze(0)).sum(dim=1)
    denominator = local.square().sum(dim=1).sqrt() * local_mean.square().sum(dim=0).sqrt().unsqueeze(0)
    cosine_map = numerator / denominator.clamp_min(eps)
    agreement = cosine_map.clamp(min=0.0, max=1.0).mean(dim=0)
    gate = agreement.pow(max(float(power), 0.0))
    return gate, {
        "local_agreement_mean": float(agreement.mean().item()),
        "local_agreement_min": float(agreement.min().item()),
    }


def project_nuisance_subspace(
    direction: torch.Tensor,
    nuisance_deltas: list[torch.Tensor],
    *,
    ridge: float = 0.05,
    max_removal: float = 0.45,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Remove the response span induced by identity-preserving perturbations.

    The solve occurs in the small nuisance Gram matrix, so the full latent
    feature dimension is never factorized. Removal is capped conservatively to
    prevent an imperfect perturbation from deleting the complete identity cue.
    """

    valid = []
    for delta in nuisance_deltas:
        if float(delta.float().square().mean().sqrt().item()) > eps:
            valid.append(rms_normalize(delta).float().reshape(-1))
    if not valid:
        return direction, {
            "nuisance_count": 0,
            "nuisance_projection_fraction": 0.0,
            "nuisance_projection_capped": False,
        }

    base = direction.float().reshape(-1)
    basis = torch.stack(valid, dim=0)
    feature_count = basis.shape[1]
    gram = basis @ basis.T / feature_count
    rhs = basis @ base / feature_count
    regularized = gram + max(float(ridge), 0.0) * torch.eye(
        gram.shape[0], device=gram.device, dtype=gram.dtype
    )
    coeff = torch.linalg.solve(regularized, rhs)
    projected = coeff @ basis
    base_norm = base.norm().clamp_min(eps)
    projection_fraction = float((projected.norm() / base_norm).item())
    capped = projection_fraction > max(float(max_removal), 0.0)
    if capped:
        projected = projected * (float(max_removal) / max(projection_fraction, eps))
    cleaned = (base - projected).reshape_as(direction)
    return cleaned.to(direction.dtype), {
        "nuisance_count": len(valid),
        "nuisance_projection_fraction": projection_fraction,
        "nuisance_projection_capped": capped,
        "nuisance_gram_eigenvalues": [
            float(value) for value in torch.linalg.eigvalsh(gram).detach().cpu()
        ],
    }


def semantic_aggregate_residuals(
    anchor_residuals: list[torch.Tensor],
    nuisance_deltas: list[torch.Tensor],
    *,
    residual_filter: str = "none",
    local_gate_power: float = 1.0,
    nuisance_ridge: float = 0.05,
    max_nuisance_removal: float = 0.45,
) -> tuple[torch.Tensor, list[float], dict[str, object]]:
    """Aggregate true references, then reject locally unstable nuisance response."""

    direction, similarities, extra = aggregate_residuals(
        anchor_residuals,
        "svd" if len(anchor_residuals) >= 3 else "mean",
        residual_filter,
    )
    filtered_anchors = [filter_residual(item, residual_filter) for item in anchor_residuals]
    gate, gate_stats = local_reference_agreement(filtered_anchors, power=local_gate_power)
    direction = direction.float() * gate.unsqueeze(0).unsqueeze(0)
    filtered_nuisance = [filter_residual(item, residual_filter) for item in nuisance_deltas]
    direction, nuisance_stats = project_nuisance_subspace(
        direction,
        filtered_nuisance,
        ridge=nuisance_ridge,
        max_removal=max_nuisance_removal,
    )
    extra.update(gate_stats)
    extra.update(nuisance_stats)
    return rms_normalize(direction), similarities, extra


def identity_interaction_residual(
    teacher_with_identity: torch.Tensor,
    base_with_identity: torch.Tensor,
    teacher_without_identity: torch.Tensor,
    base_without_identity: torch.Tensor,
) -> torch.Tensor:
    """Difference-in-differences that removes generic reference response."""

    return (teacher_with_identity - base_with_identity) - (
        teacher_without_identity - base_without_identity
    )


def parse_layer_spec(spec: str, layer_count: int) -> list[int]:
    if not spec.strip():
        return []
    layers: set[int] = set()
    for item in spec.split(","):
        token = item.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start, end = int(start_text), int(end_text)
            if end < start:
                raise ValueError(f"invalid descending hidden layer range: {token}")
            layers.update(range(start, end + 1))
        else:
            layers.add(int(token))
    invalid = sorted(layer for layer in layers if layer < 0 or layer >= layer_count)
    if invalid:
        raise ValueError(f"hidden support layers outside [0,{layer_count - 1}]: {invalid}")
    return sorted(layers)


class HiddenStateRecorder:
    """Transient post-block target-token recorder for teacher/base differences."""

    def __init__(self, transformer, layers: list[int], target_tokens: int):
        self.target_tokens = int(target_tokens)
        self.current_key: str | None = None
        self.records: dict[str, dict[int, torch.Tensor]] = {}
        self.handles = []
        for layer_index in layers:
            self.handles.append(
                transformer.layers[layer_index].register_forward_hook(self._make_hook(layer_index))
            )

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            if self.current_key is None:
                return
            if not torch.is_tensor(output):
                raise RuntimeError(f"layer {layer_index} returned non-tensor hidden state")
            if output.shape[1] < self.target_tokens:
                raise RuntimeError(
                    f"layer {layer_index} sequence {output.shape[1]} shorter than target {self.target_tokens}"
                )
            target = output[:, -self.target_tokens :, :].detach().clone()
            self.records.setdefault(self.current_key, {})[layer_index] = target

        return hook

    def clear(self) -> None:
        self.current_key = None
        self.records.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def hidden_semantic_support_gate(
    hidden_deltas: dict[str, dict[int, torch.Tensor]],
    *,
    anchor_names: list[str],
    nuisance_pairs: list[tuple[str, str]],
    output_size: tuple[int, int],
    floor: float = 0.10,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build a spatial gate from hidden differences stable across references.

    A position is supported when the teacher-minus-base hidden direction agrees
    across true references and remains stable under within-reference color or
    framing perturbations. Hidden states are diagnostic only; no hidden tensor
    is transplanted into the sampling trajectory.
    """

    common_layers = sorted(
        set.intersection(*(set(hidden_deltas[name]) for name in anchor_names))
        if anchor_names
        else set()
    )
    if not common_layers:
        raise ValueError("no common hidden layers for anchor references")
    layer_maps = []
    layer_stats: dict[str, object] = {}
    for layer_index in common_layers:
        anchors = [hidden_deltas[name][layer_index].float() for name in anchor_names]
        normalized = [F.normalize(item, dim=-1, eps=eps) for item in anchors]
        mean = torch.stack(normalized, dim=0).mean(dim=0)
        mean_unit = F.normalize(mean, dim=-1, eps=eps)
        anchor_agreement = torch.stack(
            [(item * mean_unit).sum(dim=-1).clamp(0.0, 1.0) for item in normalized],
            dim=0,
        ).mean(dim=0)

        nuisance_stabilities = []
        for nuisance_name, anchor_name in nuisance_pairs:
            if nuisance_name not in hidden_deltas or anchor_name not in hidden_deltas:
                continue
            if layer_index not in hidden_deltas[nuisance_name] or layer_index not in hidden_deltas[anchor_name]:
                continue
            nuisance = F.normalize(hidden_deltas[nuisance_name][layer_index].float(), dim=-1, eps=eps)
            anchor = F.normalize(hidden_deltas[anchor_name][layer_index].float(), dim=-1, eps=eps)
            nuisance_stabilities.append((nuisance * anchor).sum(dim=-1).clamp(0.0, 1.0))
        if nuisance_stabilities:
            nuisance_stability = torch.stack(nuisance_stabilities, dim=0).mean(dim=0)
        else:
            nuisance_stability = torch.ones_like(anchor_agreement)
        support = anchor_agreement * nuisance_stability
        token_count = support.shape[-1]
        side = int(round(math.sqrt(token_count)))
        if side * side != token_count:
            raise ValueError(f"hidden target token count {token_count} is not square")
        support_map = support.reshape(support.shape[0], 1, side, side)
        support_map = F.interpolate(support_map, size=output_size, mode="bilinear", align_corners=False)
        layer_maps.append(support_map)
        layer_stats[str(layer_index)] = {
            "anchor_agreement_mean": float(anchor_agreement.mean().item()),
            "nuisance_stability_mean": float(nuisance_stability.mean().item()),
            "support_mean": float(support.mean().item()),
        }

    support_map = torch.stack(layer_maps, dim=0).mean(dim=0).squeeze(0).squeeze(0)
    floor_value = min(max(float(floor), 0.0), 1.0)
    gate = floor_value + (1.0 - floor_value) * support_map.clamp(0.0, 1.0)
    return gate, {
        "hidden_support_layers": common_layers,
        "hidden_support_mean": float(support_map.mean().item()),
        "hidden_support_min": float(support_map.min().item()),
        "hidden_support_max": float(support_map.max().item()),
        "hidden_support_floor": floor_value,
        "hidden_support_per_layer": layer_stats,
    }


def resolve_scheduler_sigma(scheduler, timestep: torch.Tensor | float, ref_tensor: torch.Tensor) -> tuple[int, torch.Tensor]:
    """Resolve the current sigma from the scheduler timetable.

    ``scheduler.step_index`` is intentionally not used: diffusers leaves it
    unset until the first reverse step, while identity distillation runs before
    that step.
    """

    scheduler_timesteps = scheduler.timesteps.detach().to(device="cpu")
    if torch.is_tensor(timestep):
        current = timestep.detach().reshape(-1)[0].to(device="cpu", dtype=scheduler_timesteps.dtype)
    else:
        current = torch.tensor(float(timestep), device="cpu", dtype=scheduler_timesteps.dtype)
    index = int(torch.argmin((scheduler_timesteps - current).abs()).item())
    sigma = scheduler.sigmas[index].to(device=ref_tensor.device, dtype=torch.float32)
    return index, sigma


def latent_pseudo_update(
    direction: torch.Tensor,
    sigma: torch.Tensor | float,
    *,
    step_size: float,
    scale: float,
) -> torch.Tensor:
    """Map a model-prediction residual direction to a latent x0 update."""

    sigma_tensor = torch.as_tensor(sigma, device=direction.device, dtype=torch.float32)
    x0_direction = sigma_tensor * direction.float().unsqueeze(0).squeeze(2)
    return float(step_size) * float(scale) * rms_normalize(x0_direction)


def release_cuda_cache(enabled: bool) -> None:
    if enabled and torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


def lowfreq_ratio(direction: torch.Tensor) -> float:
    value = direction.float().squeeze(2)
    pooled = F.avg_pool2d(value, kernel_size=8, stride=8)
    restored = F.interpolate(pooled, size=value.shape[-2:], mode="bilinear", align_corners=False)
    return float((restored.square().mean() / value.square().mean().clamp_min(1e-8)).item())


@torch.inference_mode()
def decode_latent(kv, latent: torch.Tensor) -> Image.Image:
    kv.pipe.vae.to(kv.DEV)
    value = latent.to(kv.pipe.vae.dtype)
    value = value / kv.pipe.vae.config.scaling_factor + kv.pipe.vae.config.shift_factor
    image = kv.pipe.vae.decode(value, return_dict=False)[0]
    return kv.pipe.image_processor.postprocess(image, output_type="pil")[0]


@torch.inference_mode()
def run(args: argparse.Namespace) -> tuple[Image.Image, dict[str, object]]:
    os.environ.setdefault("ZBASE", "models/Z-Image")
    os.environ.setdefault("REF", "/home/ubuntu/smoke_anime/revref_30k/img")
    kv = load_base_probe()
    paths = reference_paths(args, kv) if args.method != "prompt_only" else []
    reference_views: list[ReferenceView] = []
    nuisance_names = [
        item.strip() for item in args.semantic_nuisance_views.split(",") if item.strip()
    ]
    for ref_index, path in enumerate(paths):
        if args.aggregation == "semantic":
            source_views = make_semantic_views(path, args.ref_size, nuisance_names)
        else:
            source_views = [
                (view_name, "anchor", image)
                for view_name, image in make_views(path, args.views_per_reference, args.ref_size)
            ]
        for view_name, role, image in source_views:
            reference_views.append(
                ReferenceView(
                    name=f"r{ref_index}:{view_name}",
                    reference_index=ref_index,
                    transform=view_name,
                    role=role,
                    latent=encode_reference_image(kv, image),
                )
            )

    identity_capsule, identity_capsule_source = load_identity_capsule(args)
    prompt_without_identity = args.prompt or kv.PROMPT
    prompt = prompt_without_identity
    if identity_capsule:
        prompt = f"{prompt}, subject identity: {identity_capsule}"
    if args.residual_contrast == "identity_interaction" and not identity_capsule:
        raise ValueError("identity_interaction residual requires an identity capsule")
    negative = args.negative or kv.NEG
    cap_pos = kv.enc(prompt)
    cap_without_identity = (
        kv.enc(prompt_without_identity)
        if args.method == "residual_tsd" and args.residual_contrast == "identity_interaction"
        else None
    )
    cap_neg = kv.enc(negative)
    cap_capture = kv.enc(kv.PROMPT) if args.method != "prompt_only" else None
    cap_ref = kv.enc("anime illustration of a character") if args.method != "prompt_only" else None
    if args.cpu_offload_static:
        kv.pipe.text_encoder.to("cpu")
        kv.pipe.vae.to("cpu")
        torch.cuda.empty_cache()

    generator = torch.Generator(kv.DEV).manual_seed(args.seed)
    hlat = 2 * (args.size // 16)
    latents = torch.randn((1, 16, hlat, hlat), generator=generator, device=kv.DEV, dtype=torch.float32)

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

    kv.INJ.update({
        "refpos": "strip",
        "route": "off",
        "vnorm": False,
        "ref_len": (args.ref_size // 16) ** 2,
        "tgt_len": (args.size // 16) ** 2,
        "w": float(args.teacher_weight),
        "measure": False,
        "ref_bias": 0.0,
    })
    hidden_layers = parse_layer_spec(args.hidden_support_layers, len(kv.tf.layers))
    hidden_recorder = HiddenStateRecorder(kv.tf, hidden_layers, kv.INJ["tgt_len"])

    def forward(cap, x_image, capture_key: str | None = None):
        hidden_recorder.current_key = capture_key
        try:
            raw = kv.tf(
                x=[[x_image]],
                t=ts,
                cap_feats=[[cap]],
                siglip_feats=[None],
                image_noise_mask=[[1]],
                return_dict=False,
            )[0][0]
            prediction = raw.float().clone()
            del raw
            return prediction
        finally:
            hidden_recorder.current_key = None

    logs: list[dict[str, object]] = []
    for step_index, timestep in enumerate(timesteps):
        if args.verbose_steps:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[step {step_index}] begin allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB", flush=True)
        ts = (1000 - timestep.expand(latents.shape[0])) / 1000
        x_image = latents.to(kv.DT).unsqueeze(2)[0]
        progress = step_index / max(1, len(timesteps) - 1)
        residual_active = (
            args.method == "residual_tsd"
            and args.active_start <= progress <= args.active_end
            and step_index % args.active_stride == 0
            and len(reference_views) > 0
        )
        active = residual_active or args.method == "direct_teacher"

        row: dict[str, object] = {
            "step": step_index,
            "timestep": float(timestep.item()),
            "progress": progress,
            "active": active,
        }
        if args.method == "direct_teacher":
            base_cond = None
            kv.INJ["mode"] = "off"
            base_unc = forward(cap_neg, x_image)
            release_cuda_cache(args.aggressive_memory_cleanup)
            anchor_view = next(view for view in reference_views if view.role == "anchor")
            kv.INJ["mode"] = "capture"
            capture_output = kv.tf(
                x=[[anchor_view.latent, x_image]],
                t=ts,
                cap_feats=[[cap_ref, cap_capture]],
                siglip_feats=[None],
                image_noise_mask=[[0, 1]],
                return_dict=False,
            )
            del capture_output
            release_cuda_cache(args.aggressive_memory_cleanup)
            kv.INJ["mode"] = "inject"
            teacher_cond = forward(cap_pos, x_image)
            release_cuda_cache(args.aggressive_memory_cleanup)
            kv.INJ["mode"] = "off"
            base_pred = teacher_cond + args.cfg * (teacher_cond - base_unc)
            row.update({"view_names": [anchor_view.name], "teacher_deployed": True})
            del teacher_cond
            kv.STORE.clear()
            release_cuda_cache(args.aggressive_memory_cleanup)
        else:
            kv.INJ["mode"] = "off"
            base_capture_key = "base" if residual_active and hidden_layers else None
            base_cond = forward(cap_pos, x_image, base_capture_key)
            release_cuda_cache(args.aggressive_memory_cleanup)
            base_without_identity = None
            if residual_active and args.residual_contrast == "identity_interaction":
                base_without_identity = forward(cap_without_identity, x_image)
                release_cuda_cache(args.aggressive_memory_cleanup)
            base_unc = forward(cap_neg, x_image)
            release_cuda_cache(args.aggressive_memory_cleanup)
            base_pred = base_cond + args.cfg * (base_cond - base_unc)

        if residual_active:
            residual_records: list[tuple[ReferenceView, torch.Tensor]] = []
            view_names = []
            for reference_view in reference_views:
                kv.INJ["mode"] = "capture"
                kv.tf(
                    x=[[reference_view.latent, x_image]],
                    t=ts,
                    cap_feats=[[cap_ref, cap_capture]],
                    siglip_feats=[None],
                    image_noise_mask=[[0, 1]],
                    return_dict=False,
                )
                release_cuda_cache(args.aggressive_memory_cleanup)
                kv.INJ["mode"] = "inject"
                teacher_capture_key = reference_view.name if hidden_layers else None
                teacher_cond = forward(cap_pos, x_image, teacher_capture_key)
                release_cuda_cache(args.aggressive_memory_cleanup)
                teacher_without_identity = None
                if args.residual_contrast == "identity_interaction":
                    teacher_without_identity = forward(cap_without_identity, x_image)
                    release_cuda_cache(args.aggressive_memory_cleanup)
                kv.INJ["mode"] = "off"
                teacher_pred = teacher_cond + args.cfg * (teacher_cond - base_unc)
                if args.residual_contrast == "identity_interaction":
                    residual = identity_interaction_residual(
                        teacher_cond,
                        base_cond,
                        teacher_without_identity,
                        base_without_identity,
                    )
                else:
                    residual = teacher_pred - base_pred
                residual_records.append((reference_view, residual.detach()))
                view_names.append(reference_view.name)
                del teacher_cond, teacher_pred
                if teacher_without_identity is not None:
                    del teacher_without_identity
                kv.STORE.clear()
                release_cuda_cache(args.aggressive_memory_cleanup)

            if args.aggregation == "semantic":
                anchors: dict[int, torch.Tensor] = {
                    view.reference_index: residual
                    for view, residual in residual_records
                    if view.role == "anchor"
                }
                anchor_residuals = [anchors[index] for index in sorted(anchors)]
                nuisance_deltas = [
                    rms_normalize(residual) - rms_normalize(anchors[view.reference_index])
                    for view, residual in residual_records
                    if view.role == "nuisance" and view.reference_index in anchors
                ]
                direction, similarities, extra = semantic_aggregate_residuals(
                    anchor_residuals,
                    nuisance_deltas,
                    residual_filter=args.residual_filter,
                    local_gate_power=args.semantic_local_gate_power,
                    nuisance_ridge=args.semantic_nuisance_ridge,
                    max_nuisance_removal=args.semantic_max_removal,
                )
                consensus_view_names = [
                    view.name for view, _ in residual_records if view.role == "anchor"
                ]
                extra["nuisance_view_names"] = [
                    view.name for view, _ in residual_records if view.role == "nuisance"
                ]
                if hidden_layers:
                    if "base" not in hidden_recorder.records:
                        raise RuntimeError("hidden support requested but base hidden state was not recorded")
                    hidden_deltas = {
                        view.name: {
                            layer_index: hidden - hidden_recorder.records["base"][layer_index]
                            for layer_index, hidden in hidden_recorder.records.get(view.name, {}).items()
                        }
                        for view, _ in residual_records
                    }
                    anchor_names = [
                        view.name for view, _ in residual_records if view.role == "anchor"
                    ]
                    anchor_by_reference = {
                        view.reference_index: view.name
                        for view, _ in residual_records
                        if view.role == "anchor"
                    }
                    nuisance_pairs = [
                        (view.name, anchor_by_reference[view.reference_index])
                        for view, _ in residual_records
                        if view.role == "nuisance" and view.reference_index in anchor_by_reference
                    ]
                    hidden_gate, hidden_stats = hidden_semantic_support_gate(
                        hidden_deltas,
                        anchor_names=anchor_names,
                        nuisance_pairs=nuisance_pairs,
                        output_size=direction.shape[-2:],
                        floor=args.hidden_support_floor,
                    )
                    direction = rms_normalize(
                        direction.float() * hidden_gate.unsqueeze(0).unsqueeze(0)
                    )
                    extra.update(hidden_stats)
            else:
                residuals = [residual for _, residual in residual_records]
                direction, similarities, extra = aggregate_residuals(
                    residuals,
                    args.aggregation,
                    args.residual_filter,
                )
                consensus_view_names = view_names
            consensus = float(np.mean(similarities))
            ratio = lowfreq_ratio(direction)
            accepted = consensus >= args.min_consensus
            if args.max_lowfreq_ratio is not None:
                accepted = accepted and ratio <= args.max_lowfreq_ratio
            scale = 1.0
            if args.taper:
                local = (progress - args.active_start) / max(args.active_end - args.active_start, 1e-8)
                scale = max(0.0, 1.0 - local)

            sigma_index, sigma = resolve_scheduler_sigma(scheduler, timestep, latents)
            update_rms = 0.0
            if accepted and scale > 0:
                update = latent_pseudo_update(
                    direction,
                    sigma,
                    step_size=args.tsd_step,
                    scale=scale,
                )
                latents = latents + update.to(latents.dtype)
                update_rms = float(update.float().square().mean().sqrt().item())
                x_image = latents.to(kv.DT).unsqueeze(2)[0]
                del base_cond, base_unc, base_pred
                if base_without_identity is not None:
                    del base_without_identity
                release_cuda_cache(args.aggressive_memory_cleanup)
                kv.INJ["mode"] = "off"
                base_cond = forward(cap_pos, x_image)
                release_cuda_cache(args.aggressive_memory_cleanup)
                base_unc = forward(cap_neg, x_image)
                release_cuda_cache(args.aggressive_memory_cleanup)
                base_pred = base_cond + args.cfg * (base_cond - base_unc)

            row.update({
                "view_names": view_names,
                "consensus_view_names": consensus_view_names,
                "view_cosine_to_mean": similarities,
                "consensus": consensus,
                "lowfreq_ratio": ratio,
                "accepted": accepted,
                "sigma_index": sigma_index,
                "sigma": float(sigma.item()),
                "taper_scale": scale,
                "update_rms": update_rms,
                **extra,
            })

        kv.INJ["mode"] = "off"
        scheduler_velocity = (-base_pred).unsqueeze(0).squeeze(2)
        latents = scheduler.step(scheduler_velocity.to(torch.float32), timestep, latents, return_dict=False)[0]
        logs.append(row)
        del x_image, base_cond, base_unc, base_pred, scheduler_velocity
        if "base_without_identity" in locals() and base_without_identity is not None:
            del base_without_identity
        if residual_active:
            del residual_records, direction
        hidden_recorder.clear()
        kv.STORE.clear()
        release_cuda_cache(args.aggressive_memory_cleanup)
        if args.verbose_steps:
            allocated = torch.cuda.memory_allocated() / (1024 ** 3)
            reserved = torch.cuda.memory_reserved() / (1024 ** 3)
            print(f"[step {step_index}] end allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB", flush=True)

    image = decode_latent(kv, latents)
    metadata = {
        "method": args.method,
        "teacher_only_reference_injection": args.method == "residual_tsd",
        "scheduler_consumes_reference_prediction": args.method == "direct_teacher",
        "uid": args.uid,
        "references": [str(path) for path in paths],
        "views": [view.name for view in reference_views],
        "view_roles": {view.name: view.role for view in reference_views},
        "prompt": prompt,
        "identity_capsule": identity_capsule,
        "identity_capsule_source": identity_capsule_source,
        "negative": negative,
        "seed": args.seed,
        "size": args.size,
        "steps": args.steps,
        "cfg": args.cfg,
        "teacher_weight": args.teacher_weight,
        "aggregation": args.aggregation,
        "residual_filter": args.residual_filter,
        "residual_contrast": args.residual_contrast,
        "tsd_step": args.tsd_step,
        "active_window": [args.active_start, args.active_end],
        "active_stride": args.active_stride,
        "taper": args.taper,
        "min_consensus": args.min_consensus,
        "max_lowfreq_ratio": args.max_lowfreq_ratio,
        "semantic_nuisance_views": nuisance_names if args.aggregation == "semantic" else [],
        "semantic_local_gate_power": args.semantic_local_gate_power,
        "semantic_nuisance_ridge": args.semantic_nuisance_ridge,
        "semantic_max_removal": args.semantic_max_removal,
        "hidden_support_layers": hidden_layers,
        "hidden_support_floor": args.hidden_support_floor,
        "cpu_offload_static": args.cpu_offload_static,
        "aggressive_memory_cleanup": args.aggressive_memory_cleanup,
        "logs": logs,
    }
    hidden_recorder.close()
    return image, metadata


def main() -> int:
    args = parse_args()
    if args.method == "residual_tsd" and args.views_per_reference < 2 and not args.references:
        print("warning: one reference view gives no cross-view consensus test", flush=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    started = time.time()
    image, metadata = run(args)
    stem = f"u{args.uid}_{args.tag}_s{args.seed}_{args.aggregation}_a{args.views_per_reference}"
    image_path = OUT_ROOT / f"{stem}.png"
    json_path = OUT_ROOT / f"{stem}.json"
    image.save(image_path)
    metadata["wall_time_s"] = time.time() - started
    metadata["image_path"] = str(image_path)
    json_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"saved {image_path}")
    print(f"saved {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Same-space Krea2 identity score distillation probe.

Three trajectories share the same initial noise:

  base:       Krea-2-Turbo text-to-image
  teacher:    Krea-2-Turbo + Identity-Edit LoRA + reference image
  distilled:  text-only attribute scheduler trajectory, with a detached
              teacher-minus-text score pseudo-target optimized inside selected
              denoising steps

The teacher is never used as the distilled trajectory's scheduler prediction.
It only supplies a same-state score residual:

    delta_v = v_teacher(z_sigma) - v_text(z_sigma)
    delta_s = -(1 - sigma) / sigma * delta_v
    x0_base = z_sigma - sigma * stopgrad(v_text)
    x0_target = stopgrad(x0_base + eta * RMSNorm(delta_s))

The packed latent update is projected to both a global RMS ball and a
per-token RMS ball around the post-scheduler base state.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from safetensors.torch import load_file as load_safetensors
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoProcessor

from diffusers import FlowMatchEulerDiscreteScheduler, Krea2Pipeline, Krea2Transformer2DModel
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.embeddings import apply_rotary_emb
from diffusers.models.transformers.transformer_krea2 import Krea2AttnProcessor
from diffusers.pipelines.krea2.pipeline_krea2 import retrieve_timesteps


ROOT = Path("/mnt/local/krea2_edit_serve/models")
BASE_MODEL = ROOT / "Krea-2-Turbo"
LORA_PATH = ROOT / "lora" / "krea2_identity_edit_v1_1.safetensors"
PROCESSOR_MODEL = ROOT / "Qwen3-VL-4B-Instruct"
DEFAULT_REF = Path("/mnt/local/smoke_anime/anime_260623/img/u4229_whiteref.png")
DEFAULT_NEGATIVE_REF = Path(
    "/mnt/local/smoke_anime/anime_260623/img/u1839_whiteref.png"
)
DEFAULT_OUT = Path("/home/ubuntu/chengyanli/items/cherry/out/krea2_tsd")
DTYPE = torch.bfloat16

SCENE_PROMPT = (
    "single character standing on a rain-soaked neon city street at night, "
    "three-quarter body, holding a transparent umbrella, cinematic reflections, "
    "coherent anatomy, face clearly visible, no other people"
)
TEACHER_PROMPT = (
    "Preserve the exact identity, face, spiky light hair with two long front "
    "strands, narrow hooded eyes, black hoop earrings, sharp jaw, and neck "
    "choker of the reference character. Create a single coherent anime character "
    "standing on a rain-soaked neon city street at night, three-quarter body, "
    "holding a transparent umbrella, cinematic reflections, face clearly visible, "
    "no other people, no collage, no split panel."
)
GROUNDED_TEMPLATE = (
    "<|im_start|>system\nDescribe the image by detailing the color, shape, size, "
    "texture, quantity, text, spatial relationships of the objects and background:"
    "<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>"
    "{}<|im_end|>\n<|im_start|>assistant\n"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", default="u4229")
    parser.add_argument("--ref", type=Path, default=DEFAULT_REF)
    parser.add_argument("--negative-ref", type=Path, default=DEFAULT_NEGATIVE_REF)
    parser.add_argument("--lora-path", type=Path, default=LORA_PATH)
    parser.add_argument(
        "--ref-boost",
        type=float,
        default=1.0,
        help=(
            "multiply target-to-reference attention mass before renormalization; "
            "v1.2 recommends about 4, while 1 disables the bias"
        ),
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--seed", type=int, default=777)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--inner-iters", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.03)
    parser.add_argument("--pseudo-step-size", type=float, default=0.2)
    parser.add_argument("--teacher-weight", type=float, default=1.0)
    parser.add_argument("--active-start", type=float, default=0.0)
    parser.add_argument("--active-end", type=float, default=0.75)
    parser.add_argument("--trust-region-rms", type=float, default=0.010)
    parser.add_argument("--local-trust-region-rms", type=float, default=0.008)
    parser.add_argument(
        "--adaptive-trust-ratio",
        type=float,
        default=0.0,
        help=(
            "if >0, use min(global cap, ratio * upcoming delta-sigma * "
            "RMS(teacher-base velocity)); the local cap scales proportionally"
        ),
    )
    parser.add_argument(
        "--normalize",
        choices=("none", "rms", "rms_token"),
        default="rms_token",
    )
    parser.add_argument(
        "--residual-mode",
        choices=(
            "raw",
            "matched_lora",
            "reference_contrast",
            "identity_anchor",
            "identity_stat",
        ),
        default="matched_lora",
        help=(
            "raw subtracts the deployment prediction; matched_lora subtracts "
            "LoRA-off Krea under identical grounded prompt/source conditioning; "
            "identity_anchor blends the edited teacher velocity toward that "
            "LoRA-off reference reconstruction before subtracting deployment; "
            "identity_stat transfers only its per-channel RMS to the spatially "
            "aligned edit residual"
        ),
    )
    parser.add_argument(
        "--identity-anchor-weight",
        type=float,
        default=0.35,
        help=(
            "identity_anchor interpolation weight: 0 is the raw edit teacher, "
            "1 is the LoRA-off reference reconstruction"
        ),
    )
    parser.add_argument(
        "--identity-anchor-power",
        type=float,
        default=0.0,
        help=(
            "time annealing exponent for identity_anchor: effective weight is "
            "weight * progress**power; 0 keeps a constant weight"
        ),
    )
    parser.add_argument("--identity-stat-gain-min", type=float, default=0.5)
    parser.add_argument("--identity-stat-gain-max", type=float, default=2.0)
    parser.add_argument(
        "--distill-roi",
        default=None,
        help="optional normalized x0,y0,x1,y1 packed-token region",
    )
    parser.add_argument(
        "--outside-weight",
        type=float,
        default=1.0,
        help="score direction multiplier outside --distill-roi",
    )
    parser.add_argument("--scene-prompt", default=SCENE_PROMPT)
    parser.add_argument(
        "--controls",
        choices=("all", "none"),
        default="all",
        help=(
            "save all paired control trajectories or only the distilled arm; "
            "the latter is intended for ablations whose controls come from "
            "the locked main run"
        ),
    )
    parser.add_argument(
        "--caption-prompt",
        default=None,
        help=(
            "text-only visible-attribute baseline prompt; defaults to "
            "--scene-prompt and is encoded without any reference image"
        ),
    )
    parser.add_argument("--teacher-prompt", default=TEACHER_PROMPT)
    parser.add_argument(
        "--negative-teacher-prompt",
        default=None,
        help=(
            "grounded instruction paired with --negative-ref; defaults to "
            "--teacher-prompt for backward compatibility"
        ),
    )
    return parser.parse_args()


def convert_lora_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    attn_map = {
        ".attn.wq.": ".attn.to_q.",
        ".attn.wk.": ".attn.to_k.",
        ".attn.wv.": ".attn.to_v.",
        ".attn.wo.": ".attn.to_out.0.",
        ".attn.gate.": ".attn.to_gate.",
        ".mlp.gate.": ".ff.gate.",
        ".mlp.up.": ".ff.up.",
        ".mlp.down.": ".ff.down.",
    }
    converted = {}
    for key, value in state_dict.items():
        new_key = key.removeprefix("diffusion_model.")
        if new_key.startswith("blocks."):
            new_key = new_key.replace("blocks.", "transformer_blocks.", 1)
        if new_key.startswith("txtfusion."):
            new_key = new_key.replace("txtfusion.", "text_fusion.", 1)
        for source, target in attn_map.items():
            new_key = new_key.replace(source, target)
        converted[new_key] = value
    return converted


def grounded_encode(
    pipe: Krea2Pipeline,
    processor,
    instruction: str,
    source: Image.Image,
    grounding_px: int = 768,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = pipe._execution_device
    image = source.convert("RGB")
    if max(image.size) > grounding_px:
        scale = grounding_px / max(image.size)
        image = image.resize(
            (max(16, round(image.width * scale)), max(16, round(image.height * scale))),
            Image.Resampling.LANCZOS,
        )
    text = GROUNDED_TEMPLATE.format(instruction)
    inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to(device)
    kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "pixel_values": inputs.get("pixel_values"),
        "image_grid_thw": inputs.get("image_grid_thw"),
        "output_hidden_states": True,
    }
    if inputs.get("mm_token_type_ids") is not None:
        kwargs["mm_token_type_ids"] = inputs["mm_token_type_ids"]
    outputs = pipe.text_encoder(**kwargs)
    hidden = torch.stack(
        [outputs.hidden_states[index] for index in pipe.text_encoder_select_layers],
        dim=2,
    )
    mask = inputs.get("attention_mask")
    if mask is None:
        mask = torch.ones(hidden.shape[:2], device=device, dtype=torch.bool)
    start = pipe.prompt_template_encode_start_idx
    return hidden[:, start:].to(DTYPE), mask[:, start:].bool()


def encode_source_latent(
    pipe: Krea2Pipeline,
    source: Image.Image,
    height: int,
    width: int,
) -> torch.Tensor:
    pixels = pipe.image_processor.preprocess(source.convert("RGB"), height=height, width=width)
    pixels = pixels.unsqueeze(2).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
    latent = pipe.vae.encode(pixels).latent_dist.mode()
    mean = torch.tensor(pipe.vae.config.latents_mean, device=latent.device, dtype=latent.dtype)
    std = torch.tensor(pipe.vae.config.latents_std, device=latent.device, dtype=latent.dtype)
    mean = mean.view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = std.view(1, pipe.vae.config.z_dim, 1, 1, 1)
    latent = ((latent - mean) / std)[:, :, 0]
    batch, channels, latent_height, latent_width = latent.shape
    return pipe._pack_latents(latent, batch, channels, latent_height, latent_width).to(DTYPE)


def edit_position_ids(
    text_length: int,
    grid_height: int,
    grid_width: int,
    num_sources: int,
    device: torch.device,
) -> torch.Tensor:
    text_ids = torch.zeros(text_length, 3, device=device)

    def image_ids(frame: int) -> torch.Tensor:
        result = torch.zeros(grid_height, grid_width, 3, device=device)
        result[..., 0] = frame
        result[..., 1] = torch.arange(grid_height, device=device)[:, None]
        result[..., 2] = torch.arange(grid_width, device=device)[None, :]
        return result.reshape(grid_height * grid_width, 3)

    return torch.cat(
        [text_ids, *[image_ids(index + 1) for index in range(num_sources)], image_ids(0)],
        dim=0,
    )


class MaskCompatibleKrea2AttnProcessor(Krea2AttnProcessor):
    """Krea2 attention with dense-bias support for GQA.

    Krea2 has 48 query heads and 12 key/value heads. PyTorch's fused attention
    path does not accept that GQA layout together with the dense ref-boost
    bias, so key/value heads are repeated only when a bias is present.
    """

    def __call__(
        self,
        attn,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        image_rotary_emb=None,
    ) -> torch.Tensor:
        if attention_mask is None:
            return super().__call__(
                attn, hidden_states, attention_mask, image_rotary_emb
            )
        query = attn.to_q(hidden_states).unflatten(
            -1, (attn.num_heads, attn.head_dim)
        )
        key = attn.to_k(hidden_states).unflatten(
            -1, (attn.num_kv_heads, attn.head_dim)
        )
        value = attn.to_v(hidden_states).unflatten(
            -1, (attn.num_kv_heads, attn.head_dim)
        )
        gate = attn.to_gate(hidden_states)
        query = attn.norm_q(query)
        key = attn.norm_k(key)
        if image_rotary_emb is not None:
            query = apply_rotary_emb(query, image_rotary_emb, sequence_dim=1)
            key = apply_rotary_emb(key, image_rotary_emb, sequence_dim=1)
        repeat = attn.num_heads // attn.num_kv_heads
        if repeat > 1:
            key = key.repeat_interleave(repeat, dim=2)
            value = value.repeat_interleave(repeat, dim=2)
        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            enable_gqa=False,
            backend=self._attention_backend,
            parallel_config=self._parallel_config,
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states * torch.sigmoid(gate)
        return attn.to_out[0](hidden_states)


def install_mask_compatible_attention(transformer: Krea2Transformer2DModel) -> None:
    for block in transformer.transformer_blocks:
        block.attn.set_processor(MaskCompatibleKrea2AttnProcessor())


def ref_boost_bias(
    text_length: int,
    source_length: int,
    target_length: int,
    ref_boost: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor | None:
    """Bias target queries toward clean reference keys by log(ref_boost)."""
    if ref_boost <= 0.0:
        raise ValueError("--ref-boost must be positive")
    if source_length == 0 or ref_boost == 1.0:
        return None
    total_length = text_length + source_length + target_length
    bias = torch.zeros(
        1, 1, total_length, total_length, device=device, dtype=dtype
    )
    target_start = text_length + source_length
    bias[:, :, target_start:, text_length:target_start] = math.log(ref_boost)
    return bias


def transformer_forward(
    transformer: Krea2Transformer2DModel,
    latents: torch.Tensor,
    source_latents: torch.Tensor,
    prompt_embeds: torch.Tensor,
    prompt_mask: torch.Tensor,
    timestep: torch.Tensor,
    position_ids: torch.Tensor,
    ref_boost: float = 1.0,
) -> torch.Tensor:
    """Worker-compatible Krea2 forward, including optional prepended source tokens."""
    combined = torch.cat([source_latents, latents], dim=1)
    temb = transformer.time_embed(timestep, dtype=latents.dtype)
    temb_mod = transformer.time_mod_proj(F.gelu(temb, approximate="tanh"))
    attention_mask = prompt_mask[:, None, None, :] if prompt_mask is not None else None
    encoded = transformer.text_fusion(prompt_embeds, attention_mask=attention_mask)
    encoded = transformer.txt_in(encoded)
    image_hidden = transformer.img_in(combined)
    hidden = torch.cat([encoded, image_hidden], dim=1)
    rotary = transformer.rotary_emb(position_ids)
    attention_bias = ref_boost_bias(
        encoded.shape[1],
        source_latents.shape[1],
        latents.shape[1],
        ref_boost,
        device=hidden.device,
        dtype=hidden.dtype,
    )
    with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
        for block in transformer.transformer_blocks:
            hidden = block(hidden, temb_mod, rotary, attention_bias)
    text_length = encoded.shape[1]
    hidden = hidden[:, text_length:][:, -latents.shape[1] :]
    return transformer.final_layer(hidden, temb)


def score_delta_from_velocity_delta(
    velocity_delta: torch.Tensor,
    sigma: torch.Tensor | float,
    eps: float = 1e-4,
) -> torch.Tensor:
    sigma_tensor = torch.as_tensor(
        sigma, device=velocity_delta.device, dtype=torch.float32
    ).clamp_min(eps)
    return -((1.0 - sigma_tensor) / sigma_tensor) * velocity_delta.float()


def normalize_field(value: torch.Tensor, mode: str, eps: float = 1e-6) -> torch.Tensor:
    if mode == "none":
        return value.float()
    if mode == "rms_token":
        rms = value.float().square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
    elif mode == "rms":
        flat = value.float().reshape(value.shape[0], -1)
        rms = flat.square().mean(dim=-1, keepdim=True).sqrt().clamp_min(eps)
        rms = rms.reshape(value.shape[0], 1, 1)
    else:
        raise ValueError(mode)
    return value.float() / rms


def identity_anchored_velocity(
    teacher_velocity: torch.Tensor,
    reference_velocity: torch.Tensor,
    weight: float,
) -> torch.Tensor:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("--identity-anchor-weight must be in [0,1]")
    return teacher_velocity.float().lerp(reference_velocity.float(), weight)


def identity_stat_transport(
    edit_delta: torch.Tensor,
    reference_delta: torch.Tensor,
    weight: float,
    gain_min: float = 0.5,
    gain_max: float = 2.0,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Transfer diagonal channel energy without copying reference token layout."""
    if not 0.0 <= weight <= 1.0:
        raise ValueError("--identity-anchor-weight must be in [0,1]")
    if not 0.0 < gain_min <= gain_max:
        raise ValueError("identity-stat gain bounds must satisfy 0 < min <= max")
    edit = edit_delta.float()
    reference = reference_delta.float()
    edit_rms = edit.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    reference_rms = (
        reference.square().mean(dim=1, keepdim=True).sqrt().clamp_min(eps)
    )
    gain = (reference_rms / edit_rms).clamp(gain_min, gain_max)
    transported = edit * gain
    return edit.lerp(transported, weight)


def parse_roi(value: str | None) -> tuple[float, float, float, float] | None:
    if value is None:
        return None
    parts = tuple(float(part) for part in value.split(","))
    if len(parts) != 4:
        raise ValueError("--distill-roi expects x0,y0,x1,y1")
    x0, y0, x1, y1 = parts
    if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
        raise ValueError("--distill-roi coordinates must form a box in [0,1]")
    return parts


def packed_roi_gate(
    grid_height: int,
    grid_width: int,
    roi: tuple[float, float, float, float] | None,
    outside_weight: float,
    *,
    device: torch.device,
) -> torch.Tensor:
    if not 0.0 <= outside_weight <= 1.0:
        raise ValueError("--outside-weight must be in [0,1]")
    if roi is None:
        return torch.ones(1, grid_height * grid_width, 1, device=device)
    x0, y0, x1, y1 = roi
    xs = (torch.arange(grid_width, device=device, dtype=torch.float32) + 0.5) / grid_width
    ys = (torch.arange(grid_height, device=device, dtype=torch.float32) + 0.5) / grid_height
    inside = (
        (xs[None, :] >= x0)
        & (xs[None, :] <= x1)
        & (ys[:, None] >= y0)
        & (ys[:, None] <= y1)
    )
    gate = torch.where(
        inside,
        torch.ones_like(inside, dtype=torch.float32),
        torch.full_like(inside, outside_weight, dtype=torch.float32),
    )
    return gate.reshape(1, grid_height * grid_width, 1)


def project_trust_region(
    candidate: torch.Tensor,
    anchor: torch.Tensor,
    global_radius: float,
    local_radius: float,
) -> tuple[torch.Tensor, float, float]:
    """Project packed BNC latents into global and per-token RMS balls."""
    delta = candidate.float() - anchor.float()
    global_rms = delta.square().mean().sqrt()
    global_scale = min(
        1.0, global_radius / max(float(global_rms.detach().item()), 1e-12)
    )
    delta = delta * global_scale
    local_rms = delta.square().mean(dim=-1, keepdim=True).sqrt()
    local_scale = (local_radius / local_rms.clamp_min(1e-12)).clamp(max=1.0)
    delta = delta * local_scale
    projected = anchor.float() + delta
    return (
        projected.to(anchor.dtype),
        float(delta.square().mean().sqrt().item()),
        float(delta.square().mean(dim=-1).sqrt().max().item()),
    )


def adaptive_trust_radii(
    velocity_delta: torch.Tensor,
    delta_sigma: float,
    ratio: float,
    global_cap: float,
    local_cap: float,
) -> tuple[float, float]:
    """Scale the trust ball by the teacher's one-Euler-step displacement."""
    if ratio <= 0.0:
        return global_cap, local_cap
    velocity_rms = float(
        velocity_delta.detach().float().square().mean().sqrt().item()
    )
    global_radius = min(global_cap, ratio * abs(delta_sigma) * velocity_rms)
    local_to_global = local_cap / max(global_cap, 1e-12)
    local_radius = min(local_cap, global_radius * local_to_global)
    return max(global_radius, 1e-6), max(local_radius, 1e-6)


def make_schedule(pipe: Krea2Pipeline, steps: int, device: torch.device):
    scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipe.scheduler.config)
    sigmas = np.linspace(1.0, 1.0 / steps, steps)
    timesteps, _ = retrieve_timesteps(scheduler, steps, device, sigmas=sigmas, mu=1.15)
    scheduler.set_begin_index(0)
    return scheduler, timesteps


def decode_latents(
    pipe: Krea2Pipeline,
    latents: torch.Tensor,
    height: int,
    width: int,
) -> Image.Image:
    unpacked = pipe._unpack_latents(latents, height, width).to(pipe.vae.dtype)
    mean = torch.tensor(pipe.vae.config.latents_mean, device=unpacked.device, dtype=unpacked.dtype)
    std = torch.tensor(pipe.vae.config.latents_std, device=unpacked.device, dtype=unpacked.dtype)
    mean = mean.view(1, pipe.vae.config.z_dim, 1, 1, 1)
    std = std.view(1, pipe.vae.config.z_dim, 1, 1, 1)
    decoded = pipe.vae.decode(unpacked * std + mean, return_dict=False)[0][:, :, 0]
    return pipe.image_processor.postprocess(decoded, output_type="pil")[0]


def main() -> int:
    args = parse_args()
    distill_roi = parse_roi(args.distill_roi)
    if not 0.0 <= args.identity_anchor_weight <= 1.0:
        raise ValueError("--identity-anchor-weight must be in [0,1]")
    if args.identity_anchor_power < 0.0:
        raise ValueError("--identity-anchor-power must be non-negative")
    if not 0.0 < args.identity_stat_gain_min <= args.identity_stat_gain_max:
        raise ValueError(
            "--identity-stat-gain-min/max must satisfy 0 < min <= max"
        )
    if not args.lora_path.is_file():
        raise FileNotFoundError(args.lora_path)
    if args.width % 16 or args.height % 16:
        raise ValueError("width and height must be divisible by 16")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    started = time.time()

    print("[load] shared Krea2 pipeline (unfused base)", flush=True)
    pipe = Krea2Pipeline.from_pretrained(BASE_MODEL, torch_dtype=DTYPE).to(device)
    pipe.set_progress_bar_config(disable=True)
    base_transformer = pipe.transformer

    print("[load] second Krea2 transformer + Identity-Edit LoRA teacher", flush=True)
    teacher_transformer = Krea2Transformer2DModel.from_pretrained(
        BASE_MODEL, subfolder="transformer", torch_dtype=DTYPE
    ).to(device)
    lora = convert_lora_keys(load_safetensors(args.lora_path))
    teacher_transformer.load_lora_adapter(lora, adapter_name="identity", prefix=None)
    teacher_transformer.set_adapters("identity", weights=1.0)
    teacher_transformer.fuse_lora(lora_scale=1.0)
    teacher_transformer.unload_lora()
    teacher_transformer.eval()
    base_transformer.eval()
    install_mask_compatible_attention(teacher_transformer)
    install_mask_compatible_attention(base_transformer)
    processor = AutoProcessor.from_pretrained(PROCESSOR_MODEL)

    reference = Image.open(args.ref).convert("RGB")
    negative_reference = Image.open(args.negative_ref).convert("RGB")
    with torch.inference_mode():
        base_embeds, base_mask = pipe.encode_prompt(args.scene_prompt, device=device)
        caption_prompt = (
            args.caption_prompt
            if args.caption_prompt is not None
            else args.scene_prompt
        )
        # This arm must remain genuinely text-only. Reusing teacher_embeds here
        # leaks reference-image tokens from grounded_encode into the baseline.
        caption_embeds, caption_mask = pipe.encode_prompt(
            caption_prompt, device=device
        )
        teacher_embeds, teacher_mask = grounded_encode(
            pipe, processor, args.teacher_prompt, reference
        )
        source_latents = encode_source_latent(pipe, reference, args.height, args.width)
        negative_teacher_prompt = (
            args.negative_teacher_prompt
            if args.negative_teacher_prompt is not None
            else args.teacher_prompt
        )
        negative_embeds, negative_mask = grounded_encode(
            pipe, processor, negative_teacher_prompt, negative_reference
        )
        negative_source_latents = encode_source_latent(
            pipe, negative_reference, args.height, args.width
        )

    grid_height = args.height // (pipe.vae_scale_factor * pipe.patch_size)
    grid_width = args.width // (pipe.vae_scale_factor * pipe.patch_size)
    base_position_ids = edit_position_ids(
        base_embeds.shape[1], grid_height, grid_width, 0, device
    )
    teacher_position_ids = edit_position_ids(
        teacher_embeds.shape[1], grid_height, grid_width, 1, device
    )
    caption_base_position_ids = edit_position_ids(
        caption_embeds.shape[1], grid_height, grid_width, 0, device
    )
    negative_position_ids = edit_position_ids(
        negative_embeds.shape[1], grid_height, grid_width, 1, device
    )
    distill_gate = packed_roi_gate(
        grid_height,
        grid_width,
        distill_roi,
        args.outside_weight,
        device=device,
    )
    empty_source = source_latents[:, :0]
    channels = base_transformer.config.in_channels // (pipe.patch_size**2)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    initial = pipe.prepare_latents(
        1, channels, args.height, args.width, DTYPE, device, generator, None
    )

    def predict(
        model: Krea2Transformer2DModel,
        latent: torch.Tensor,
        source: torch.Tensor,
        embeds: torch.Tensor,
        mask: torch.Tensor,
        position_ids: torch.Tensor,
        timestep: torch.Tensor,
        scheduler,
        ref_boost: float = 1.0,
    ) -> torch.Tensor:
        normalized_timestep = (
            timestep / scheduler.config.num_train_timesteps
        ).expand(latent.shape[0]).to(latent.dtype)
        return transformer_forward(
            model,
            latent,
            source,
            embeds,
            mask,
            normalized_timestep,
            position_ids,
            ref_boost,
        )

    @torch.inference_mode()
    def run_plain(mode: str) -> torch.Tensor:
        scheduler, timesteps = make_schedule(pipe, args.steps, device)
        latent = initial.clone()
        for timestep in timesteps:
            if mode == "teacher":
                velocity = predict(
                    teacher_transformer,
                    latent,
                    source_latents,
                    teacher_embeds,
                    teacher_mask,
                    teacher_position_ids,
                    timestep,
                    scheduler,
                    args.ref_boost,
                )
            elif mode == "matched_control":
                velocity = predict(
                    base_transformer,
                    latent,
                    source_latents,
                    teacher_embeds,
                    teacher_mask,
                    teacher_position_ids,
                    timestep,
                    scheduler,
                    args.ref_boost,
                )
            elif mode == "negative_teacher":
                velocity = predict(
                    teacher_transformer,
                    latent,
                    negative_source_latents,
                    negative_embeds,
                    negative_mask,
                    negative_position_ids,
                    timestep,
                    scheduler,
                    args.ref_boost,
                )
            elif mode == "base":
                velocity = predict(
                    base_transformer,
                    latent,
                    empty_source,
                    base_embeds,
                    base_mask,
                    base_position_ids,
                    timestep,
                    scheduler,
                )
            elif mode == "caption_base":
                velocity = predict(
                    base_transformer,
                    latent,
                    empty_source,
                    caption_embeds,
                    caption_mask,
                    caption_base_position_ids,
                    timestep,
                    scheduler,
                )
            else:
                raise ValueError(mode)
            latent = scheduler.step(velocity, timestep, latent, return_dict=False)[0]
        return latent

    plain_finals = {}
    if args.controls == "all":
        print(
            "[run] base, caption-only base, LoRA-off matched control, "
            "and direct teacher controls",
            flush=True,
        )
        for mode in (
            "base",
            "caption_base",
            "matched_control",
            "negative_teacher",
            "teacher",
        ):
            plain_finals[mode] = run_plain(mode)

    print("[run] inner-loop same-space score distillation", flush=True)
    scheduler, timesteps = make_schedule(pipe, args.steps, device)
    distilled = initial.clone()
    diagnostics = []
    for outer_index, timestep in enumerate(timesteps):
        with torch.inference_mode():
            base_velocity = predict(
                base_transformer,
                distilled,
                empty_source,
                caption_embeds,
                caption_mask,
                caption_base_position_ids,
                timestep,
                scheduler,
            )
            distilled = scheduler.step(
                base_velocity, timestep, distilled, return_dict=False
            )[0]

        next_index = outer_index + 1
        if next_index >= len(timesteps):
            continue
        progress = next_index / max(len(timesteps) - 1, 1)
        if not (args.active_start <= progress <= args.active_end):
            continue

        anchor = distilled.detach().clone()
        next_timestep = timesteps[next_index]
        sigma = scheduler.sigmas[next_index].to(device=device, dtype=torch.float32)
        following_sigma = scheduler.sigmas[next_index + 1].to(
            device=device, dtype=torch.float32
        )
        delta_sigma = float((sigma - following_sigma).abs().item())
        inner_records = []
        for inner_index in range(args.inner_iters):
            latent_parameter = distilled.detach().clone().requires_grad_(True)
            with torch.no_grad():
                base_velocity = predict(
                    base_transformer,
                    latent_parameter,
                    empty_source,
                    caption_embeds,
                    caption_mask,
                    caption_base_position_ids,
                    next_timestep,
                    scheduler,
                )
                teacher_velocity = predict(
                    teacher_transformer,
                    latent_parameter,
                    source_latents,
                    teacher_embeds,
                    teacher_mask,
                    teacher_position_ids,
                    next_timestep,
                    scheduler,
                    args.ref_boost,
                )
                if args.residual_mode in (
                    "matched_lora",
                    "identity_anchor",
                    "identity_stat",
                ):
                    reference_velocity = predict(
                        base_transformer,
                        latent_parameter,
                        source_latents,
                        teacher_embeds,
                        teacher_mask,
                        teacher_position_ids,
                        next_timestep,
                        scheduler,
                        args.ref_boost,
                    )
                    if args.residual_mode == "identity_anchor":
                        effective_anchor_weight = (
                            args.identity_anchor_weight
                            * progress**args.identity_anchor_power
                        )
                        target_velocity = identity_anchored_velocity(
                            teacher_velocity,
                            reference_velocity,
                            effective_anchor_weight,
                        )
                        comparison_velocity = base_velocity
                    elif args.residual_mode == "identity_stat":
                        effective_anchor_weight = (
                            args.identity_anchor_weight
                            * progress**args.identity_anchor_power
                        )
                        delta_velocity = identity_stat_transport(
                            teacher_velocity.float() - base_velocity.float(),
                            reference_velocity.float() - base_velocity.float(),
                            effective_anchor_weight,
                            args.identity_stat_gain_min,
                            args.identity_stat_gain_max,
                        )
                        target_velocity = None
                        comparison_velocity = None
                    else:
                        target_velocity = teacher_velocity
                        comparison_velocity = reference_velocity
                elif args.residual_mode == "reference_contrast":
                    comparison_velocity = predict(
                        teacher_transformer,
                        latent_parameter,
                        negative_source_latents,
                        negative_embeds,
                        negative_mask,
                        negative_position_ids,
                        next_timestep,
                        scheduler,
                        args.ref_boost,
                    )
                    target_velocity = teacher_velocity
                else:
                    target_velocity = teacher_velocity
                    comparison_velocity = base_velocity
                if args.residual_mode != "identity_stat":
                    delta_velocity = (
                        target_velocity.float() - comparison_velocity.float()
                    )
                delta_score = score_delta_from_velocity_delta(delta_velocity, sigma)
                direction = normalize_field(delta_score, args.normalize) * distill_gate
                global_radius, local_radius = adaptive_trust_radii(
                    delta_velocity,
                    delta_sigma,
                    args.adaptive_trust_ratio,
                    args.trust_region_rms,
                    args.local_trust_region_rms,
                )

            x0_base = latent_parameter.float() - sigma * base_velocity.detach().float()
            x0_target = (
                x0_base.detach()
                + args.pseudo_step_size * args.teacher_weight * direction.detach()
            )
            loss = 0.5 * F.mse_loss(x0_base, x0_target, reduction="mean")
            gradient = torch.autograd.grad(loss, latent_parameter)[0]
            candidate = (
                latent_parameter.float()
                - args.lr * latent_parameter.numel() * gradient.float()
            ).to(distilled.dtype)
            distilled, global_rms, max_token_rms = project_trust_region(
                candidate,
                anchor,
                global_radius,
                local_radius,
            )
            inner_records.append(
                {
                    "inner": inner_index,
                    "loss": float(loss.item()),
                    "delta_v_rms": float(delta_velocity.square().mean().sqrt().item()),
                    "delta_s_rms": float(delta_score.square().mean().sqrt().item()),
                    "delta_sigma": delta_sigma,
                    "global_radius": global_radius,
                    "local_radius": local_radius,
                    "anchor_global_rms": global_rms,
                    "anchor_max_token_rms": max_token_rms,
                    "identity_anchor_weight": (
                        effective_anchor_weight
                        if args.residual_mode in ("identity_anchor", "identity_stat")
                        else 0.0
                    ),
                }
            )
        diagnostics.append(
            {
                "outer_after_step": outer_index,
                "next_timestep": float(next_timestep.item()),
                "sigma": float(sigma.item()),
                "progress": progress,
                "inner": inner_records,
            }
        )
        print(
            f"  outer={outer_index} sigma={float(sigma):.4f} "
            f"trust={inner_records[-1]['anchor_global_rms']:.5f}/"
            f"{inner_records[-1]['anchor_max_token_rms']:.5f}",
            flush=True,
        )

    residual_tag = args.residual_mode
    if args.residual_mode in ("identity_anchor", "identity_stat"):
        residual_tag += (
            f"{args.identity_anchor_weight:g}p{args.identity_anchor_power:g}"
        )
    if args.residual_mode == "identity_stat":
        residual_tag += (
            f"c{args.identity_stat_gain_min:g}-{args.identity_stat_gain_max:g}"
        )
    stem = (
        f"{args.uid}_s{args.seed}_{args.width}x{args.height}_"
        f"i{args.inner_iters}_p{args.pseudo_step_size:g}_"
        f"g{args.trust_region_rms:g}_l{args.local_trust_region_rms:g}_"
        f"a{args.adaptive_trust_ratio:g}_rb{args.ref_boost:g}_"
        f"{args.normalize}_{residual_tag}_dcap_"
        f"roi{args.distill_roi or 'full'}_o{args.outside_weight:g}"
        f"{'_cnone' if args.controls == 'none' else ''}"
    )
    with torch.inference_mode():
        outputs = {
            label: decode_latents(pipe, latent, args.height, args.width)
            for label, latent in plain_finals.items()
        }
        outputs["distilled"] = decode_latents(
            pipe, distilled, args.height, args.width
        )
    paths = {}
    for label, image in outputs.items():
        path = args.out_dir / f"{stem}_{label}.png"
        image.save(path)
        paths[label] = str(path)
        print(f"[save] {label}: {path}", flush=True)

    formula_by_mode = {
        "raw": "delta_s=-(1-sigma)/sigma*(v_teacher-v_text)",
        "matched_lora": (
            "delta_s=-(1-sigma)/sigma*"
            "(v_teacher-v_lora_off_reference)"
        ),
        "reference_contrast": (
            "delta_s=-(1-sigma)/sigma*"
            "(v_correct_reference-v_wrong_reference)"
        ),
        "identity_anchor": (
            "delta_s=-(1-sigma)/sigma*"
            "(((1-w_t)*v_teacher+w_t*v_lora_off_reference)-v_text)"
        ),
        "identity_stat": (
            "d_edit=v_teacher-v_text; d_ref=v_lora_off_reference-v_text; "
            "delta_s=-(1-sigma)/sigma*((1-w_t)*d_edit+w_t*d_edit*"
            "clip(RMS_tokens(d_ref)/RMS_tokens(d_edit),g_min,g_max))"
        ),
    }
    metadata = {
        "method": "krea2_same_space_identity_score_distillation",
        "formula": formula_by_mode[args.residual_mode],
        "teacher_scheduler_usage": "none_in_distilled_trajectory",
        "deployment_conditioning": "text_only_automatic_visible_attributes",
        "controls": args.controls,
        "base_model": str(BASE_MODEL),
        "teacher_lora": str(args.lora_path),
        "ref_boost": args.ref_boost,
        "reference": str(args.ref),
        "uid": args.uid,
        "negative_reference": str(args.negative_ref),
        "seed": args.seed,
        "size": [args.width, args.height],
        "steps": args.steps,
        "inner_iters": args.inner_iters,
        "lr": args.lr,
        "pseudo_step_size": args.pseudo_step_size,
        "teacher_weight": args.teacher_weight,
        "residual_mode": args.residual_mode,
        "identity_anchor_weight": args.identity_anchor_weight,
        "identity_anchor_power": args.identity_anchor_power,
        "identity_stat_gain_bounds": [
            args.identity_stat_gain_min,
            args.identity_stat_gain_max,
        ],
        "active_window": [args.active_start, args.active_end],
        "normalize": args.normalize,
        "trust_region_rms": args.trust_region_rms,
        "local_trust_region_rms": args.local_trust_region_rms,
        "adaptive_trust_ratio": args.adaptive_trust_ratio,
        "distill_roi": distill_roi,
        "outside_weight": args.outside_weight,
        "scene_prompt": args.scene_prompt,
        "caption_prompt": caption_prompt,
        "caption_conditioning": "text_only_no_reference_tokens_or_latents",
        "teacher_prompt": args.teacher_prompt,
        "negative_teacher_prompt": negative_teacher_prompt,
        "outputs": paths,
        "diagnostics": diagnostics,
        "elapsed_seconds": time.time() - started,
    }
    metadata_path = args.out_dir / f"{stem}.json"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    print(f"[save] metadata: {metadata_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

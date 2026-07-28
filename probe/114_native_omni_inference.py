#!/usr/bin/env python3
"""Direct native-Omni inference diagnostic for the frozen Z-Image checkpoint.

This deliberately bypasses score distillation and the legacy K/V injector.
Clean reference latents and the noisy target latent are presented jointly on
every scheduler step using the official vision-token prompt segmentation. The
result answers whether the checkpoint's native reference-conditioned inference
path itself carries the requested identity.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
TSD_SCRIPT = ROOT / "probe" / "112_true_score_tsd.py"
OUT_ROOT = ROOT / "out" / "native_omni"


def load_tsd():
    spec = importlib.util.spec_from_file_location("cherry_native_omni_tsd", TSD_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {TSD_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--references", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--negative", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--ref-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--cfg", type=float, default=5.0)
    parser.add_argument("--tag", default="direct")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    tsd = load_tsd()
    impl = tsd.load_impl()
    os.environ.setdefault("ZBASE", "/mnt/local/yzh/models/Z-Image")
    os.environ.setdefault("REF", "/home/ubuntu/smoke_anime/revref_30k/img")
    kv = impl.load_base_probe()
    reference_paths = [
        Path(value.strip()).expanduser().resolve()
        for value in args.references.split(",")
        if value.strip()
    ]
    if not reference_paths:
        raise ValueError("at least one reference is required")
    for path in reference_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
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
        cap_pos = tsd.encode_omni_prompt(
            kv.pipe,
            args.prompt,
            device=kv.DEV,
            num_condition_images=len(reference_latents),
        )
        cap_neg = tsd.encode_omni_prompt(
            kv.pipe,
            negative,
            device=kv.DEV,
            num_condition_images=len(reference_latents),
        )
    kv.pipe.text_encoder.to("cpu")
    torch.cuda.empty_cache()

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
        timesteps, _ = kv.retrieve_timesteps(
            scheduler,
            args.steps,
            kv.DEV,
            sigmas=None,
            mu=mu,
        )
    except Exception:
        timesteps, _ = kv.retrieve_timesteps(
            scheduler,
            args.steps,
            kv.DEV,
            sigmas=None,
        )

    kv.INJ["mode"] = "off"
    with torch.no_grad():
        for index, timestep in enumerate(timesteps):
            normalized_timestep = (
                1000 - timestep.expand(latents.shape[0])
            ) / 1000
            target = latents.to(kv.DT).unsqueeze(2)[0]
            model_input = [[*reference_latents, target]]
            noise_mask = [
                [0] * len(reference_latents) + [1]
            ]
            positive = kv.tf(
                x=model_input,
                t=normalized_timestep,
                cap_feats=[cap_pos],
                siglip_feats=[None],
                image_noise_mask=noise_mask,
                return_dict=False,
            )[0][0].float()
            negative_prediction = kv.tf(
                x=model_input,
                t=normalized_timestep,
                cap_feats=[cap_neg],
                siglip_feats=[None],
                image_noise_mask=noise_mask,
                return_dict=False,
            )[0][0].float()
            guided = tsd.classifier_free_guided_prediction(
                positive,
                negative_prediction,
                args.cfg,
            )
            velocity = tsd.prediction_to_velocity(guided)
            latents = scheduler.step(
                velocity.to(torch.float32),
                timestep,
                latents,
                return_dict=False,
            )[0]
            if args.verbose:
                print(
                    f"step={index:02d} t={float(normalized_timestep[0]):.4f} "
                    f"pred_rms={tsd.tensor_rms(guided):.5f}",
                    flush=True,
                )

        image = impl.decode_latent(kv, latents)

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    stem = f"u{args.uid}_{args.tag}_s{args.seed}_n{args.steps}"
    image_path = OUT_ROOT / f"{stem}.png"
    metadata_path = OUT_ROOT / f"{stem}.json"
    image.save(image_path)
    metadata_path.write_text(
        json.dumps(
            {
                "method": "direct_native_omni_scheduler",
                "uid": args.uid,
                "references": [str(path) for path in reference_paths],
                "prompt": args.prompt,
                "negative": negative,
                "seed": args.seed,
                "size": args.size,
                "ref_size": args.ref_size,
                "steps": args.steps,
                "cfg": args.cfg,
                "siglip_features": None,
                "prompt_protocol": "official_zimage_omni_vision_token_segments",
                "legacy_kv_injection": False,
                "image_path": str(image_path),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved {image_path}")
    print(f"saved {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Trust-region latent optimizer for the 112 score-distillation experiment.

The pseudo-target loss in 112 uses a mean reduction over every latent element.
For a detached score target its useful gradient is therefore divided by the
latent element count. AdamW largely cancels that scale on its first steps and
turns the update into a coordinate-wise sign step.

This wrapper keeps the audited teacher/scheduler implementation unchanged and
replaces only its per-outer latent optimizer with:

    delta = -lr * numel(latent) * grad(mean_loss)
    z <- project_RMS(z + delta, anchor=z_after_scheduler, radius=rho)

The numel correction restores the intended pseudo-step magnitude, while the
RMS trust region prevents repeated inner updates from leaving the local base
trajectory too far at any one scheduler state.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
BASE_SCRIPT = ROOT / "probe" / "112_true_score_tsd.py"
OUT_ROOT = ROOT / "out" / "true_score_tsd"
ORIGINAL_ADAMW = torch.optim.AdamW


def load_base():
    spec = importlib.util.spec_from_file_location("cherry_true_score_tsd_112", BASE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {BASE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class MeanCorrectedTrustRegionSGD(torch.optim.Optimizer):
    """SGD whose step scale matches a mean-reduced detached pseudo-target."""

    trust_region_rms: float = 0.003
    local_trust_region_rms: float | None = None
    correction_power: float = 1.0

    def __init__(self, params, lr: float, weight_decay: float = 0.0):
        parameter_list = list(params)
        if weight_decay != 0.0:
            raise ValueError("trust-region latent optimizer requires weight_decay=0")
        super().__init__(parameter_list, {"lr": lr})
        self._anchors = {
            id(parameter): parameter.detach().clone()
            for parameter in parameter_list
        }

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            learning_rate = float(group["lr"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                gradient = parameter.grad
                if not torch.isfinite(gradient).all():
                    raise FloatingPointError("non-finite latent gradient")

                mean_correction = float(parameter.numel()) ** self.correction_power
                parameter.add_(gradient, alpha=-learning_rate * mean_correction)

                anchor = self._anchors[id(parameter)]
                delta = parameter - anchor
                delta_rms = delta.float().square().mean().sqrt()
                radius = float(self.trust_region_rms)
                if float(delta_rms.item()) > radius:
                    parameter.copy_(anchor + delta * (radius / delta_rms.clamp_min(1e-12)))

                local_radius = self.local_trust_region_rms
                if local_radius is not None:
                    if parameter.ndim != 4:
                        raise ValueError(
                            "local trust region expects a BCHW latent, found "
                            f"shape={tuple(parameter.shape)}"
                        )
                    delta = parameter - anchor
                    local_rms = delta.float().square().mean(dim=1, keepdim=True).sqrt()
                    local_scale = (
                        float(local_radius) / local_rms.clamp_min(1e-12)
                    ).clamp(max=1.0)
                    parameter.copy_(anchor + delta * local_scale.to(delta.dtype))
        return loss


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--trust-region-rms", type=float, default=0.003)
    parser.add_argument(
        "--local-trust-region-rms",
        type=float,
        default=None,
        help="optional per-spatial-location RMS radius across latent channels",
    )
    parser.add_argument(
        "--mean-correction-power",
        type=float,
        default=1.0,
        help="1 restores the full element-count scale; values below 1 are more conservative",
    )
    args, remaining = parser.parse_known_args()
    if args.trust_region_rms <= 0:
        raise ValueError("--trust-region-rms must be positive")
    if args.local_trust_region_rms is not None and args.local_trust_region_rms <= 0:
        raise ValueError("--local-trust-region-rms must be positive")
    if not 0.0 <= args.mean_correction_power <= 1.0:
        raise ValueError("--mean-correction-power must be in [0, 1]")
    return args, remaining


def annotate_new_metadata(started: float, wrapper_args: argparse.Namespace) -> None:
    candidates = [
        path
        for path in OUT_ROOT.glob("*.json")
        if path.stat().st_mtime >= started
    ]
    if len(candidates) != 1:
        raise RuntimeError(
            "expected exactly one new metadata file, found "
            f"{len(candidates)}: {[path.name for path in candidates]}"
        )
    metadata_path = candidates[0]
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["latent_optimizer"] = "mean_corrected_trust_region_sgd"
    metadata["trust_region_rms"] = wrapper_args.trust_region_rms
    metadata["local_trust_region_rms"] = wrapper_args.local_trust_region_rms
    metadata["mean_correction_power"] = wrapper_args.mean_correction_power
    metadata["base_experiment_script"] = str(BASE_SCRIPT)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"annotated {metadata_path}")


def main() -> int:
    wrapper_args, base_args = parse_wrapper_args()
    MeanCorrectedTrustRegionSGD.trust_region_rms = wrapper_args.trust_region_rms
    MeanCorrectedTrustRegionSGD.local_trust_region_rms = (
        wrapper_args.local_trust_region_rms
    )
    MeanCorrectedTrustRegionSGD.correction_power = wrapper_args.mean_correction_power

    base = load_base()
    base.torch.optim.AdamW = MeanCorrectedTrustRegionSGD
    sys.argv = [sys.argv[0], *base_args]
    started = time.time()
    result = int(base.main())
    annotate_new_metadata(started, wrapper_args)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

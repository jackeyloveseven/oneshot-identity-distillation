#!/usr/bin/env python3
"""Scale the existing head/token/time matched-control probes to holdout30 x 2.

Head/token protocol reproduces the prior quantitative probe's deliberately
high-defect setting: lowfreq k=8, band 10-25, 512px. Time protocol reproduces
the prior production-path screen: strip, band 21-25, 512px. All candidates use
the same held-out prompt_B and fixed seeds 0/1. The script is resumable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT_ROOT = Path("/mnt/local/cherry_out")
HEAD_OUT = OUT_ROOT / "localization_headtoken_q60"
TIME_OUT = OUT_ROOT / "localization_time_q60"
PROTOCOL_OUT = DATA / "localization_sweeps_protocol.json"

spec = importlib.util.spec_from_file_location("kv", ROOT / "probe" / "30_kv_inject.py")
kv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kv)

H = None
for attr in ("num_attention_heads", "num_heads", "n_heads", "num_heads_kv"):
    H = getattr(getattr(kv.tf, "config", object()), attr, None)
    if H:
        break
H = int(H or 30)

HEAD_CONFIGS = [
    ("full", dict(head_gate="off")),
    ("top7", dict(head_gate="top", head_keep=max(1, H // 4))),
    ("random7", dict(head_gate="random", head_keep=max(1, H // 4), head_seed=0)),
    ("bottom7", dict(head_gate="bottom", head_keep=max(1, H // 4))),
    ("clip_norm2", dict(head_gate="off", clip_ref=2.0, clip_mode="norm")),
    ("clip_random2", dict(head_gate="off", clip_ref=2.0, clip_mode="random", head_seed=0)),
]

TIME_CONFIGS = [
    ("taper_w4", 4.0, dict(sched_mode="taper")),
    ("late50_w4", 4.0, dict(sched_mode="late", sched_frac=0.5)),
    ("late50_w8", 8.0, dict(sched_mode="late", sched_frac=0.5)),
    ("late30_w8", 8.0, dict(sched_mode="late", sched_frac=0.3)),
]


def load_tags() -> dict:
    return {k: v for k, v in json.loads((DATA / "holdout30_tags.json").read_text()).items() if not k.startswith("_")}


def run_headtoken(tags: dict, subjects: list[str], seeds: list[int]) -> None:
    HEAD_OUT.mkdir(parents=True, exist_ok=True)
    kv.INJ["band"] = kv.parse_band("10-25")
    total = len(subjects) * len(seeds) * len(HEAD_CONFIGS)
    done = 0
    for uid in subjects:
        prompt = tags[uid]["prompt_B"]
        for seed in seeds:
            for name, config in HEAD_CONFIGS:
                out = HEAD_OUT / f"u{uid}_{name}_s{seed}.png"
                if out.exists():
                    done += 1
                    print(f"[headtoken {done}/{total}] skip {out.name}", flush=True)
                    continue
                image = kv.generate(
                    uid,
                    4.0,
                    refpos="lowfreq",
                    lowfreq_k=8,
                    sched_mode="taper",
                    seed=seed,
                    prompt=prompt,
                    size=512,
                    refsize=512,
                    **config,
                )
                image.save(out)
                done += 1
                print(f"[headtoken {done}/{total}] wrote {out.name}", flush=True)


def run_time(tags: dict, subjects: list[str], seeds: list[int]) -> None:
    TIME_OUT.mkdir(parents=True, exist_ok=True)
    kv.INJ["band"] = kv.parse_band("21-25")
    total = len(subjects) * len(seeds) * len(TIME_CONFIGS)
    done = 0
    for uid in subjects:
        prompt = tags[uid]["prompt_B"]
        for seed in seeds:
            for name, weight, config in TIME_CONFIGS:
                out = TIME_OUT / f"u{uid}_{name}_s{seed}.png"
                if out.exists():
                    done += 1
                    print(f"[time {done}/{total}] skip {out.name}", flush=True)
                    continue
                image = kv.generate(
                    uid,
                    weight,
                    refpos="strip",
                    seed=seed,
                    prompt=prompt,
                    size=512,
                    refsize=512,
                    **config,
                )
                image.save(out)
                done += 1
                print(f"[time {done}/{total}] wrote {out.name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=["headtoken", "time", "all"], default="all")
    parser.add_argument("--subjects", default=None, help="optional comma-separated uid subset")
    parser.add_argument("--seeds", default="0,1")
    args = parser.parse_args()

    tags = load_tags()
    subjects = sorted(tags, key=int) if not args.subjects else [x.strip().lstrip("u") for x in args.subjects.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    missing = [uid for uid in subjects if uid not in tags]
    if missing:
        raise ValueError(f"unknown subjects: {missing}")

    protocol = {
        "date": "2026-07-22",
        "subjects": subjects,
        "seeds": seeds,
        "headtoken": {
            "refpos": "lowfreq",
            "band": "10-25",
            "lowfreq_k": 8,
            "weight": 4.0,
            "schedule": "taper",
            "size": 512,
            "configs": [name for name, _ in HEAD_CONFIGS],
            "matched_controls": {"top7": ["random7", "bottom7"], "clip_norm2": ["clip_random2"]},
        },
        "time": {
            "refpos": "strip",
            "band": "21-25",
            "size": 512,
            "configs": [name for name, _, _ in TIME_CONFIGS],
            "baseline": "taper_w4",
        },
    }
    PROTOCOL_OUT.write_text(json.dumps(protocol, indent=2))
    print(f"wrote protocol {PROTOCOL_OUT}", flush=True)

    if args.experiment in {"headtoken", "all"}:
        run_headtoken(tags, subjects, seeds)
    if args.experiment in {"time", "all"}:
        run_time(tags, subjects, seeds)


if __name__ == "__main__":
    main()

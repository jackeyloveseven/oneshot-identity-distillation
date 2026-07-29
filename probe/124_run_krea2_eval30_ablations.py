#!/usr/bin/env python3
"""Run a subject-stratified mechanistic ablation of locked Krea2 TSD v3."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import sys
import threading
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path(sys.executable)
PROBE = ROOT / "probe" / "115_krea2_identity_tsd.py"
PROTOCOL = ROOT / "data" / "krea2_eval30_protocol.json"
OUT = ROOT / "out" / "krea2_tsd_eval30_ablations_v3"
LORA = ROOT / (
    "models/lora/"
    "krea2_identity_edit_v1_2.safetensors"
)
SEED = 2027

# First five identities of each automatic class in the frozen protocol order.
# This rule is fixed without inspecting any v3 output.
ABLATION_UIDS = (
    "5319",
    "2364",
    "1391",
    "1070",
    "9993",
    "4977",
    "5064",
    "6456",
    "2407",
    "4017",
)

VARIANTS = {
    "fixed_trust": {
        "adaptive_trust_ratio": 0.0,
    },
    "global_only": {
        "local_trust_region_rms": 10.0,
    },
    "loose_trust": {
        "adaptive_trust_ratio": 0.0,
        "trust_region_rms": 10.0,
        "local_trust_region_rms": 10.0,
    },
    "one_inner": {
        "inner_iters": 1,
    },
    "global_rms": {
        "normalize": "rms",
    },
    "raw_score": {
        "normalize": "none",
    },
    "no_ref_boost": {
        "ref_boost": 1.0,
    },
    "early_window": {
        "active_end": 0.55,
    },
    "late_window": {
        "active_start": 0.45,
    },
    "matched_lora": {
        "residual_mode": "matched_lora",
    },
    "identity_anchor_const": {
        "residual_mode": "identity_anchor",
        "identity_anchor_weight": 0.25,
        "identity_anchor_power": 0.0,
    },
    "identity_anchor_late": {
        "residual_mode": "identity_anchor",
        "identity_anchor_weight": 0.40,
        "identity_anchor_power": 2.0,
    },
    "identity_stat_const": {
        "residual_mode": "identity_stat",
        "identity_anchor_weight": 0.50,
        "identity_anchor_power": 0.0,
        "identity_stat_gain_min": 0.5,
        "identity_stat_gain_max": 2.0,
    },
    "identity_stat_late": {
        "residual_mode": "identity_stat",
        "identity_anchor_weight": 0.75,
        "identity_anchor_power": 2.0,
        "identity_stat_gain_min": 0.5,
        "identity_stat_gain_max": 2.0,
    },
    "wrong_ref_same_text": {},
}


def settings(protocol: dict, variant: str) -> dict:
    result = dict(protocol["method"])
    result.update(VARIANTS[variant])
    return result


def expected_metadata(job: dict, variant: str, protocol: dict) -> Path:
    method = settings(protocol, variant)
    residual_tag = method["residual_mode"]
    if residual_tag in ("identity_anchor", "identity_stat"):
        residual_tag += (
            f"{method['identity_anchor_weight']:g}"
            f"p{method['identity_anchor_power']:g}"
        )
    if method["residual_mode"] == "identity_stat":
        residual_tag += (
            f"c{method['identity_stat_gain_min']:g}-"
            f"{method['identity_stat_gain_max']:g}"
        )
    suffix = (
        f"s{SEED}_{job['width']}x{job['height']}_"
        f"i{method['inner_iters']}_p{method['pseudo_step_size']:g}_"
        f"g{method['trust_region_rms']:g}_"
        f"l{method['local_trust_region_rms']:g}_"
        f"a{method['adaptive_trust_ratio']:g}_"
        f"rb{method['ref_boost']:g}_"
        f"{method['normalize']}_{residual_tag}_dcap_"
        "roifull_o1_cnone.json"
    )
    return OUT / variant / f"{job['job_id']}_{suffix}"


def command(job: dict, variant: str, protocol: dict) -> list[str]:
    method = settings(protocol, variant)
    reference = (
        job["negative_reference"]
        if variant == "wrong_ref_same_text"
        else job["reference"]
    )
    negative_reference = (
        job["reference"]
        if variant == "wrong_ref_same_text"
        else job["negative_reference"]
    )
    result = [
        str(PYTHON),
        str(PROBE),
        "--uid",
        job["job_id"],
        "--ref",
        reference,
        "--negative-ref",
        negative_reference,
        "--lora-path",
        str(LORA),
        "--out-dir",
        str(OUT / variant),
        "--seed",
        str(SEED),
        "--width",
        str(job["width"]),
        "--height",
        str(job["height"]),
        "--steps",
        str(method["steps"]),
        "--inner-iters",
        str(method["inner_iters"]),
        "--lr",
        str(method["lr"]),
        "--pseudo-step-size",
        str(method["pseudo_step_size"]),
        "--teacher-weight",
        str(method["teacher_weight"]),
        "--active-start",
        str(method["active_start"]),
        "--active-end",
        str(method["active_end"]),
        "--trust-region-rms",
        str(method["trust_region_rms"]),
        "--local-trust-region-rms",
        str(method["local_trust_region_rms"]),
        "--adaptive-trust-ratio",
        str(method["adaptive_trust_ratio"]),
        "--normalize",
        method["normalize"],
        "--residual-mode",
        method["residual_mode"],
        "--ref-boost",
        str(method["ref_boost"]),
        "--outside-weight",
        str(method["outside_weight"]),
        "--controls",
        "none",
        "--scene-prompt",
        job["scene_prompt"],
        "--caption-prompt",
        job["caption_prompt"],
        "--teacher-prompt",
        job["teacher_prompt"],
        "--negative-teacher-prompt",
        job["negative_teacher_prompt"],
    ]
    if method["residual_mode"] in ("identity_anchor", "identity_stat"):
        result.extend(
            [
                "--identity-anchor-weight",
                str(method["identity_anchor_weight"]),
                "--identity-anchor-power",
                str(method["identity_anchor_power"]),
            ]
        )
    if method["residual_mode"] == "identity_stat":
        result.extend(
            [
                "--identity-stat-gain-min",
                str(method["identity_stat_gain_min"]),
                "--identity-stat-gain-max",
                str(method["identity_stat_gain_max"]),
            ]
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--variants", nargs="*", choices=tuple(VARIANTS))
    parser.add_argument("--uids", nargs="*")
    parser.add_argument("--scenes", nargs="*")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="print one line for every already-complete task",
    )
    parser.add_argument(
        "--verbose-manifest",
        action="store_true",
        help="print the full manifest instead of a one-line completion summary",
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--pending-limit",
        type=int,
        help="run at most this many not-yet-complete tasks in this invocation",
    )
    args = parser.parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.pending_limit is not None and args.pending_limit < 1:
        raise ValueError("--pending-limit must be positive")

    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    chosen_variants = args.variants or list(VARIANTS)
    uid_filter = set(args.uids) if args.uids else None
    scene_filter = set(args.scenes) if args.scenes else None
    jobs = [
        job
        for job in protocol["jobs"]
        if job["uid"] in ABLATION_UIDS
        and (uid_filter is None or job["uid"] in uid_filter)
        and (scene_filter is None or job["scene"] in scene_filter)
    ]
    expected_jobs = len(ABLATION_UIDS) * protocol["scene_count"]
    if uid_filter is None and scene_filter is None and len(jobs) != expected_jobs:
        raise AssertionError(f"expected {expected_jobs} jobs, found {len(jobs)}")
    tasks = [(job, variant) for variant in chosen_variants for job in jobs]
    if args.limit is not None:
        tasks = tasks[: args.limit]

    pending = []
    skipped = 0
    for job, variant in tasks:
        path = expected_metadata(job, variant, protocol)
        if path.is_file():
            skipped += 1
            if args.verbose_skips:
                print(f"[skip] {variant}/{path.name}", flush=True)
        else:
            pending.append((job, variant))
    if args.pending_limit is not None:
        pending = pending[: args.pending_limit]

    lock = threading.Lock()

    def execute(task: tuple[dict, str]) -> tuple[str, str]:
        job, variant = task
        output_dir = OUT / variant
        output_dir.mkdir(parents=True, exist_ok=True)
        path = expected_metadata(job, variant, protocol)
        cmd = command(job, variant, protocol)
        with lock:
            print(
                f"[run] {variant} uid={job['uid']} scene={job['scene']}",
                flush=True,
            )
        if args.dry_run:
            with lock:
                print(" ".join(cmd), flush=True)
            return variant, path.name
        log_path = path.with_suffix(".run.log")
        try:
            with log_path.open("w", encoding="utf-8") as log_handle:
                subprocess.run(
                    cmd,
                    check=True,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
        except subprocess.CalledProcessError:
            tail = log_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()[-30:]
            with lock:
                print("\n".join(tail), flush=True)
            raise
        if not path.is_file():
            raise FileNotFoundError(path)
        return variant, path.name

    completed = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.workers
    ) as executor:
        futures = [executor.submit(execute, task) for task in pending]
        for future in concurrent.futures.as_completed(futures):
            variant, name = future.result()
            completed += 1
            print(
                f"[done {completed}/{len(pending)}] {variant}/{name}",
                flush=True,
            )

    manifest = {
        "protocol": protocol["name"],
        "selection_rule": (
            "first five identities per automatic class in frozen protocol order"
        ),
        "uids": list(ABLATION_UIDS),
        "scenes": sorted({job["scene"] for job in jobs}),
        "seed": SEED,
        "variants": {
            variant: {
                "settings": settings(protocol, variant),
                "reference_override": (
                    "same-class wrong reference with target text held fixed"
                    if variant == "wrong_ref_same_text"
                    else "none"
                ),
            }
            for variant in chosen_variants
        },
        "selected": len(tasks),
        "completed": 0 if args.dry_run else completed,
        "simulated": completed if args.dry_run else 0,
        "skipped": skipped,
        "dry_run": args.dry_run,
        "elapsed_seconds": time.time() - started,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    if args.verbose_manifest:
        print(json.dumps(manifest, indent=2), flush=True)
    else:
        print(
            "[summary] "
            f"completed={manifest['completed']} "
            f"skipped={manifest['skipped']} "
            f"selected={manifest['selected']} "
            f"elapsed={manifest['elapsed_seconds']:.1f}s",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

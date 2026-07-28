#!/usr/bin/env python3
"""Execute the locked Krea2 eval30 protocol with resumable per-sample jobs."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import subprocess
import threading
import time
from pathlib import Path


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
PYTHON = Path("/mnt/local/krea2_edit_serve/venv/bin/python")
PROBE = ROOT / "probe" / "115_krea2_identity_tsd.py"
DEFAULT_PROTOCOL = ROOT / "data" / "krea2_eval30_protocol.json"
DEFAULT_OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
LORA = Path(
    "/mnt/local/krea2_edit_serve/models/lora/"
    "krea2_identity_edit_v1_2.safetensors"
)


def expected_metadata(job: dict, seed: int, output_dir: Path) -> Path:
    method = {
        "inner_iters": 3,
        "pseudo_step_size": 1.0,
        "trust_region_rms": 0.10,
        "local_trust_region_rms": 0.085,
        "adaptive_trust_ratio": 3.0,
        "ref_boost": 4.0,
    }
    suffix = (
        f"s{seed}_{job['width']}x{job['height']}_"
        f"i{method['inner_iters']}_p{method['pseudo_step_size']:g}_"
        f"g{method['trust_region_rms']:g}_"
        f"l{method['local_trust_region_rms']:g}_"
        f"a{method['adaptive_trust_ratio']:g}_"
        f"rb{method['ref_boost']:g}_rms_token_raw_dcap_roifull_o1.json"
    )
    return output_dir / f"{job['job_id']}_{suffix}"


def command_for(job: dict, seed: int, protocol: dict, output_dir: Path) -> list[str]:
    method = protocol["method"]
    return [
        str(PYTHON),
        str(PROBE),
        "--uid",
        job["job_id"],
        "--ref",
        job["reference"],
        "--negative-ref",
        job["negative_reference"],
        "--lora-path",
        str(LORA),
        "--out-dir",
        str(output_dir),
        "--seed",
        str(seed),
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
        "--scene-prompt",
        job["scene_prompt"],
        "--caption-prompt",
        job["caption_prompt"],
        "--teacher-prompt",
        job["teacher_prompt"],
        "--negative-teacher-prompt",
        job["negative_teacher_prompt"],
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--uids", nargs="*", default=None)
    parser.add_argument("--scenes", nargs="*", default=None)
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--pending-limit",
        type=int,
        default=None,
        help="run at most this many not-yet-complete tasks in this invocation",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--verbose-skips",
        action="store_true",
        help="print every completed metadata path instead of only the total",
    )
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    uid_filter = set(args.uids) if args.uids else None
    scene_filter = set(args.scenes) if args.scenes else None
    seed_filter = set(args.seeds) if args.seeds else None

    tasks: list[tuple[dict, int]] = []
    for job in protocol["jobs"]:
        if uid_filter is not None and job["uid"] not in uid_filter:
            continue
        if scene_filter is not None and job["scene"] not in scene_filter:
            continue
        for seed in job["seeds"]:
            if seed_filter is not None and seed not in seed_filter:
                continue
            tasks.append((job, seed))
    if args.limit is not None:
        tasks = tasks[: args.limit]

    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.pending_limit is not None and args.pending_limit < 1:
        raise ValueError("--pending-limit must be positive")

    pending = []
    skipped = 0
    for job, seed in tasks:
        metadata = expected_metadata(job, seed, args.out_dir)
        if metadata.is_file():
            skipped += 1
            if args.verbose_skips:
                print(f"[skip] {metadata.name}", flush=True)
        else:
            pending.append((job, seed))
    if args.pending_limit is not None:
        pending = pending[: args.pending_limit]

    print_lock = threading.Lock()

    def execute(task: tuple[dict, int]) -> str:
        job, seed = task
        metadata = expected_metadata(job, seed, args.out_dir)
        command = command_for(job, seed, protocol, args.out_dir)
        with print_lock:
            print(
                f"[run] uid={job['uid']} scene={job['scene']} seed={seed}",
                flush=True,
            )
        if args.dry_run:
            with print_lock:
                print(" ".join(command), flush=True)
            return metadata.name
        subprocess.run(command, check=True)
        if not metadata.is_file():
            raise FileNotFoundError(f"probe finished without metadata: {metadata}")
        return metadata.name

    completed = 0
    started = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, task) for task in pending]
        for future in concurrent.futures.as_completed(futures):
            name = future.result()
            completed += 1
            with print_lock:
                print(
                    f"[done {completed}/{len(pending)}] {name}",
                    flush=True,
                )

    elapsed = time.time() - started
    print(
        json.dumps(
            {
                "selected": len(tasks),
                "completed": completed,
                "skipped": skipped,
                "elapsed_seconds": elapsed,
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

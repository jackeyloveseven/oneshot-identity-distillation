#!/usr/bin/env python3
"""Protocol-matched equal-width band analysis for the band-LoRA study.

Primary comparison: early 0-15 r8, clean mid 10-25 r8, late 14-29 r8.
All three used train_manifest_mix3.jsonl, rank/alpha 8, seed 0, uniform-t,
4000 steps, raw prompt_B generation, and the same Gemini 3.1 Pro judge.
The old full-depth 0-29 r4 arm is intentionally excluded from the primary
location comparison because rank and trainable parameter count differ.
"""

from __future__ import annotations

import json
import math
import random
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "uniform_band_report.json"
BOOT_N = 50_000
BOOT_SEED = 20260722

SETS = {
    "repprobe30": {
        "early_0-15_r8": DATA / "gemini_judge_band_0-15_r8_step4000.json",
        "mid_10-25_r8_clean": DATA / "gemini_judge_cleanmid_rep30_band_10-25_r8_step4000.json",
        "late_14-29_r8": DATA / "gemini_judge_band_14-29_r8_step4000.json",
    },
    "holdout30": {
        "early_0-15_r8": DATA / "gemini_judge_ho30_band_0-15_r8_step4000.json",
        "mid_10-25_r8_clean": DATA / "gemini_judge_cleanmid_ho30_band_10-25_r8_step4000.json",
        "late_14-29_r8": DATA / "gemini_judge_ho30_band_14-29_r8_step4000.json",
    },
}

METRICS = {
    "clean_yes": lambda v: v.get("same_character") == "yes" and v.get("visible_defect") == "no",
    "defect": lambda v: v.get("visible_defect") == "yes",
    "clean_usable": lambda v: v.get("same_character") in {"yes", "partial"} and v.get("visible_defect") == "no",
}


def parse_uid_seed(pid: str) -> tuple[str, int]:
    m = re.search(r"(?:^|_)(\d+).*_s([01])$", pid)
    if not m:
        raise ValueError(f"cannot parse uid/seed from {pid!r}")
    return m.group(1), int(m.group(2))


def load_subject_values(path: Path) -> dict[str, dict[str, float]]:
    raw = json.loads(path.read_text())
    by_subject: dict[str, dict[int, dict]] = {}
    for pid, verdict in raw.items():
        uid, seed = parse_uid_seed(pid)
        if seed in by_subject.setdefault(uid, {}):
            raise ValueError(f"duplicate {uid=} {seed=} in {path}")
        by_subject[uid][seed] = verdict
    if len(by_subject) != 30:
        raise ValueError(f"expected 30 subjects in {path}, found {len(by_subject)}")
    out = {}
    for uid, seeds in by_subject.items():
        if set(seeds) != {0, 1}:
            raise ValueError(f"expected seeds 0/1 for {uid} in {path}, found {sorted(seeds)}")
        out[uid] = {
            metric: sum(float(fn(seeds[s])) for s in (0, 1)) / 2.0
            for metric, fn in METRICS.items()
        }
    return out


def percentile(xs: list[float], q: float) -> float:
    ys = sorted(xs)
    pos = (len(ys) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return ys[lo]
    return ys[lo] * (hi - pos) + ys[hi] * (pos - lo)


def bootstrap_rate(values: dict[str, float], rng: random.Random) -> tuple[float, list[float]]:
    uids = sorted(values)
    point = sum(values.values()) / len(uids)
    draws = []
    for _ in range(BOOT_N):
        draws.append(sum(values[rng.choice(uids)] for _ in uids) / len(uids))
    return point, [percentile(draws, 0.025), percentile(draws, 0.975)]


def paired_bootstrap_diff(a: dict[str, float], b: dict[str, float], rng: random.Random) -> dict:
    if set(a) != set(b):
        raise ValueError("paired arms have different subject sets")
    uids = sorted(a)
    diffs = {u: a[u] - b[u] for u in uids}
    point = sum(diffs.values()) / len(uids)
    draws = []
    for _ in range(BOOT_N):
        draws.append(sum(diffs[rng.choice(uids)] for _ in uids) / len(uids))
    lo, hi = percentile(draws, 0.025), percentile(draws, 0.975)
    p_two = 2.0 * min(sum(x <= 0 for x in draws), sum(x >= 0 for x in draws)) / len(draws)
    return {"difference": point, "ci95": [lo, hi], "bootstrap_p_two_sided": min(1.0, p_two)}


def main() -> None:
    rng = random.Random(BOOT_SEED)
    report = {
        "protocol": {
            "primary_arms": list(next(iter(SETS.values())).keys()),
            "common_training": "train_manifest_mix3.jsonl; rank=alpha=8; seed=0; uniform-t; 4000 steps",
            "common_generation": "raw prompt_B; 30 subjects x seeds {0,1}; no style prefix",
            "judge": "Gemini 3.1 Pro defect-aware prompt",
            "full_depth_note": "0-29 r4 excluded from primary location comparison because rank/parameter count differ",
            "bootstrap": {"unit": "subject", "n": BOOT_N, "seed": BOOT_SEED},
        },
        "sets": {},
    }
    for set_name, paths in SETS.items():
        arms = {name: load_subject_values(path) for name, path in paths.items()}
        set_out = {"arms": {}, "paired_differences": {}}
        for arm, values in arms.items():
            set_out["arms"][arm] = {}
            for metric in METRICS:
                point, ci = bootstrap_rate({u: v[metric] for u, v in values.items()}, rng)
                set_out["arms"][arm][metric] = {"rate": point, "ci95": ci}
        names = list(arms)
        for i, left in enumerate(names):
            for right in names[i + 1:]:
                key = f"{left}_minus_{right}"
                set_out["paired_differences"][key] = {}
                for metric in METRICS:
                    set_out["paired_differences"][key][metric] = paired_bootstrap_diff(
                        {u: v[metric] for u, v in arms[left].items()},
                        {u: v[metric] for u, v in arms[right].items()},
                        rng,
                    )
        report["sets"][set_name] = set_out

    clean_yes_cis = [
        comparison["clean_yes"]["ci95"]
        for set_out in report["sets"].values()
        for comparison in set_out["paired_differences"].values()
    ]
    report["interpretation"] = {
        "any_pairwise_clean_yes_ci_excludes_zero": any(lo > 0 or hi < 0 for lo, hi in clean_yes_cis),
        "claim": "No tested equal-width band shows a detected clean-yes advantage on either subject set; this is failure to detect localization, not proof of equivalence.",
    }
    OUT.write_text(json.dumps(report, indent=2))
    print(f"wrote {OUT}")
    for set_name, set_out in report["sets"].items():
        print(f"\n[{set_name}]")
        for arm, metrics in set_out["arms"].items():
            vals = " ".join(f"{m}={100*x['rate']:.1f}% [{100*x['ci95'][0]:.1f},{100*x['ci95'][1]:.1f}]" for m, x in metrics.items())
            print(f"  {arm}: {vals}")
        for pair, metrics in set_out["paired_differences"].items():
            x = metrics["clean_yes"]
            print(f"  {pair} clean_yes diff={100*x['difference']:+.1f}pp CI=[{100*x['ci95'][0]:+.1f},{100*x['ci95'][1]:+.1f}] p={x['bootstrap_p_two_sided']:.3f}")


if __name__ == "__main__":
    main()

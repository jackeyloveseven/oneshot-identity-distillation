#!/usr/bin/env python3
"""Subject-clustered inference for anonymous multi-pass VLM evaluations.

Each summary contains majority labels for two fixed seeds per subject.  The
subject, not the image, is the resampling unit.  The report includes clustered
rate intervals and paired arm differences for every metric and arm pair.
"""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
DEFAULT_INPUTS = [
    DATA / "gemini3flash_blind3arm_summary.json",
    DATA / "gemini3flash_headtoken_blind_summary.json",
    DATA / "gemini3flash_time_blind_summary.json",
]
BOOT_N = 50_000
BOOT_SEED = 20260722


def labels_to_metrics(item: dict) -> dict[str, float]:
    identity = item["identity"]
    defect = item["visible_defect"] == "yes"
    return {
        "clean_yes": float(identity == "yes" and not defect),
        "defect": float(defect),
        "clean_usable": float(identity in {"yes", "partial"} and not defect),
    }


def subject_matrix(items: list[dict]) -> tuple[list[str], dict[str, np.ndarray]]:
    by_subject: dict[str, list[dict[str, float]]] = {}
    seen = set()
    for item in items:
        key = (str(item["uid"]), int(item["seed"]))
        if key in seen:
            raise ValueError(f"duplicate subject/seed item: {key}")
        seen.add(key)
        by_subject.setdefault(key[0], []).append(labels_to_metrics(item))
    if len(by_subject) != 30:
        raise ValueError(f"expected 30 subjects, found {len(by_subject)}")
    if any(len(rows) != 2 for rows in by_subject.values()):
        bad = {uid: len(rows) for uid, rows in by_subject.items() if len(rows) != 2}
        raise ValueError(f"expected two seeds per subject: {bad}")
    uids = sorted(by_subject, key=int)
    metrics = {
        metric: np.array([np.mean([row[metric] for row in by_subject[uid]]) for uid in uids], dtype=float)
        for metric in ("clean_yes", "defect", "clean_usable")
    }
    return uids, metrics


def percentile_ci(draws: np.ndarray) -> list[float]:
    return [float(x) for x in np.quantile(draws, [0.025, 0.975])]


def bootstrap_rate(values: np.ndarray, indices: np.ndarray) -> dict:
    draws = values[indices].mean(axis=1)
    return {"rate": float(values.mean()), "ci95": percentile_ci(draws)}


def paired_difference(left: np.ndarray, right: np.ndarray, indices: np.ndarray) -> dict:
    values = left - right
    draws = values[indices].mean(axis=1)
    p_two = 2.0 * min(float(np.mean(draws <= 0)), float(np.mean(draws >= 0)))
    lo, hi = percentile_ci(draws)
    return {
        "difference": float(values.mean()),
        "ci95": [lo, hi],
        "bootstrap_p_two_sided": min(1.0, p_two),
        "detected_direction": "positive" if lo > 0 else "negative" if hi < 0 else "none",
    }


def analyze(path: Path, n_boot: int, seed: int) -> tuple[dict, Path]:
    summary = json.loads(path.read_text())
    majority_items = summary.get("majority_items", {})
    if not majority_items:
        raise ValueError(f"missing majority_items in {path}")

    matrices = {}
    canonical_uids = None
    for arm, items in majority_items.items():
        uids, metrics = subject_matrix(items)
        if canonical_uids is None:
            canonical_uids = uids
        elif uids != canonical_uids:
            raise ValueError(f"subject mismatch for arm {arm} in {path}")
        matrices[arm] = metrics

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(canonical_uids), size=(n_boot, len(canonical_uids)))
    report = {
        "source": path.name,
        "definition": summary.get("definition"),
        "model": summary.get("model"),
        "passes": summary.get("passes"),
        "agreement": summary.get("agreement"),
        "bootstrap": {"unit": "subject", "subjects": len(canonical_uids), "seeds_per_subject": 2, "n": n_boot, "seed": seed},
        "arms": {},
        "paired_differences": {},
    }
    for arm, metrics in matrices.items():
        report["arms"][arm] = {metric: bootstrap_rate(values, indices) for metric, values in metrics.items()}

    for left, right in combinations(matrices, 2):
        key = f"{left}_minus_{right}"
        report["paired_differences"][key] = {
            metric: paired_difference(matrices[left][metric], matrices[right][metric], indices)
            for metric in ("clean_yes", "defect", "clean_usable")
        }

    report["interpretation"] = {
        "inference_scope": "Failure to detect a paired difference is not evidence of equivalence.",
        "pareto_rule": "A strict detected improvement requires higher clean-yes or clean-usable and lower defect, with paired 95% intervals excluding zero in the favorable direction.",
    }
    output_stem = path.stem.replace("_summary", "_clustered_stats")
    if path.stem == "gemini3flash_blind3arm_summary":
        output_stem = "gemini3flash_threearm_clustered_stats"
    output = path.with_name(output_stem + path.suffix)
    output.write_text(json.dumps(report, indent=2))
    return report, output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", type=Path)
    parser.add_argument("--boot", type=int, default=BOOT_N)
    parser.add_argument("--seed", type=int, default=BOOT_SEED)
    args = parser.parse_args()
    inputs = args.inputs or [path for path in DEFAULT_INPUTS if path.exists()]
    if not inputs:
        raise FileNotFoundError("no blind-evaluation summaries found")
    for path in inputs:
        report, output = analyze(path, args.boot, args.seed)
        print(f"wrote {output}")
        for arm, metrics in report["arms"].items():
            text = " ".join(
                f"{name}={100 * value['rate']:.1f}% [{100 * value['ci95'][0]:.1f},{100 * value['ci95'][1]:.1f}]"
                for name, value in metrics.items()
            )
            print(f"  {arm}: {text}")


if __name__ == "__main__":
    main()

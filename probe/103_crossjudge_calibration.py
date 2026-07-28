#!/usr/bin/env python3
"""Compare Gemini 3 Flash majority labels with Gemini 3.1 Pro labels.

Both judges see the same 30 subjects x two seeds for three retained methods. The output
is a calibration diagnostic, not a claim that either VLM is human ground truth.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FLASH = DATA / "gemini3flash_blind3arm_summary.json"
OUT = DATA / "gemini3flash_vs_gemini31pro_threearm_calibration.json"
PRO_SUMMARY_OUT = DATA / "gemini31pro_threearm_summary.json"

PRO_SOURCES = {
    "strip": (DATA / "gemini_judge_holdout.json", re.compile(r"^(\d+)_old_s([01])$")),
    "krea2": (DATA / "gemini_judge_krea2.json", re.compile(r"^(\d+)_krea2_s([01])$")),
    "band_lora_step6000": (
        DATA / "gemini_judge_m1s6k_ho30_band_10-25_r8_step6000.json",
        re.compile(r"^(\d+)_lora_ckptlink_m1s6k_ho30_band10-25_r8_s([01])$"),
    ),
}
IDENTITY_ORDER = {"no": 0, "partial": 1, "yes": 2}


def cohen_kappa(pairs: list[tuple[str, str]], categories: list[str]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    observed = sum(a == b for a, b in pairs) / n
    left = Counter(a for a, _ in pairs)
    right = Counter(b for _, b in pairs)
    expected = sum((left[c] / n) * (right[c] / n) for c in categories)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def load_pro(path: Path, pattern: re.Pattern[str]) -> dict[tuple[str, int], dict]:
    raw = json.loads(path.read_text())
    out = {}
    for pid, verdict in raw.items():
        match = pattern.match(pid)
        if not match:
            continue
        key = (match.group(1), int(match.group(2)))
        if key in out:
            raise ValueError(f"duplicate {key} in {path}")
        out[key] = {
            "identity": verdict["same_character"],
            "visible_defect": verdict.get("visible_defect", "no"),
        }
    if len(out) != 60:
        raise ValueError(f"expected 60 matched records in {path}, found {len(out)}")
    return out


def summarize_pairs(pairs: list[tuple[dict, dict]]) -> dict:
    identity_pairs = [(a["identity"], b["identity"]) for a, b in pairs]
    defect_pairs = [(a["visible_defect"], b["visible_defect"]) for a, b in pairs]
    identity_delta = [IDENTITY_ORDER[a] - IDENTITY_ORDER[b] for a, b in identity_pairs]
    return {
        "n": len(pairs),
        "identity": {
            "exact_agreement": sum(a == b for a, b in identity_pairs) / len(identity_pairs),
            "cohen_kappa": cohen_kappa(identity_pairs, ["yes", "partial", "no"]),
            "flash_stricter_rate": sum(delta < 0 for delta in identity_delta) / len(identity_delta),
            "flash_more_lenient_rate": sum(delta > 0 for delta in identity_delta) / len(identity_delta),
            "confusion_flash_rows_pro_columns": {
                flash: {pro: sum(a == flash and b == pro for a, b in identity_pairs) for pro in ("yes", "partial", "no")}
                for flash in ("yes", "partial", "no")
            },
        },
        "visible_defect": {
            "exact_agreement": sum(a == b for a, b in defect_pairs) / len(defect_pairs),
            "cohen_kappa": cohen_kappa(defect_pairs, ["yes", "no"]),
            "flash_yes_pro_no_rate": sum(a == "yes" and b == "no" for a, b in defect_pairs) / len(defect_pairs),
            "flash_no_pro_yes_rate": sum(a == "no" and b == "yes" for a, b in defect_pairs) / len(defect_pairs),
            "confusion_flash_rows_pro_columns": {
                flash: {pro: sum(a == flash and b == pro for a, b in defect_pairs) for pro in ("yes", "no")}
                for flash in ("yes", "no")
            },
        },
    }


def main() -> None:
    flash_summary = json.loads(FLASH.read_text())
    report = {
        "definition": "Same-image VLM calibration comparison; neither judge is treated as human ground truth.",
        "flash_model": flash_summary["model"],
        "pro_model": "Gemini 3.1 Pro (single-pass project judge)",
        "methods": {},
    }
    pro_summary = {
        "model": "Gemini 3.1 Pro (single-pass project judge)",
        "definition": "single-pass blinded VLM evaluation; not human evaluation",
        "passes": 1,
        "agreement": None,
        "majority_items": {},
    }
    all_pairs = []
    for method, items in flash_summary["majority_items"].items():
        flash = {(str(item["uid"]), int(item["seed"])): item for item in items}
        pro = load_pro(*PRO_SOURCES[method])
        if set(flash) != set(pro):
            raise ValueError(f"subject/seed mismatch for {method}")
        pairs = [(flash[key], pro[key]) for key in sorted(flash, key=lambda x: (int(x[0]), x[1]))]
        report["methods"][method] = summarize_pairs(pairs)
        pro_summary["majority_items"][method] = [
            {"uid": uid, "seed": seed, **pro[(uid, seed)]}
            for uid, seed in sorted(pro, key=lambda x: (int(x[0]), x[1]))
        ]
        all_pairs.extend(pairs)
    report["overall"] = summarize_pairs(all_pairs)
    OUT.write_text(json.dumps(report, indent=2))
    PRO_SUMMARY_OUT.write_text(json.dumps(pro_summary, indent=2))
    print(f"wrote {OUT}")
    print(f"wrote {PRO_SUMMARY_OUT}")
    for method, result in report["methods"].items():
        ident = result["identity"]
        defect = result["visible_defect"]
        print(
            f"{method}: identity agreement={100 * ident['exact_agreement']:.1f}% "
            f"kappa={ident['cohen_kappa']:.3f}; defect agreement={100 * defect['exact_agreement']:.1f}% "
            f"kappa={defect['cohen_kappa']:.3f}"
        )


if __name__ == "__main__":
    main()

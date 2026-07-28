#!/usr/bin/env python3
"""Blinded multi-candidate Gemini 3 Flash evaluation for localization sweeps."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
REF_ROOT = Path("/mnt/local/smoke_anime/anime_260623/img")
KEY_SRC = Path("/home/ubuntu/smoke_anime/flash_caption_smoke.py")
BASE_URL = "https://kaon-router.kaonai.com/v1"
MODEL = "kaon/gemini-3-flash-preview"

EXPERIMENTS = {
    "headtoken": {
        "root": Path("/mnt/local/cherry_out/localization_headtoken_q60"),
        "configs": ["full", "top7", "random7", "bottom7", "clip_norm2", "clip_random2"],
    },
    "time": {
        "root": Path("/mnt/local/cherry_out/localization_time_q60"),
        "configs": ["taper_w4", "late50_w4", "late50_w8", "late30_w8"],
    },
}

PROMPT = """You are performing a blinded evaluation of reference-conditioned anime generation.
Image 1 is the REFERENCE. The following lettered images are anonymous generated candidates intended to depict the same subject.

Score EACH candidate independently:
- identity: yes = same subject including distinctive face, hair shape/color, eyes, gender presentation, skin tone, masks, ears, scars, or accessories; partial = recognizable overlap but an important identity cue is missing/changed; no = different subject. Ignore pose, outfit, expression, background, and style.
- visible_defect: yes only for duplicated/tiled/collaged sub-images, malformed or extra anatomy, broken faces/limbs, or structurally incoherent composition. Identity mismatch alone is not a defect.

Do not infer methods. Return ONLY one JSON object with exactly the supplied letter keys. Each value must be:
{"identity":"yes|partial|no","visible_defect":"yes|no","confidence":0..1,"note":"brief visible evidence"}
"""


def load_key() -> str:
    match = re.search(r"KAON_KEY\s*=\s*[\"']([^\"']+)", KEY_SRC.read_text())
    if not match:
        raise RuntimeError("KAON_KEY not found")
    return match.group(1)


def data_uri(path: Path, px: int = 768) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((px, px))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def extract_json(text: str, labels: list[str]) -> dict:
    match = re.search(r"\{.*\}", text.strip(), re.S)
    if not match:
        raise ValueError(f"no JSON: {text[:160]!r}")
    obj = json.loads(match.group(0))
    if set(obj) != set(labels):
        raise ValueError(f"expected {labels}, got {sorted(obj)}")
    for label in labels:
        if obj[label].get("identity") not in {"yes", "partial", "no"}:
            raise ValueError(f"bad identity: {obj[label]}")
        if obj[label].get("visible_defect") not in {"yes", "no"}:
            raise ValueError(f"bad defect: {obj[label]}")
    return obj


def call(ref: Path, paths: dict[str, Path], key: str, retries: int = 5) -> dict:
    labels = list(paths)
    content = [
        {"type": "text", "text": PROMPT + "\nSupplied keys: " + ", ".join(labels)},
        {"type": "text", "text": "REFERENCE (Image 1)"},
        {"type": "image_url", "image_url": {"url": data_uri(ref)}},
    ]
    for label, path in paths.items():
        content.extend([
            {"type": "text", "text": f"Candidate {label}"},
            {"type": "image_url", "image_url": {"url": data_uri(path)}},
        ])
    body = {"model": MODEL, "messages": [{"role": "user", "content": content}], "max_tokens": 1500, "temperature": 0.2}
    last = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(BASE_URL + "/chat/completions", headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, json=body, timeout=300)
            if response.status_code in {408, 429} or response.status_code >= 500:
                last = f"HTTP {response.status_code}: {response.text[:160]}"
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return {"status": "ok", "verdicts_by_label": extract_json(text, labels), "raw_response": text, "request_id": response.headers.get("X-Request-ID")}
        except Exception as exc:
            last = f"{type(exc).__name__}: {exc}"[:300]
            time.sleep(2 * (attempt + 1))
    return {"status": "error", "error": last}


def fleiss(rows: list[list[str]], categories: list[str]) -> float | None:
    if not rows or len(rows[0]) < 2:
        return None
    raters = len(rows[0])
    agreements, totals = [], Counter()
    for row in rows:
        counts = Counter(row)
        agreements.append((sum(v * v for v in counts.values()) - raters) / (raters * (raters - 1)))
        totals.update(row)
    p_bar = sum(agreements) / len(agreements)
    n = len(rows) * raters
    p_e = sum((totals[c] / n) ** 2 for c in categories)
    return (p_bar - p_e) / (1 - p_e) if p_e < 1 else 1.0


def metric(items: list[dict]) -> dict:
    n = len(items)
    return {
        "n": n,
        "identity_counts": dict(Counter(x["identity"] for x in items)),
        "defect_rate": sum(x["visible_defect"] == "yes" for x in items) / n if n else None,
        "clean_yes_rate": sum(x["identity"] == "yes" and x["visible_defect"] == "no" for x in items) / n if n else None,
        "clean_usable_rate": sum(x["identity"] in {"yes", "partial"} and x["visible_defect"] == "no" for x in items) / n if n else None,
    }


def summarize(raw: dict, configs: list[str], passes: int) -> dict:
    grouped: dict[tuple[str, int, str], list[dict]] = {}
    errors = []
    for case_id, record in raw["cases"].items():
        if record.get("status") != "ok":
            errors.append(case_id)
            continue
        for label, config in record["mapping"].items():
            grouped.setdefault((record["uid"], record["seed"], config), []).append(record["verdicts_by_label"][label])
    identity_rows, defect_rows = [], []
    majority = {config: [] for config in configs}
    for (uid, seed, config), values in sorted(grouped.items()):
        if len(values) != passes:
            continue
        ids = [v["identity"] for v in values]
        defs = [v["visible_defect"] for v in values]
        identity_rows.append(ids)
        defect_rows.append(defs)
        majority[config].append({"uid": uid, "seed": seed, "identity": Counter(ids).most_common(1)[0][0], "visible_defect": Counter(defs).most_common(1)[0][0]})
    return {
        "model": MODEL,
        "definition": "anonymous multi-pass VLM evaluation; not human evaluation",
        "passes": passes,
        "errors": errors,
        "agreement": {
            "identity_fleiss_kappa": fleiss(identity_rows, ["yes", "partial", "no"]),
            "defect_fleiss_kappa": fleiss(defect_rows, ["yes", "no"]),
            "identity_unanimous_rate": sum(len(set(x)) == 1 for x in identity_rows) / len(identity_rows) if identity_rows else None,
            "defect_unanimous_rate": sum(len(set(x)) == 1 for x in defect_rows) / len(defect_rows) if defect_rows else None,
        },
        "majority": {config: metric(items) for config, items in majority.items()},
        "majority_items": majority,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=sorted(EXPERIMENTS), required=True)
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="evaluate only subject/seed cases whose complete candidate set already exists",
    )
    args = parser.parse_args()
    spec = EXPERIMENTS[args.experiment]
    labels = [chr(ord("A") + i) for i in range(len(spec["configs"]))]
    tags = {k: v for k, v in json.loads((DATA / "holdout30_tags.json").read_text()).items() if not k.startswith("_")}
    base_cases = [(u, s) for u in sorted(tags, key=int) for s in (0, 1)]
    if args.limit is not None:
        base_cases = base_cases[: args.limit]
    if args.available_only:
        base_cases = [
            (uid, seed)
            for uid, seed in base_cases
            if all((spec["root"] / f"u{uid}_{config}_s{seed}.png").exists() for config in spec["configs"])
        ]
        if not base_cases:
            raise FileNotFoundError(f"no complete {args.experiment} subject/seed cases are available yet")
    cases = []
    for pass_id in range(args.passes):
        for uid, seed in base_cases:
            configs = list(spec["configs"])
            random.Random(20260722 + pass_id * 10000 + int(uid) * 11 + seed).shuffle(configs)
            mapping = dict(zip(labels, configs))
            paths = {label: spec["root"] / f"u{uid}_{config}_s{seed}.png" for label, config in mapping.items()}
            for path in paths.values():
                if not path.exists():
                    raise FileNotFoundError(path)
            cases.append((f"u{uid}_s{seed}_p{pass_id}", uid, seed, pass_id, mapping, paths))

    raw_path = DATA / f"gemini3flash_{args.experiment}_blind_raw.json"
    summary_path = DATA / f"gemini3flash_{args.experiment}_blind_summary.json"
    raw = {"protocol": {"model": MODEL, "experiment": args.experiment, "passes": args.passes, "prompt": PROMPT}, "cases": {}}
    if raw_path.exists() and not args.fresh:
        prior = json.loads(raw_path.read_text())
        if prior.get("protocol", {}).get("model") == MODEL and prior.get("protocol", {}).get("experiment") == args.experiment:
            raw = prior
    todo = [case for case in cases if raw["cases"].get(case[0], {}).get("status") != "ok"]
    print(f"[blind-{args.experiment}] total={len(cases)} remaining={len(todo)}", flush=True)
    key = load_key()

    def work(case):
        case_id, uid, seed, pass_id, mapping, paths = case
        result = call(REF_ROOT / f"u{uid}_ref.png", paths, key)
        return case_id, {"uid": uid, "seed": seed, "pass": pass_id, "mapping": mapping, **result}

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, case) for case in todo]
        for index, future in enumerate(as_completed(futures), 1):
            case_id, record = future.result()
            raw["cases"][case_id] = record
            raw_path.write_text(json.dumps(raw, indent=2))
            print(f"  [{index}/{len(todo)}] {case_id}: {record['status']}", flush=True)
    summary = summarize(raw, spec["configs"], args.passes)
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps({"agreement": summary["agreement"], "majority": summary["majority"], "errors": summary["errors"]}, indent=2))
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()

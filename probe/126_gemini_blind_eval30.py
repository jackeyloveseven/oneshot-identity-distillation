#!/usr/bin/env python3
"""Blinded multi-pass VLM audit for the Krea2 identity-distillation eval30 run."""

from __future__ import annotations

import argparse
import base64
import importlib.util
import io
import json
import os
import random
import re
import statistics
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from PIL import Image


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
PROTOCOL_PATH = ROOT / "data" / "krea2_eval30_protocol.json"
OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
KEY_SOURCES = (
    Path("/home/ubuntu/smoke_anime/flash_caption_smoke.py"),
    Path("/home/ubuntu/.codex/skills/kaon-router/SKILL.md"),
)
BASE_URL = "https://kaon-router.kaonai.com/v1"
MODEL = "kaon/gemini-3-flash-preview"
CONFIGS = ("caption_base", "distilled", "teacher")

PROMPT = """You are performing a blinded evaluation of one-shot anime identity personalization.
The first image is the REFERENCE identity. The following lettered images are anonymous generated candidates for the supplied target scene.

Score EACH candidate independently:
- identity: yes = clearly the same subject, including distinctive face geometry, hair shape and color, eyes, skin tone, scars, masks, ears, or accessories; partial = recognizable overlap but an important identity cue is missing or changed; no = a different subject. Ignore pose, outfit, expression, background, rendering style, and small age shifts.
- visible_defect: yes only for duplicated or collaged sub-images, malformed or extra anatomy, broken face or limbs, severe stripe or color corruption, or structurally incoherent composition. Identity mismatch alone is not a defect.
- scene_fidelity: 2 = the requested scene, framing, action, and single-subject constraint are substantially satisfied; 1 = partly satisfied with a visible omission or mismatch; 0 = substantially wrong scene or composition.

Do not infer methods. Return ONLY one JSON object with exactly the supplied letter keys. Each value must be:
{"identity":"yes|partial|no","visible_defect":"yes|no","scene_fidelity":0|1|2,"confidence":0..1,"note":"brief visible evidence"}
"""


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


RUNNER = load_module(
    "krea_eval30_runner", ROOT / "probe" / "121_run_krea2_eval30.py"
)


def load_key() -> str:
    if os.environ.get("KAON_API_KEY"):
        return os.environ["KAON_API_KEY"]
    patterns = (
        r"API Key[^`]*`(sk-kaon-[^`]+)`",
        r"KAON_KEY\s*=\s*[\"']([^\"']+)",
    )
    for source in KEY_SOURCES:
        if not source.is_file():
            continue
        text = source.read_text()
        for pattern in patterns:
            match = re.search(pattern, text)
            if match:
                return match.group(1)
    raise RuntimeError("set KAON_API_KEY for the blinded VLM audit")


def data_uri(path: Path, px: int = 640) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((px, px))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=86)
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/jpeg;base64," + encoded


def extract_json(text: str, labels: list[str]) -> dict:
    match = re.search(r"\{.*\}", text.strip(), re.S)
    if not match:
        raise ValueError(f"no JSON object: {text[:180]!r}")
    result = json.loads(match.group(0))
    if set(result) != set(labels):
        raise ValueError(f"expected {labels}, got {sorted(result)}")
    for label in labels:
        verdict = result[label]
        if verdict.get("identity") not in {"yes", "partial", "no"}:
            raise ValueError(f"bad identity verdict: {verdict}")
        if verdict.get("visible_defect") not in {"yes", "no"}:
            raise ValueError(f"bad defect verdict: {verdict}")
        if verdict.get("scene_fidelity") not in {0, 1, 2}:
            raise ValueError(f"bad scene verdict: {verdict}")
    return result


def call(
    reference: Path,
    candidates: dict[str, Path],
    scene_prompt: str,
    key: str,
    retries: int = 5,
) -> dict:
    labels = list(candidates)
    content = [
        {
            "type": "text",
            "text": (
                PROMPT
                + "\nTarget scene: "
                + scene_prompt
                + "\nSupplied keys: "
                + ", ".join(labels)
            ),
        },
        {"type": "text", "text": "REFERENCE"},
        {"type": "image_url", "image_url": {"url": data_uri(reference)}},
    ]
    for label, path in candidates.items():
        content.extend(
            [
                {"type": "text", "text": f"Candidate {label}"},
                {"type": "image_url", "image_url": {"url": data_uri(path)}},
            ]
        )
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 1600,
        "temperature": 0.2,
    }
    last_error = ""
    for attempt in range(retries + 1):
        try:
            response = requests.post(
                BASE_URL + "/chat/completions",
                headers={
                    "Authorization": "Bearer " + key,
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=300,
            )
            if response.status_code in {408, 429} or response.status_code >= 500:
                last_error = (
                    f"HTTP {response.status_code}: {response.text[:180]}"
                )
                time.sleep(3 * (attempt + 1))
                continue
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return {
                "status": "ok",
                "verdicts_by_label": extract_json(text, labels),
                "raw_response": text,
                "request_id": response.headers.get("X-Request-ID"),
            }
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"[:400]
            time.sleep(2 * (attempt + 1))
    return {"status": "error", "error": last_error}


def fleiss(rows: list[list[str]], categories: list[str]) -> float | None:
    if not rows or len(rows[0]) < 2:
        return None
    raters = len(rows[0])
    agreements = []
    totals = Counter()
    for row in rows:
        counts = Counter(row)
        agreements.append(
            (sum(value * value for value in counts.values()) - raters)
            / (raters * (raters - 1))
        )
        totals.update(row)
    observed = statistics.mean(agreements)
    count = len(rows) * raters
    expected = sum((totals[category] / count) ** 2 for category in categories)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def majority(values: list, order: list) -> object:
    counts = Counter(values)
    best_count = max(counts.values())
    return next(value for value in order if counts[value] == best_count)


def summarize(raw: dict, passes: int) -> dict:
    grouped: dict[tuple[str, str, int, str], list[dict]] = {}
    errors = []
    for case_id, record in raw["cases"].items():
        if record.get("status") != "ok":
            errors.append(case_id)
            continue
        for label, config in record["mapping"].items():
            key = (
                record["uid"],
                record["scene"],
                record["seed"],
                config,
            )
            grouped.setdefault(key, []).append(
                record["verdicts_by_label"][label]
            )

    identity_rows = []
    defect_rows = []
    majority_items = {config: [] for config in CONFIGS}
    for (uid, scene, seed, config), verdicts in sorted(grouped.items()):
        if len(verdicts) != passes:
            continue
        identities = [item["identity"] for item in verdicts]
        defects = [item["visible_defect"] for item in verdicts]
        scenes = [item["scene_fidelity"] for item in verdicts]
        identity_rows.append(identities)
        defect_rows.append(defects)
        majority_items[config].append(
            {
                "uid": uid,
                "scene": scene,
                "seed": seed,
                "identity": majority(
                    identities, ["no", "partial", "yes"]
                ),
                "visible_defect": majority(defects, ["yes", "no"]),
                "scene_fidelity": majority(scenes, [0, 1, 2]),
            }
        )

    def metrics(items: list[dict]) -> dict:
        count = len(items)
        identity_value = {"no": 0.0, "partial": 0.5, "yes": 1.0}
        return {
            "n": count,
            "identity_counts": dict(
                Counter(item["identity"] for item in items)
            ),
            "identity_score": statistics.mean(
                identity_value[item["identity"]] for item in items
            )
            if items
            else None,
            "defect_rate": sum(
                item["visible_defect"] == "yes" for item in items
            )
            / count
            if items
            else None,
            "scene_fidelity": statistics.mean(
                item["scene_fidelity"] for item in items
            )
            if items
            else None,
            "clean_identity_rate": sum(
                item["identity"] == "yes"
                and item["visible_defect"] == "no"
                for item in items
            )
            / count
            if items
            else None,
            "clean_usable_rate": sum(
                item["identity"] in {"yes", "partial"}
                and item["visible_defect"] == "no"
                for item in items
            )
            / count
            if items
            else None,
        }

    return {
        "model": MODEL,
        "definition": "anonymous multi-pass VLM audit; not human evaluation",
        "passes": passes,
        "errors": sorted(errors),
        "agreement": {
            "identity_fleiss_kappa": fleiss(
                identity_rows, ["yes", "partial", "no"]
            ),
            "defect_fleiss_kappa": fleiss(defect_rows, ["yes", "no"]),
            "identity_unanimous_rate": sum(
                len(set(row)) == 1 for row in identity_rows
            )
            / len(identity_rows)
            if identity_rows
            else None,
            "defect_unanimous_rate": sum(
                len(set(row)) == 1 for row in defect_rows
            )
            / len(defect_rows)
            if defect_rows
            else None,
        },
        "majority": {
            config: metrics(items)
            for config, items in majority_items.items()
        },
        "majority_items": majority_items,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--passes", type=int, default=3)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    if args.passes < 1 or args.workers < 1:
        raise ValueError("passes and workers must be positive")

    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    observations = [
        (job, seed)
        for job in protocol["jobs"]
        for seed in job["seeds"]
    ]
    if args.limit is not None:
        observations = observations[: args.limit]

    cases = []
    for pass_id in range(args.passes):
        for job, seed in observations:
            metadata_path = RUNNER.expected_metadata(job, seed, OUT)
            if not metadata_path.is_file():
                raise FileNotFoundError(
                    f"main run incomplete; missing {metadata_path}"
                )
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            configs = list(CONFIGS)
            random.Random(
                20260727
                + pass_id * 100000
                + int(job["uid"]) * 17
                + seed * 31
                + sum(ord(char) for char in job["scene"])
            ).shuffle(configs)
            labels = [chr(ord("A") + index) for index in range(len(configs))]
            mapping = dict(zip(labels, configs))
            candidates = {
                label: Path(metadata["outputs"][config])
                for label, config in mapping.items()
            }
            case_id = (
                f"u{job['uid']}_{job['scene']}_s{seed}_p{pass_id}"
            )
            cases.append(
                (
                    case_id,
                    job,
                    seed,
                    pass_id,
                    mapping,
                    candidates,
                )
            )

    raw_path = OUT / "gemini3flash_blind_raw.json"
    summary_path = OUT / "gemini3flash_blind_summary.json"
    raw = {
        "protocol": {
            "model": MODEL,
            "passes": args.passes,
            "configs": list(CONFIGS),
            "prompt": PROMPT,
            "observation_count": len(observations),
        },
        "cases": {},
    }
    if raw_path.is_file() and not args.fresh:
        prior = json.loads(raw_path.read_text(encoding="utf-8"))
        prior_protocol = prior.get("protocol", {})
        if (
            prior_protocol.get("model") == MODEL
            and prior_protocol.get("passes") == args.passes
            and prior_protocol.get("configs") == list(CONFIGS)
        ):
            raw = prior
    pending = [
        case
        for case in cases
        if raw["cases"].get(case[0], {}).get("status") != "ok"
    ]
    print(f"[blind-eval30] total={len(cases)} remaining={len(pending)}")
    key = load_key()

    def work(case: tuple) -> tuple[str, dict]:
        case_id, job, seed, pass_id, mapping, candidates = case
        result = call(
            Path(job["reference"]),
            candidates,
            job["scene_prompt"],
            key,
        )
        return case_id, {
            "uid": job["uid"],
            "scene": job["scene"],
            "seed": seed,
            "pass": pass_id,
            "mapping": mapping,
            **result,
        }

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(work, case) for case in pending]
        for index, future in enumerate(as_completed(futures), 1):
            case_id, record = future.result()
            raw["cases"][case_id] = record
            raw_path.write_text(
                json.dumps(raw, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(
                f"[{index}/{len(pending)}] {case_id}: {record['status']}",
                flush=True,
            )

    summary = summarize(raw, args.passes)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "agreement": summary["agreement"],
                "majority": summary["majority"],
                "errors": summary["errors"],
            },
            indent=2,
        )
    )
    print(f"raw: {raw_path}")
    print(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

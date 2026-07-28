#!/usr/bin/env python3
"""Extract a concise cross-reference identity capsule with Gemini 3 Flash.

The capsule is a semantic coordinate system for the frozen T2I base. It keeps
only stable visible subject attributes and excludes pose, expression,
background, framing, lighting, and attributes that conflict across references.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import requests
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "out" / "identity_capsules"
DEFAULT_BASE_URL = "https://kaon-router.kaonai.com/v1"
DEFAULT_MODEL = "kaon/gemini-3-flash-preview"
KEY_SOURCE = Path("/home/ubuntu/smoke_anime/flash_caption_smoke.py")

SYSTEM_PROMPT = """You extract stable visual identity from multiple reference images of ONE subject.
Compare all images before answering. Keep only attributes that identify the same subject across views:
species, age group, gender presentation, face shape, skin/fur tone, eye color and shape, hair color/length/style,
permanent marks, signature ears/horns/masks, and highly distinctive recurring accessories or clothing.

Exclude pose, expression, action, camera angle, crop, body position, background, lighting, color cast, image quality,
and any detail that appears in only one image or conflicts across images. Do not name an art style unless it is
needed to distinguish a non-photorealistic subject. Do not infer personality, ethnicity, or hidden anatomy.

The generation capsule must be compact: 12-28 English words, permanent traits plus at most TWO signature
accessory/clothing cues. Never enumerate a full outfit or palette. Put other recurring but nonessential details
in optional_signature_cues; they are audit information and will not be sent to the image generator. Avoid repeating
the same color word: if hair and eyes share a hue, use a neutral light/dark hair description and state the precise
color only once. Use woman/man rather than girl/boy when the subject visibly appears adult.

Return ONLY valid JSON with this schema:
{
  "capsule": "compact comma-separated English generation phrase, 12-28 words, no sentence",
  "stable_attributes": ["short visible fact", "..."],
  "optional_signature_cues": ["recurring but nonessential visible detail", "..."],
  "excluded_conflicts": ["detail excluded because references disagree", "..."],
  "confidence": 0.0
}
The capsule must explicitly state gender presentation or non-human species when visible, because this prevents
the text-to-image prior from drifting to a different subject class."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uid", required=True)
    parser.add_argument("--references", required=True, help="comma-separated paths to the same subject")
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output", default=None)
    parser.add_argument("--max-attempts", type=int, default=4)
    return parser.parse_args()


def load_api_key() -> str:
    if os.environ.get("KAON_API_KEY"):
        return os.environ["KAON_API_KEY"]
    if KEY_SOURCE.is_file():
        match = re.search(r"KAON_KEY\s*=\s*[\"']([^\"']+)", KEY_SOURCE.read_text())
        if match:
            return match.group(1)
    skill_path = Path.home() / ".codex" / "skills" / "kaon-router" / "SKILL.md"
    if skill_path.is_file():
        match = re.search(r"API Key\*\*: `(sk-[^`]+)`", skill_path.read_text())
        if match:
            return match.group(1)
    raise RuntimeError("Set KAON_API_KEY or provide the existing local Kaon credential")


def data_uri(path: Path, max_px: int = 896) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((max_px, max_px))
        buffer = io.BytesIO()
        image.save(buffer, "JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def extract_json(text: str) -> dict[str, object]:
    match = re.search(r"\{.*\}", text.strip(), re.S)
    if not match:
        raise ValueError(f"response contains no JSON object: {text[:200]!r}")
    payload = json.loads(match.group(0))
    capsule = payload.get("capsule")
    if not isinstance(capsule, str) or not capsule.strip():
        raise ValueError("response has no non-empty capsule")
    if not isinstance(payload.get("stable_attributes"), list):
        raise ValueError("stable_attributes must be a list")
    if not isinstance(payload.get("optional_signature_cues"), list):
        raise ValueError("optional_signature_cues must be a list")
    if not isinstance(payload.get("excluded_conflicts"), list):
        raise ValueError("excluded_conflicts must be a list")
    confidence = float(payload.get("confidence", 0.0))
    payload["confidence"] = min(max(confidence, 0.0), 1.0)
    payload["capsule"] = capsule.strip().rstrip(".")
    words = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?", payload["capsule"])
    if not 12 <= len(words) <= 28:
        raise ValueError(f"capsule must contain 12-28 words, got {len(words)}")
    color_words = {
        "black", "white", "grey", "gray", "red", "orange", "yellow", "green",
        "blue", "purple", "violet", "pink", "brown", "blonde", "silver", "golden",
    }
    repeated = sorted({word.lower() for word in words if word.lower() in color_words and sum(
        candidate.lower() == word.lower() for candidate in words
    ) > 1})
    if repeated:
        raise ValueError(f"capsule repeats color words: {repeated}")
    return payload


def call_model(args: argparse.Namespace, paths: list[Path]) -> tuple[dict[str, object], str | None]:
    content: list[dict[str, object]] = [
        {
            "type": "text",
            "text": (
                f"Subject identifier: {args.name or 'unknown'}. "
                f"There are {len(paths)} references of the same subject. Extract the stable identity capsule."
            ),
        }
    ]
    for index, path in enumerate(paths, start=1):
        content.append({"type": "text", "text": f"REFERENCE {index}"})
        content.append({"type": "image_url", "image_url": {"url": data_uri(path)}})

    session = requests.Session()
    failures = []
    for attempt in range(1, args.max_attempts + 1):
        response = session.post(
            f"{args.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {load_api_key()}", "Content-Type": "application/json"},
            json={
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": content},
                ],
                "temperature": 0.0,
                "max_tokens": 1200,
            },
            timeout=180,
        )
        request_id = response.headers.get("x-request-id")
        if response.ok:
            text = response.json()["choices"][0]["message"]["content"]
            try:
                return extract_json(text), request_id
            except (ValueError, json.JSONDecodeError) as error:
                failures.append({"attempt": attempt, "status": 200, "body": str(error)})
                if attempt < args.max_attempts:
                    time.sleep(2 * attempt)
                    continue
                break
        failures.append({"attempt": attempt, "status": response.status_code, "body": response.text[:500]})
        if response.status_code not in {429, 500, 502, 503, 504}:
            break
        time.sleep(5 * attempt)
    raise RuntimeError(f"Kaon capsule request failed: {failures}")


def main() -> int:
    args = parse_args()
    paths = [Path(item.strip()).expanduser().resolve() for item in args.references.split(",") if item.strip()]
    if len(paths) < 2:
        raise ValueError("identity capsule extraction requires at least two same-subject references")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    payload, request_id = call_model(args, paths)
    payload.update(
        {
            "uid": str(args.uid),
            "name": args.name,
            "references": [str(path) for path in paths],
            "model": args.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "request_id": request_id,
            "intent": "stable multi-reference identity semantics for frozen-T2I residual distillation",
            "prompt_version": "core_capsule_v3_color_guard",
        }
    )
    output = Path(args.output).expanduser().resolve() if args.output else OUT_ROOT / f"u{args.uid}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved {output}")
    print(f"capsule: {payload['capsule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

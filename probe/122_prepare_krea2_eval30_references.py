#!/usr/bin/env python3
"""Prepare deterministic square identity views for the Krea2 eval30 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from imgutils.detect import detect_heads
from PIL import Image


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
IMAGE_ROOT = Path("/mnt/local/smoke_anime/anime_260623/img")
PROTOCOL = ROOT / "data" / "krea2_eval30_protocol.json"
DEFAULT_OUT = ROOT / "out" / "krea2_eval30_refs"
DEFAULT_METADATA = ROOT / "data" / "krea2_eval30_reference_preprocessing.json"
PADDING_RATIO = 0.35
OUTPUT_SIZE = 768


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def square_identity_crop(
    image: Image.Image,
    box: tuple[int, int, int, int],
    padding_ratio: float,
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    x0, y0, x1, y1 = box
    width = x1 - x0
    height = y1 - y0
    center_x = (x0 + x1) / 2
    center_y = (y0 + y1) / 2
    side = max(width, height) * (1.0 + 2.0 * padding_ratio)
    left = round(center_x - side / 2)
    top = round(center_y - side / 2)
    right = round(center_x + side / 2)
    bottom = round(center_y + side / 2)

    canvas = Image.new("RGB", (right - left, bottom - top), "white")
    source_box = (
        max(0, left),
        max(0, top),
        min(image.width, right),
        min(image.height, bottom),
    )
    source_crop = image.crop(source_box)
    canvas.paste(source_crop, (source_box[0] - left, source_box[1] - top))
    return canvas, (left, top, right, bottom)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for uid in protocol["evaluation_uids"]:
        source = IMAGE_ROOT / f"u{uid}_whiteref.png"
        detections = detect_heads(str(source))
        if not detections:
            raise RuntimeError(f"no head detected: {source}")
        box, label, confidence = max(
            detections,
            key=lambda detection: (
                (detection[0][2] - detection[0][0])
                * (detection[0][3] - detection[0][1])
            ),
        )
        image = Image.open(source).convert("RGB")
        crop, crop_box = square_identity_crop(image, box, PADDING_RATIO)
        crop = crop.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.Resampling.LANCZOS)
        output = args.out_dir / f"u{uid}_identity.png"
        crop.save(output)
        records.append(
            {
                "uid": uid,
                "source": str(source),
                "source_sha256": sha256(source),
                "source_size": list(image.size),
                "detector_label": label,
                "detector_confidence": float(confidence),
                "head_box": list(box),
                "crop_box_before_padding": list(crop_box),
                "padding_ratio": PADDING_RATIO,
                "output_size": [OUTPUT_SIZE, OUTPUT_SIZE],
                "output": str(output),
                "output_sha256": sha256(output),
            }
        )

    metadata = {
        "method": (
            "largest anime-head detection; square crop with fixed 0.35 "
            "head-box padding; white out-of-bounds fill; 768x768 resize"
        ),
        "input_only": True,
        "result_independent": True,
        "count": len(records),
        "records": records,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in metadata.items() if key != "records"}, indent=2))
    print(f"metadata: {args.metadata}")
    print(f"references: {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build the locked 30-subject Krea2 identity-distillation protocol.

The evaluation pool contains human 2D anime/manga references selected by an
input-only domain audit from two pre-existing subject sets.  It excludes the
three development identities used for the Krea2 v1.2 pilot. The protocol
separates a class-only prompt, a text-only automatic visible-attribute prompt,
and a reference-conditioned teacher instruction. Wrong-reference controls are
deterministically matched within the same subject class.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
IMAGE_ROOT = Path("/mnt/local/smoke_anime/anime_260623/img")
REFERENCE_ROOT = ROOT / "out" / "krea2_eval30_refs"
TAG_PATHS = (
    ROOT / "data" / "repprobe30_tags.json",
    ROOT / "data" / "holdout30_tags.json",
)
DEFAULT_OUTPUT = ROOT / "data" / "krea2_eval30_protocol.json"
DEVELOPMENT_UIDS = {"1839", "4229", "9728"}
SEEDS = (2027, 2028)

# Fixed before the large Krea2 run by inspecting reference images only.
# Inclusion criterion: one human 2D anime/manga subject with visible identity
# cues.  Exclusions cover photographic/3D, non-human, and development inputs.
EVALUATION_UIDS = (
    "5319", "2364", "1391", "1070", "4977", "5064", "9993", "6456",
    "4096", "2407", "4017", "1740", "4573", "6371", "943", "5844",
    "4224", "68", "3802", "5494",
    "2542", "4073", "1317", "1620", "3090", "5412", "6159", "7122",
    "9564", "192",
)

SCENES = {
    "portrait": {
        "width": 640,
        "height": 640,
        "description": (
            "in a centered close-up head-and-shoulders portrait inside a "
            "modern glass atrium, soft daylight, detailed face, looking toward "
            "the camera, coherent anatomy, no other people"
        ),
    },
    "rain": {
        "width": 512,
        "height": 768,
        "description": (
            "standing on a rain-soaked neon city street at night, three-quarter "
            "body, holding a transparent umbrella, cinematic reflections, "
            "detailed visible face, coherent anatomy, no other people"
        ),
    },
    "kitchen": {
        "width": 768,
        "height": 512,
        "description": (
            "preparing tea at a bright contemporary kitchen counter, waist-up, "
            "hands visible, turning toward the camera, warm morning light, "
            "detailed visible face, coherent anatomy, no other people"
        ),
    },
}

LOCKED_METHOD = {
    "steps": 12,
    "inner_iters": 3,
    "lr": 0.05,
    "pseudo_step_size": 1.0,
    "teacher_weight": 1.0,
    "active_start": 0.0,
    "active_end": 1.0,
    "trust_region_rms": 0.10,
    "local_trust_region_rms": 0.085,
    "adaptive_trust_ratio": 3.0,
    "normalize": "rms_token",
    "residual_mode": "raw",
    "ref_boost": 4.0,
    "distill_roi": None,
    "outside_weight": 1.0,
    "deployment_conditioning": "text_only_automatic_visible_attributes",
}


def subject_class(record: dict) -> tuple[str, str, str]:
    struct = record["struct"]
    girl = float(struct.get("1girl", 0.0))
    boy = float(struct.get("1boy", 0.0))
    if girl == boy == 0:
        phrase = record.get("phrase", "").lower()
        if "woman" in phrase or "girl" in phrase:
            girl = 1.0
        elif "man" in phrase or "boy" in phrase:
            boy = 1.0
        else:
            raise ValueError("subject has neither a class tag nor class phrase")
    if girl >= boy:
        return "woman", "her", "her"
    return "man", "his", "him"


def build_protocol() -> dict:
    tags = {}
    source_by_uid = {}
    for tag_path in TAG_PATHS:
        source = json.loads(tag_path.read_text(encoding="utf-8"))
        for uid, record in source.items():
            if uid in tags and tags[uid] != record:
                raise AssertionError(f"conflicting tag records for {uid}")
            tags[uid] = record
            source_by_uid[uid] = str(tag_path)
    missing_uids = set(EVALUATION_UIDS).difference(tags)
    if missing_uids:
        raise KeyError(f"evaluation identities missing tags: {sorted(missing_uids)}")
    selected = {uid: tags[uid] for uid in EVALUATION_UIDS}
    if len(selected) != 30:
        raise AssertionError(f"expected 30 evaluation identities, found {len(selected)}")
    overlap = DEVELOPMENT_UIDS.intersection(selected)
    if overlap:
        raise AssertionError(f"evaluation/development overlap: {sorted(overlap)}")

    grouped: dict[str, list[str]] = {"woman": [], "man": []}
    classes: dict[str, tuple[str, str, str]] = {}
    for uid, record in selected.items():
        classes[uid] = subject_class(record)
        grouped[classes[uid][0]].append(uid)

    negative_for: dict[str, str] = {}
    for class_word, uids in grouped.items():
        if len(uids) < 2:
            raise AssertionError(f"not enough {class_word} identities for controls")
        for index, uid in enumerate(uids):
            negative_for[uid] = uids[(index + 1) % len(uids)]

    jobs = []
    for uid in selected:
        class_word, possessive, object_pronoun = classes[uid]
        reference = REFERENCE_ROOT / f"u{uid}_identity.png"
        negative_uid = negative_for[uid]
        negative_reference = REFERENCE_ROOT / f"u{negative_uid}_identity.png"
        if not reference.is_file() or not negative_reference.is_file():
            raise FileNotFoundError(reference if not reference.is_file() else negative_reference)

        identity_phrase = selected[uid]["phrase"]
        negative_phrase = selected[negative_uid]["phrase"]
        for scene_name, scene in SCENES.items():
            scene_prompt = f"single anime {class_word} {scene['description']}"
            caption_prompt = (
                f"Create a high-quality anime illustration of one {class_word} "
                f"as the only subject. Visible attributes: {identity_phrase}. "
                f"Depict {object_pronoun} {scene['description']}. "
                "No collage and no split panel."
            )
            teacher_prompt = (
                f"Create a high-quality anime illustration of the exact "
                f"{class_word} shown in the reference image as the only subject. "
                f"Automatically extracted visible attributes: {identity_phrase}. "
                "Use the reference image for identity details beyond this short "
                "attribute list. "
                f"Preserve {possessive} identity without inventing a different "
                f"person. Depict {object_pronoun} {scene['description']}. "
                "Do not copy the reference background, framing, or pose. "
                "No collage and no split panel."
            )
            negative_teacher_prompt = (
                f"Create a high-quality anime illustration of the exact "
                f"{class_word} shown in the reference image as the only subject. "
                f"Automatically extracted visible attributes: {negative_phrase}. "
                "Use the reference image for identity details beyond this short "
                "attribute list. "
                f"Preserve {possessive} identity without inventing a different "
                f"person. Depict {object_pronoun} {scene['description']}. "
                "Do not copy the reference background, framing, or pose. "
                "No collage and no split panel."
            )
            jobs.append(
                {
                    "job_id": f"u{uid}_eval30t_{scene_name}",
                    "uid": uid,
                    "source_set": source_by_uid[uid],
                    "subject_class": class_word,
                    "scene": scene_name,
                    "reference": str(reference),
                    "negative_uid": negative_uid,
                    "negative_reference": str(negative_reference),
                    "width": scene["width"],
                    "height": scene["height"],
                    "scene_prompt": scene_prompt,
                    "caption_prompt": caption_prompt,
                    "teacher_prompt": teacher_prompt,
                    "negative_teacher_prompt": negative_teacher_prompt,
                    "seeds": list(SEEDS),
                }
            )

    return {
        "name": "krea2_eval30_tagged_teacher_v3",
        "purpose": (
            "held-out evaluation of edit-to-generation test-time identity "
            "score distillation"
        ),
        "development_uids": sorted(DEVELOPMENT_UIDS),
        "evaluation_sources": [str(path) for path in TAG_PATHS],
        "selection_rule": (
            "input-only audit: single human 2D anime/manga subject with visible "
            "identity cues; exclude photographic/3D, non-human, and development inputs"
        ),
        "evaluation_uids": list(EVALUATION_UIDS),
        "identity_count": len(selected),
        "scene_count": len(SCENES),
        "seeds": list(SEEDS),
        "paired_observation_count": len(jobs) * len(SEEDS),
        "negative_control": "deterministic same-class cyclic derangement",
        "reference_policy": (
            "deterministic head-and-context crop from a foreground-isolated "
            "white-background one-shot reference; no manually authored identity "
            "caption"
        ),
        "teacher_identity_text": (
            "pre-existing automatically extracted visible-attribute phrase; "
            "used in the text-only baseline and teacher, never the class-only "
            "deployment base"
        ),
        "reference_preprocessing": str(
            ROOT / "data" / "krea2_eval30_reference_preprocessing.json"
        ),
        "method": LOCKED_METHOD,
        "jobs": jobs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    protocol = build_protocol()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in protocol.items() if key != "jobs"}, indent=2))
    print(f"protocol: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

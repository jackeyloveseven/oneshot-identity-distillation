#!/usr/bin/env python3
"""Score and visualize the locked Krea2 eval30 identity-distillation run."""

from __future__ import annotations

import csv
import html
import json
import math
import os
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import open_clip
import torch
from imgutils.detect import detect_heads
from imgutils.metrics import (
    anime_dbaesthetic,
    ccip_difference,
    ccip_extract_feature,
)
from PIL import Image


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
PROTOCOL_PATH = ROOT / "data" / "krea2_eval30_protocol.json"
OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
HEAD_OUT = OUT / "head_crops"
METRICS_PATH = OUT / "eval30_metrics.json"
CSV_PATH = OUT / "eval30_metrics.csv"
REPORT_PATH = OUT / "eval30_report.html"
VLM_SUMMARY_PATH = OUT / "gemini3flash_blind_summary.json"
THRESHOLD = 0.178
KINDS = (
    "base",
    "caption_base",
    "matched_control",
    "distilled",
    "teacher",
    "negative_teacher",
)
HEAD_KINDS = ("base", "caption_base", "distilled", "teacher")


@lru_cache(maxsize=None)
def ccip_feature(path: str) -> np.ndarray:
    return ccip_extract_feature(path)


def ccip(left: str | Path, right: str | Path) -> float:
    return float(
        ccip_difference(ccip_feature(str(left)), ccip_feature(str(right)))
    )


def expected_metadata(job: dict, seed: int, method: dict) -> Path:
    suffix = (
        f"s{seed}_{job['width']}x{job['height']}_"
        f"i{method['inner_iters']}_p{method['pseudo_step_size']:g}_"
        f"g{method['trust_region_rms']:g}_"
        f"l{method['local_trust_region_rms']:g}_"
        f"a{method['adaptive_trust_ratio']:g}_"
        f"rb{method['ref_boost']:g}_"
        f"{method['normalize']}_{method['residual_mode']}_dcap_roifull_o1.json"
    )
    return OUT / f"{job['job_id']}_{suffix}"


def check_metadata(metadata: dict, job: dict, seed: int, protocol: dict) -> None:
    method = protocol["method"]
    expected = {
        "reference": job["reference"],
        "negative_reference": job["negative_reference"],
        "seed": seed,
        "size": [job["width"], job["height"]],
        "steps": method["steps"],
        "inner_iters": method["inner_iters"],
        "lr": method["lr"],
        "pseudo_step_size": method["pseudo_step_size"],
        "teacher_weight": method["teacher_weight"],
        "active_window": [method["active_start"], method["active_end"]],
        "normalize": method["normalize"],
        "residual_mode": method["residual_mode"],
        "trust_region_rms": method["trust_region_rms"],
        "local_trust_region_rms": method["local_trust_region_rms"],
        "adaptive_trust_ratio": method["adaptive_trust_ratio"],
        "ref_boost": method["ref_boost"],
        "scene_prompt": job["scene_prompt"],
        "caption_prompt": job["caption_prompt"],
        "caption_conditioning": "text_only_no_reference_tokens_or_latents",
        "deployment_conditioning": method["deployment_conditioning"],
        "teacher_prompt": job["teacher_prompt"],
        "negative_teacher_prompt": job["negative_teacher_prompt"],
    }
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise AssertionError(
                f"{job['job_id']} seed {seed}: {key} expected {value!r}, "
                f"found {metadata.get(key)!r}"
            )
    if metadata.get("teacher_scheduler_usage") != "none_in_distilled_trajectory":
        raise AssertionError(f"{job['job_id']} seed {seed}: teacher entered scheduler")
    if set(metadata["outputs"]) != set(KINDS):
        raise AssertionError(
            f"{job['job_id']} seed {seed}: output arms are "
            f"{sorted(metadata['outputs'])}, expected {sorted(KINDS)}"
        )


def largest_head_crop(path: Path, output: Path, pad_ratio: float = 0.12) -> int:
    image = Image.open(path).convert("RGB")
    detections = detect_heads(str(path))
    if detections:
        x0, y0, x1, y1 = max(
            (detection[0] for detection in detections),
            key=lambda box: (box[2] - box[0]) * (box[3] - box[1]),
        )
        pad_x = round((x1 - x0) * pad_ratio)
        pad_y = round((y1 - y0) * pad_ratio)
        crop = image.crop(
            (
                max(0, x0 - pad_x),
                max(0, y0 - pad_y),
                min(image.width, x1 + pad_x),
                min(image.height, y1 + pad_y),
            )
        )
    else:
        side = min(image.size)
        left = (image.width - side) // 2
        top = max(0, (image.height - side) // 3)
        crop = image.crop((left, top, left + side, top + side))
    if not output.is_file():
        output.parent.mkdir(parents=True, exist_ok=True)
        crop.resize((384, 384), Image.Resampling.LANCZOS).save(output)
    return len(detections)


def pixel_l1(left_path: Path, right_path: Path) -> float:
    left = np.asarray(
        Image.open(left_path).convert("RGB").resize(
            (256, 256), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    right = np.asarray(
        Image.open(right_path).convert("RGB").resize(
            (256, 256), Image.Resampling.LANCZOS
        ),
        dtype=np.float32,
    )
    return float(np.abs(left - right).mean() / 255.0)


def collect_rows(protocol: dict) -> list[dict]:
    rows = []
    missing = []
    for job in protocol["jobs"]:
        for seed in job["seeds"]:
            metadata_path = expected_metadata(job, seed, protocol["method"])
            if not metadata_path.is_file():
                missing.append(str(metadata_path))
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            check_metadata(metadata, job, seed, protocol)
            paths = {kind: Path(metadata["outputs"][kind]) for kind in KINDS}
            reference = Path(job["reference"])
            negative_reference = Path(job["negative_reference"])
            for path in (reference, negative_reference, *paths.values()):
                if not path.is_file():
                    raise FileNotFoundError(path)

            row = {
                "uid": job["uid"],
                "subject_class": job["subject_class"],
                "scene": job["scene"],
                "seed": seed,
                "reference": str(reference),
                "negative_uid": job["negative_uid"],
                "negative_reference": str(negative_reference),
                "scene_prompt": job["scene_prompt"],
                "caption_prompt": job["caption_prompt"],
                **{kind: str(path) for kind, path in paths.items()},
            }
            for kind, path in paths.items():
                row[f"{kind}_ccip"] = ccip(reference, path)
                aesthetic_label, aesthetic_percentile = anime_dbaesthetic(
                    str(path)
                )
                row[f"{kind}_aesthetic_label"] = aesthetic_label
                row[f"{kind}_aesthetic_percentile"] = float(
                    aesthetic_percentile
                )
            for kind in ("caption_base", "distilled", "teacher", "negative_teacher"):
                row[f"{kind}_wrong_ccip"] = ccip(
                    negative_reference, paths[kind]
                )

            reference_head = HEAD_OUT / f"u{job['uid']}_reference.png"
            largest_head_crop(reference, reference_head)
            row["reference_head"] = str(reference_head)
            for kind in HEAD_KINDS:
                output_head = HEAD_OUT / f"{metadata_path.stem}_{kind}.png"
                row[f"{kind}_head_count"] = largest_head_crop(
                    paths[kind], output_head
                )
                row[f"{kind}_head"] = str(output_head)
                row[f"{kind}_head_ccip"] = ccip(
                    reference_head, output_head
                )

            row["gain_vs_base"] = row["base_ccip"] - row["distilled_ccip"]
            row["gain_vs_caption"] = (
                row["caption_base_ccip"] - row["distilled_ccip"]
            )
            row["teacher_gap"] = row["distilled_ccip"] - row["teacher_ccip"]
            row["identity_gain_vs_teacher"] = (
                row["teacher_ccip"] - row["distilled_ccip"]
            )
            row["specificity_margin"] = (
                row["distilled_wrong_ccip"] - row["distilled_ccip"]
            )
            row["negative_switch"] = (
                row["negative_teacher_ccip"]
                - row["negative_teacher_wrong_ccip"]
            )
            row["distilled_pass"] = row["distilled_ccip"] < THRESHOLD
            row["caption_pass"] = row["caption_base_ccip"] < THRESHOLD
            for kind in KINDS:
                row[f"{kind}_pixel_l1_reference"] = pixel_l1(
                    reference, paths[kind]
                )
            rows.append(row)
    if missing:
        raise RuntimeError(
            f"evaluation is incomplete: {len(missing)} metadata files missing; "
            f"first is {missing[0]}"
        )
    return rows


def add_identity_retrieval_metrics(rows: list[dict], protocol: dict) -> None:
    """Evaluate whether each output retrieves its source among all references."""
    reference_for: dict[str, str] = {}
    class_for: dict[str, str] = {}
    for job in protocol["jobs"]:
        reference_for[job["uid"]] = job["reference"]
        class_for[job["uid"]] = job["subject_class"]

    for row in rows:
        correct_uid = row["uid"]
        same_class_uids = [
            uid
            for uid in protocol["evaluation_uids"]
            if class_for[uid] == row["subject_class"]
        ]
        for kind in KINDS:
            output_feature = ccip_feature(row[kind])
            distances = {
                uid: float(
                    ccip_difference(
                        ccip_feature(reference_for[uid]), output_feature
                    )
                )
                for uid in protocol["evaluation_uids"]
            }
            ordered = sorted(distances, key=distances.get)
            same_class_ordered = sorted(
                same_class_uids, key=distances.get
            )
            nearest_wrong = min(
                distance
                for uid, distance in distances.items()
                if uid != correct_uid
            )
            row[f"{kind}_retrieval_rank"] = ordered.index(correct_uid) + 1
            row[f"{kind}_retrieval_top1"] = ordered[0] == correct_uid
            row[f"{kind}_same_class_rank"] = (
                same_class_ordered.index(correct_uid) + 1
            )
            row[f"{kind}_same_class_top1"] = (
                same_class_ordered[0] == correct_uid
            )
            row[f"{kind}_retrieval_margin"] = (
                nearest_wrong - distances[correct_uid]
            )


def add_clip_metrics(rows: list[dict]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model.eval()
    entries = [(row, kind) for row in rows for kind in KINDS]
    batch_size = 64
    for start in range(0, len(entries), batch_size):
        batch = entries[start : start + batch_size]
        images = torch.stack(
            [preprocess(Image.open(row[kind]).convert("RGB")) for row, kind in batch]
        ).to(device)
        references = torch.stack(
            [preprocess(Image.open(row["reference"]).convert("RGB")) for row, _ in batch]
        ).to(device)
        texts = tokenizer([row["scene_prompt"] for row, _ in batch]).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(images)
            reference_features = model.encode_image(references)
            text_features = model.encode_text(texts)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            reference_features = (
                reference_features
                / reference_features.norm(dim=-1, keepdim=True)
            )
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            scene_scores = (
                (image_features * text_features).sum(dim=-1).cpu().tolist()
            )
            reference_scores = (
                (image_features * reference_features).sum(dim=-1).cpu().tolist()
            )
        for (row, kind), scene_score, reference_score in zip(
            batch, scene_scores, reference_scores
        ):
            row[f"{kind}_scene_clip"] = float(scene_score)
            row[f"{kind}_reference_clip"] = float(reference_score)
        print(
            f"[scene-clip] {min(start + len(batch), len(entries))}/{len(entries)}",
            flush=True,
        )


def bootstrap_subject_mean(
    rows: list[dict], key: str, samples: int = 20000
) -> tuple[float, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(float(row[key]))
    values = np.asarray(
        [statistics.mean(grouped[uid]) for uid in sorted(grouped)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260727)
    draws = rng.choice(values, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def sign_test_subjects(rows: list[dict], key: str) -> float:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(float(row[key]))
    values = [
        statistics.mean(grouped[uid])
        for uid in sorted(grouped)
        if statistics.mean(grouped[uid]) != 0
    ]
    positives = sum(value > 0 for value in values)
    n = len(values)
    tail = sum(math.comb(n, k) for k in range(0, min(positives, n - positives) + 1))
    return min(1.0, 2.0 * tail / (2**n))


def mean(rows: list[dict], key: str) -> float:
    return statistics.mean(float(row[key]) for row in rows)


def summarize(rows: list[dict], protocol: dict) -> dict:
    for row in rows:
        row["scene_gain_vs_base"] = (
            row["distilled_scene_clip"] - row["base_scene_clip"]
        )
        row["scene_gain_vs_caption"] = (
            row["distilled_scene_clip"] - row["caption_base_scene_clip"]
        )
        row["scene_gain_vs_teacher"] = (
            row["distilled_scene_clip"] - row["teacher_scene_clip"]
        )
        row["reference_clip_gain_vs_caption"] = (
            row["distilled_reference_clip"]
            - row["caption_base_reference_clip"]
        )
        row["aesthetic_gain_vs_caption"] = (
            row["distilled_aesthetic_percentile"]
            - row["caption_base_aesthetic_percentile"]
        )
        row["retrieval_margin_gain_vs_caption"] = (
            row["distilled_retrieval_margin"]
            - row["caption_base_retrieval_margin"]
        )
        row["retrieval_margin_gain_vs_teacher"] = (
            row["distilled_retrieval_margin"]
            - row["teacher_retrieval_margin"]
        )
    comparisons = {}
    for key in (
        "gain_vs_base",
        "gain_vs_caption",
        "identity_gain_vs_teacher",
        "specificity_margin",
        "negative_switch",
        "scene_gain_vs_base",
        "scene_gain_vs_caption",
        "scene_gain_vs_teacher",
        "reference_clip_gain_vs_caption",
        "aesthetic_gain_vs_caption",
        "retrieval_margin_gain_vs_caption",
        "retrieval_margin_gain_vs_teacher",
    ):
        comparisons[key] = {
            "mean": mean(rows, key),
            "subject_clustered_ci95": list(bootstrap_subject_mean(rows, key)),
            "subject_sign_p_two_sided": sign_test_subjects(rows, key),
        }

    per_scene = {}
    for scene in sorted({row["scene"] for row in rows}):
        subset = [row for row in rows if row["scene"] == scene]
        per_scene[scene] = {
            "n": len(subset),
            "base_ccip": mean(subset, "base_ccip"),
            "caption_base_ccip": mean(subset, "caption_base_ccip"),
            "distilled_ccip": mean(subset, "distilled_ccip"),
            "teacher_ccip": mean(subset, "teacher_ccip"),
            "distilled_head_ccip": mean(subset, "distilled_head_ccip"),
            "base_scene_clip": mean(subset, "base_scene_clip"),
            "caption_base_scene_clip": mean(subset, "caption_base_scene_clip"),
            "distilled_scene_clip": mean(subset, "distilled_scene_clip"),
            "teacher_scene_clip": mean(subset, "teacher_scene_clip"),
            "caption_base_aesthetic_percentile": mean(
                subset, "caption_base_aesthetic_percentile"
            ),
            "distilled_aesthetic_percentile": mean(
                subset, "distilled_aesthetic_percentile"
            ),
            "caption_base_reference_clip": mean(
                subset, "caption_base_reference_clip"
            ),
            "distilled_reference_clip": mean(
                subset, "distilled_reference_clip"
            ),
            "distilled_passes": sum(row["distilled_pass"] for row in subset),
            "distilled_retrieval_top1": sum(
                row["distilled_retrieval_top1"] for row in subset
            ),
        }

    per_identity = {}
    for uid in protocol["evaluation_uids"]:
        subset = [row for row in rows if row["uid"] == uid]
        per_identity[uid] = {
            "subject_class": subset[0]["subject_class"],
            "n": len(subset),
            "base_ccip": mean(subset, "base_ccip"),
            "caption_base_ccip": mean(subset, "caption_base_ccip"),
            "distilled_ccip": mean(subset, "distilled_ccip"),
            "teacher_ccip": mean(subset, "teacher_ccip"),
            "gain_vs_caption": mean(subset, "gain_vs_caption"),
            "specificity_margin": mean(subset, "specificity_margin"),
            "distilled_passes": sum(row["distilled_pass"] for row in subset),
            "distilled_retrieval_top1": sum(
                row["distilled_retrieval_top1"] for row in subset
            ),
        }

    return {
        "protocol": protocol["name"],
        "identity_count": protocol["identity_count"],
        "scene_count": protocol["scene_count"],
        "seed_count": len(protocol["seeds"]),
        "n": len(rows),
        "threshold": THRESHOLD,
        "means": {
            f"{kind}_{metric}": mean(rows, f"{kind}_{metric}")
            for kind in KINDS
            for metric in ("ccip", "scene_clip", "reference_clip")
        },
        "head_means": {
            f"{kind}_head_ccip": mean(rows, f"{kind}_head_ccip")
            for kind in HEAD_KINDS
        },
        "aesthetic_means": {
            kind: mean(rows, f"{kind}_aesthetic_percentile")
            for kind in KINDS
        },
        "aesthetic_labels": {
            kind: {
                label: sum(
                    row[f"{kind}_aesthetic_label"] == label for row in rows
                )
                for label in ("worst", "low", "normal", "high", "masterpiece")
            }
            for kind in KINDS
        },
        "retrieval": {
            kind: {
                "top1": sum(row[f"{kind}_retrieval_top1"] for row in rows),
                "same_class_top1": sum(
                    row[f"{kind}_same_class_top1"] for row in rows
                ),
                "mean_rank": mean(rows, f"{kind}_retrieval_rank"),
                "mean_same_class_rank": mean(
                    rows, f"{kind}_same_class_rank"
                ),
                "mean_margin": mean(rows, f"{kind}_retrieval_margin"),
            }
            for kind in KINDS
        },
        "distilled_passes": sum(row["distilled_pass"] for row in rows),
        "caption_passes": sum(row["caption_pass"] for row in rows),
        "specific_count": sum(row["specificity_margin"] > 0 for row in rows),
        "negative_switch_count": sum(row["negative_switch"] > 0 for row in rows),
        "head_detection": {
            kind: sum(row[f"{kind}_head_count"] > 0 for row in rows)
            for kind in HEAD_KINDS
        },
        "single_head": {
            kind: sum(row[f"{kind}_head_count"] == 1 for row in rows)
            for kind in HEAD_KINDS
        },
        "multiple_heads": {
            kind: sum(row[f"{kind}_head_count"] > 1 for row in rows)
            for kind in HEAD_KINDS
        },
        "mean_pixel_l1_reference": {
            kind: mean(rows, f"{kind}_pixel_l1_reference")
            for kind in KINDS
        },
        "comparisons": comparisons,
        "per_scene": per_scene,
        "per_identity": per_identity,
        "locked_method": protocol["method"],
        "reference_policy": protocol["reference_policy"],
        "teacher_identity_text": protocol["teacher_identity_text"],
    }


def rel(path: str) -> str:
    return os.path.relpath(path, REPORT_PATH.parent)


def figure(path: str, label: str, metric: float | None = None) -> str:
    detail = "" if metric is None else f"<span>CCIP {metric:.3f}</span>"
    return f"""
      <figure>
        <div class="image-frame">
          <img src="{html.escape(rel(path))}" alt="{html.escape(label)}" loading="lazy">
        </div>
        <figcaption><strong>{html.escape(label)}</strong>{detail}</figcaption>
      </figure>"""


def build_html(
    rows: list[dict],
    summary: dict,
    protocol: dict,
    vlm_summary: dict | None = None,
) -> str:
    rows_by_uid: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        rows_by_uid[row["uid"]].append(row)

    identity_sections = []
    for uid in protocol["evaluation_uids"]:
        identity = summary["per_identity"][uid]
        observations = []
        for row in sorted(rows_by_uid[uid], key=lambda item: (item["scene"], item["seed"])):
            observations.append(
                f"""
          <article class="observation"
            data-scene="{row['scene']}"
            data-seed="{row['seed']}"
            data-fail="{'true' if not row['distilled_pass'] else 'false'}"
            data-low-quality="{'true' if row['distilled_aesthetic_label'] in ('worst', 'low') else 'false'}">
            <header>
              <h3>{html.escape(row['scene'].title())}</h3>
              <p>Seed {row['seed']} | caption gain {row['gain_vs_caption']:+.3f}</p>
            </header>
            <div class="image-grid">
              {figure(row['reference'], 'Reference')}
              {figure(row['base'], 'Class-only input', row['base_ccip'])}
              {figure(row['caption_base'], 'Attribute-only', row['caption_base_ccip'])}
              {figure(row['distilled'], 'Distilled output', row['distilled_ccip'])}
              {figure(row['teacher'], 'Edit teacher', row['teacher_ccip'])}
            </div>
            <details>
              <summary>Controls and diagnostics</summary>
              <div class="control-grid">
                {figure(row['negative_reference'], 'Wrong reference')}
                {figure(row['negative_teacher'], 'Wrong-ID teacher', row['negative_teacher_wrong_ccip'])}
                {figure(row['matched_control'], 'Matched reconstruction', row['matched_control_ccip'])}
              </div>
              <p class="diagnostic">
                Correct-reference margin {row['specificity_margin']:+.3f}.
                Retrieval rank {row['distilled_retrieval_rank']}/30,
                margin {row['distilled_retrieval_margin']:+.3f}.
                Scene CLIP: base {row['base_scene_clip']:.3f},
                caption {row['caption_base_scene_clip']:.3f},
                distilled {row['distilled_scene_clip']:.3f},
                teacher {row['teacher_scene_clip']:.3f}.
                Reference CLIP: caption {row['caption_base_reference_clip']:.3f},
                distilled {row['distilled_reference_clip']:.3f}.
                Aesthetic percentile: caption {row['caption_base_aesthetic_percentile']:.3f},
                distilled {row['distilled_aesthetic_percentile']:.3f}.
              </p>
            </details>
          </article>"""
            )
        identity_sections.append(
            f"""
      <section class="identity"
        data-uid="{uid}"
        data-class="{identity['subject_class']}"
        data-fail="{'true' if identity['distilled_passes'] < identity['n'] else 'false'}">
        <div class="identity-head">
          <div>
            <h2>u{uid}</h2>
            <p>{identity['subject_class']} | {identity['distilled_passes']}/{identity['n']} strict passes</p>
          </div>
          <dl>
            <div><dt>Distilled</dt><dd>{identity['distilled_ccip']:.3f}</dd></div>
            <div><dt>Caption gain</dt><dd>{identity['gain_vs_caption']:+.3f}</dd></div>
            <div><dt>ID margin</dt><dd>{identity['specificity_margin']:+.3f}</dd></div>
          </dl>
        </div>
        {''.join(observations)}
      </section>"""
        )

    scene_rows = "".join(
        f"""
          <div class="scene-cell">
            <h3>{scene.title()}</h3>
            <dl>
              <div><dt>Base</dt><dd>{values['base_ccip']:.3f}</dd></div>
              <div><dt>Caption</dt><dd>{values['caption_base_ccip']:.3f}</dd></div>
              <div><dt>Distilled</dt><dd>{values['distilled_ccip']:.3f}</dd></div>
              <div><dt>Teacher</dt><dd>{values['teacher_ccip']:.3f}</dd></div>
            </dl>
          </div>"""
        for scene, values in summary["per_scene"].items()
    )
    means = summary["means"]
    gain = summary["comparisons"]["gain_vs_caption"]
    ci = gain["subject_clustered_ci95"]
    if vlm_summary:
        vlm_cells = "".join(
            f"""
          <div class="scene-cell">
            <h3>{label.replace('_', ' ').title()}</h3>
            <dl>
              <div><dt>Identity</dt><dd>{values['identity_score']:.3f}</dd></div>
              <div><dt>Defect rate</dt><dd>{values['defect_rate']:.1%}</dd></div>
              <div><dt>Scene / 2</dt><dd>{values['scene_fidelity']:.3f}</dd></div>
              <div><dt>Clean ID</dt><dd>{values['clean_identity_rate']:.1%}</dd></div>
            </dl>
          </div>"""
            for label, values in vlm_summary["majority"].items()
        )
        vlm_html = f"""
  <section class="scenes">
    <h2>Anonymous three-pass VLM audit</h2>
    <p>Gemini 3 Flash evaluated randomized candidates. This is an auxiliary
    model judgment, not human evaluation. Identity Fleiss kappa:
    {vlm_summary['agreement']['identity_fleiss_kappa']:.3f}. Defect Fleiss
    kappa: {vlm_summary['agreement']['defect_fleiss_kappa']:.3f}.</p>
    <div class="scene-grid">{vlm_cells}</div>
  </section>"""
    else:
        vlm_html = ""
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="Complete held-out audit of test-time identity score distillation">
  <title>Krea2 Identity Distillation | Eval30 Audit</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #eef1f2;
      --surface: #f8faf9;
      --surface-2: #e3e8e9;
      --text: #182022;
      --muted: #596568;
      --line: #c6ced0;
      --accent: #087b78;
      --accent-soft: #d3e9e7;
      --radius: 12px;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111719;
        --surface: #192124;
        --surface-2: #222d30;
        --text: #edf2f2;
        --muted: #aab5b7;
        --line: #344145;
        --accent: #70c8c3;
        --accent-soft: #173b3c;
      }}
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Helvetica Neue", "Noto Sans", "Noto Sans SC", Arial, sans-serif;
      line-height: 1.5;
    }}
    main {{ width: min(1680px, calc(100% - 40px)); margin: 0 auto; padding-bottom: 96px; }}
    .hero {{
      min-height: min(720px, 100dvh);
      display: grid;
      grid-template-columns: minmax(0, 1.3fr) minmax(380px, .7fr);
      align-items: end;
      gap: 64px;
      padding: 64px 0 48px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font: 700 12px/1.2 ui-monospace, monospace;
      letter-spacing: .12em;
      text-transform: uppercase;
    }}
    h1 {{
      max-width: 780px;
      margin: 18px 0 18px;
      font-size: clamp(48px, 6vw, 92px);
      line-height: .96;
      letter-spacing: -.055em;
    }}
    .hero-copy {{ max-width: 58ch; margin: 0; color: var(--muted); font-size: 18px; }}
    .hero-metrics {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 1px;
      margin: 0;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: var(--line);
    }}
    .hero-metrics div {{ min-height: 118px; padding: 20px; background: var(--surface); }}
    dt {{ color: var(--muted); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }}
    dd {{ margin: 7px 0 0; font-size: 31px; font-weight: 680; letter-spacing: -.04em; }}
    .method {{ max-width: 920px; padding: 64px 0; }}
    .method h2, .scenes h2 {{ margin: 0 0 16px; font-size: clamp(34px, 4vw, 56px); letter-spacing: -.04em; }}
    .method p {{ color: var(--muted); font-size: 17px; }}
    code {{ padding: 3px 7px; border-radius: 6px; background: var(--accent-soft); color: var(--accent); }}
    .scenes {{ padding: 0 0 64px; }}
    .scene-grid {{ display: grid; grid-template-columns: 1.15fr .9fr .95fr; gap: 12px; }}
    .scene-cell {{ padding: 24px; border-radius: var(--radius); background: var(--surface); }}
    .scene-cell h3 {{ margin: 0 0 20px; font-size: 27px; }}
    .scene-cell dl {{ display: grid; grid-template-columns: 1fr 1fr; gap: 18px; margin: 0; }}
    .scene-cell dd {{ font-size: 24px; }}
    .filters {{
      position: sticky;
      top: 0;
      z-index: 10;
      display: grid;
      grid-template-columns: 1fr repeat(3, minmax(150px, auto));
      gap: 10px;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: var(--radius);
      background: color-mix(in srgb, var(--bg) 92%, transparent);
      backdrop-filter: blur(18px);
    }}
    input, select {{
      min-width: 0;
      min-height: 44px;
      padding: 0 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--surface);
      color: var(--text);
      font: inherit;
    }}
    input:focus, select:focus {{ outline: 2px solid var(--accent); outline-offset: 2px; }}
    .identity {{ padding: 72px 0 18px; border-bottom: 1px solid var(--line); }}
    .identity[hidden], .observation[hidden] {{ display: none; }}
    .identity-head {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 36px;
      align-items: end;
      margin-bottom: 32px;
    }}
    .identity-head h2 {{ margin: 0; font-size: clamp(38px, 5vw, 68px); letter-spacing: -.05em; }}
    .identity-head p {{ margin: 5px 0 0; color: var(--muted); }}
    .identity-head dl {{ display: grid; grid-template-columns: repeat(3, minmax(110px, 1fr)); gap: 28px; margin: 0; }}
    .identity-head dd {{ font-size: 25px; }}
    .observation {{ margin-bottom: 56px; }}
    .observation > header {{
      display: flex;
      align-items: baseline;
      justify-content: space-between;
      gap: 24px;
      margin-bottom: 12px;
    }}
    .observation h3 {{ margin: 0; font-size: 20px; }}
    .observation header p {{ margin: 0; color: var(--muted); font-family: ui-monospace, monospace; }}
    .image-grid {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 10px; }}
    figure {{ min-width: 0; margin: 0; }}
    .image-frame {{
      aspect-ratio: 1;
      overflow: hidden;
      border-radius: var(--radius);
      background: var(--surface-2);
    }}
    img {{ display: block; width: 100%; height: 100%; object-fit: contain; }}
    figcaption {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      padding: 9px 2px 0;
      font-size: 12px;
    }}
    figcaption span {{ color: var(--muted); font-family: ui-monospace, monospace; }}
    details {{ margin-top: 14px; padding: 14px; border-radius: var(--radius); background: var(--surface); }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 650; }}
    .control-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0, 260px)); gap: 12px; margin-top: 16px; }}
    .diagnostic {{ color: var(--muted); font-size: 13px; }}
    .empty {{ display: none; padding: 64px 0; color: var(--muted); font-size: 22px; }}
    @media (max-width: 1000px) {{
      main {{ width: min(100% - 24px, 1680px); }}
      .hero {{ min-height: auto; grid-template-columns: 1fr; padding-top: 48px; }}
      .scene-grid {{ grid-template-columns: 1fr; }}
      .filters {{ grid-template-columns: 1fr 1fr; }}
      .identity-head {{ grid-template-columns: 1fr; align-items: start; }}
      .image-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 20px 10px; }}
    }}
    @media (max-width: 600px) {{
      h1 {{ font-size: 48px; }}
      .hero-metrics, .filters, .identity-head dl {{ grid-template-columns: 1fr; }}
      .image-grid, .control-grid {{ grid-template-columns: 1fr; }}
      .observation > header {{ align-items: flex-start; flex-direction: column; gap: 3px; }}
    }}
    @media (prefers-reduced-transparency: reduce) {{
      .filters {{ background: var(--bg); backdrop-filter: none; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <div class="eyebrow">30 identities | 3 scenes | 2 seeds</div>
      <h1>Identity distillation, audited.</h1>
      <p class="hero-copy">Reference, class-only input, attribute baseline, distilled output, and edit teacher. Every held-out sample is shown.</p>
    </div>
    <dl class="hero-metrics">
      <div><dt>Distilled CCIP</dt><dd>{means['distilled_ccip']:.3f}</dd></div>
      <div><dt>Caption gain</dt><dd>{gain['mean']:+.3f}</dd></div>
      <div><dt>Strict passes</dt><dd>{summary['distilled_passes']}/{summary['n']}</dd></div>
      <div><dt>ID retrieval top-1</dt><dd>{summary['retrieval']['distilled']['top1']}/{summary['n']}</dd></div>
    </dl>
  </section>

  <section class="method">
    <h2>Edit teacher to generation trajectory</h2>
    <p>At each active state, a text-only automatic-attribute base and reference-conditioned edit teacher are evaluated at the same latent and timestep. <code>Delta s = -(1-sigma)/sigma (vT-vText)</code> defines a detached clean-image target. Three inner updates are projected into adaptive global and per-token trust regions. The teacher prediction never enters the distilled scheduler.</p>
    <p>Against the attribute-only baseline, mean CCIP gain is {gain['mean']:+.4f}, with a subject-clustered 95% interval [{ci[0]:+.4f}, {ci[1]:+.4f}] and two-sided subject sign-test p={gain['subject_sign_p_two_sided']:.4g}.</p>
  </section>

  <section class="scenes">
    <h2>Performance by scene</h2>
    <div class="scene-grid">{scene_rows}</div>
  </section>
  {vlm_html}

  <form class="filters" onsubmit="return false">
    <input id="identity-filter" type="search" aria-label="Filter identity" placeholder="Filter identity, for example 5319">
    <select id="scene-filter" aria-label="Filter scene">
      <option value="all">All scenes</option>
      <option value="portrait">Portrait</option>
      <option value="rain">Rain street</option>
      <option value="kitchen">Kitchen</option>
    </select>
    <select id="seed-filter" aria-label="Filter seed">
      <option value="all">Both seeds</option>
      <option value="2027">Seed 2027</option>
      <option value="2028">Seed 2028</option>
    </select>
    <select id="result-filter" aria-label="Filter result">
      <option value="all">All results</option>
      <option value="fail">Strict failures</option>
      <option value="quality">Low aesthetic</option>
      <option value="pass">Strict passes</option>
    </select>
  </form>
  <p id="empty" class="empty">No observations match the current filters.</p>

  <div id="identity-list">{''.join(identity_sections)}</div>
</main>
<script>
  const identityInput = document.querySelector("#identity-filter");
  const sceneInput = document.querySelector("#scene-filter");
  const seedInput = document.querySelector("#seed-filter");
  const resultInput = document.querySelector("#result-filter");
  const sections = [...document.querySelectorAll(".identity")];
  const empty = document.querySelector("#empty");

  function applyFilters() {{
    const identityQuery = identityInput.value.trim().toLowerCase();
    const scene = sceneInput.value;
    const seed = seedInput.value;
    const result = resultInput.value;
    let visibleCount = 0;
    for (const section of sections) {{
      const identityMatch = section.dataset.uid.includes(identityQuery);
      let sectionVisible = false;
      for (const observation of section.querySelectorAll(".observation")) {{
        const sceneMatch = scene === "all" || observation.dataset.scene === scene;
        const seedMatch = seed === "all" || observation.dataset.seed === seed;
        const resultMatch =
          result === "all" ||
          (result === "fail" && observation.dataset.fail === "true") ||
          (result === "quality" && observation.dataset.lowQuality === "true") ||
          (result === "pass" && observation.dataset.fail === "false");
        const visible = identityMatch && sceneMatch && seedMatch && resultMatch;
        observation.hidden = !visible;
        sectionVisible ||= visible;
        visibleCount += visible ? 1 : 0;
      }}
      section.hidden = !sectionVisible;
    }}
    empty.style.display = visibleCount ? "none" : "block";
  }}
  for (const control of [identityInput, sceneInput, seedInput, resultInput]) {{
    control.addEventListener("input", applyFilters);
    control.addEventListener("change", applyFilters);
  }}
</script>
</body>
</html>
"""


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    rows = collect_rows(protocol)
    add_identity_retrieval_metrics(rows, protocol)
    add_clip_metrics(rows)
    summary = summarize(rows, protocol)
    vlm_summary = (
        json.loads(VLM_SUMMARY_PATH.read_text(encoding="utf-8"))
        if VLM_SUMMARY_PATH.is_file()
        else None
    )
    METRICS_PATH.write_text(
        json.dumps(
            {
                "summary": summary,
                "vlm_summary": vlm_summary,
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    with CSV_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    REPORT_PATH.write_text(
        build_html(rows, summary, protocol, vlm_summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"metrics: {METRICS_PATH}")
    print(f"csv: {CSV_PATH}")
    print(f"report: {REPORT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

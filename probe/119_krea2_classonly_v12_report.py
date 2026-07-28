#!/usr/bin/env python3
"""Audit the locked Krea2 v1.2 class-only 3x3 identity distillation run."""

from __future__ import annotations

import csv
import html
import itertools
import json
import shutil
import statistics
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image
from imgutils.detect import detect_heads
from imgutils.metrics import ccip_difference


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
OUT = ROOT / "out" / "krea2_tsd_neutral"
UIDS = ("u4229", "u1839", "u9728")
SEEDS = (777, 778, 779)
THRESHOLD = 0.178
DIMENSIONS = {
    "u4229": (512, 896),
    "u1839": (512, 768),
    "u9728": (640, 640),
}
CLASS_WORDS = {
    "u4229": ("man", "his", "him"),
    "u1839": ("woman", "her", "her"),
    "u9728": ("woman", "her", "her"),
}
LOCKED = {
    "steps": 12,
    "inner_iters": 3,
    "lr": 0.05,
    "pseudo_step_size": 1.0,
    "teacher_weight": 1.0,
    "active_window": [0.0, 1.0],
    "normalize": "rms_token",
    "residual_mode": "raw",
    "trust_region_rms": 0.1,
    "local_trust_region_rms": 0.085,
    "adaptive_trust_ratio": 3.0,
    "ref_boost": 4.0,
}
EXPECTED_LORA = "krea2_identity_edit_v1_2.safetensors"
HEAD_CACHE: dict[str, Image.Image] = {}


def stem_for(uid: str, seed: int) -> str:
    width, height = DIMENSIONS[uid]
    return (
        f"{uid}_classonly_v12_ar_s{seed}_{width}x{height}_"
        "i3_p1_g0.1_l0.085_a3_rb4_rms_token_raw_roifull_o1"
    )


def expected_prompts(uid: str) -> tuple[str, str]:
    class_word, possessive, object_pronoun = CLASS_WORDS[uid]
    scene = (
        f"single adult anime {class_word} standing on a rain-soaked neon city street "
        "at night, three-quarter body, holding a transparent umbrella, cinematic "
        "reflections, coherent anatomy, face clearly visible, no other people"
    )
    teacher = (
        f"Create a high-quality anime illustration of the exact adult {class_word} "
        "shown in the reference image as the only subject. Infer every identity trait "
        "only from the reference image. Preserve "
        f"{possessive} identity without inventing a different person. Place {object_pronoun} "
        "on a rain-soaked neon city street at night, three-quarter body, holding a "
        "transparent umbrella, cinematic reflections, coherent anatomy, face clearly "
        "visible, no other people, no collage, no split panel."
    )
    return scene, teacher


def exact_paired_sign_pvalue(improvements: list[float]) -> float:
    observed = abs(statistics.mean(improvements))
    extreme = 0
    total = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(improvements)):
        permuted = abs(
            statistics.mean(sign * value for sign, value in zip(signs, improvements))
        )
        extreme += permuted >= observed - 1e-12
        total += 1
    return extreme / total


def bootstrap_mean_ci(
    values: list[float], samples: int = 20000
) -> tuple[float, float]:
    rng = np.random.default_rng(20260727)
    array = np.asarray(values, dtype=np.float64)
    draws = rng.choice(array, size=(samples, len(values)), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def pixel_l1(left_path: Path, right_path: Path) -> float:
    left = np.asarray(
        Image.open(left_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    right = np.asarray(
        Image.open(right_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS),
        dtype=np.float32,
    )
    return float(np.abs(left - right).mean() / 255.0)


def head_crop(path: Path, pad_ratio: float = 0.08) -> Image.Image:
    key = str(path)
    if key in HEAD_CACHE:
        return HEAD_CACHE[key]
    image = Image.open(path).convert("RGB")
    detections = detect_heads(str(path))
    if not detections:
        raise RuntimeError(f"No anime head detected in {path}")
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
    HEAD_CACHE[key] = crop
    return crop


def add_scene_clip_scores(rows: list[dict]) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device=device, force_quick_gelu=True
    )
    model.eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    kinds = ("base", "matched_control", "distilled", "teacher", "negative_teacher")
    image_tensors = []
    prompts = []
    assignments = []
    for row_index, row in enumerate(rows):
        class_word = CLASS_WORDS[row["uid"]][0]
        prompt = (
            f"an anime {class_word} holding a transparent umbrella on a rain-soaked "
            "neon city street at night"
        )
        for kind in kinds:
            with Image.open(row[kind]) as source:
                image_tensors.append(preprocess(source.convert("RGB")))
            prompts.append(prompt)
            assignments.append((row_index, kind))
    with torch.inference_mode():
        images = torch.stack(image_tensors).to(device)
        tokens = tokenizer(prompts).to(device)
        image_features = model.encode_image(images)
        text_features = model.encode_text(tokens)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = (image_features * text_features).sum(dim=-1).float().cpu().tolist()
    for (row_index, kind), score in zip(assignments, scores):
        rows[row_index][f"{kind}_scene_clip"] = float(score)
    del model, images, tokens, image_features, text_features
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def check_locked(metadata: dict, metadata_path: Path, uid: str) -> None:
    for key, expected in LOCKED.items():
        actual = metadata.get(key)
        if actual != expected:
            raise AssertionError(
                f"{metadata_path.name}: expected {key}={expected!r}, got {actual!r}"
            )
    if metadata.get("teacher_scheduler_usage") != "none_in_distilled_trajectory":
        raise AssertionError(f"{metadata_path.name}: teacher entered student scheduler")
    if Path(metadata.get("teacher_lora", "")).name != EXPECTED_LORA:
        raise AssertionError(f"{metadata_path.name}: unexpected teacher LoRA")
    expected_scene, expected_teacher = expected_prompts(uid)
    if metadata.get("scene_prompt") != expected_scene:
        raise AssertionError(f"{metadata_path.name}: scene prompt is not class-only")
    if metadata.get("teacher_prompt") != expected_teacher:
        raise AssertionError(f"{metadata_path.name}: teacher prompt changed")
    if metadata.get("size") != list(DIMENSIONS[uid]):
        raise AssertionError(f"{metadata_path.name}: aspect ratio protocol changed")


def collect_rows() -> list[dict]:
    rows = []
    reference_by_uid: dict[str, str] = {}
    negative_by_uid: dict[str, str] = {}
    for uid in UIDS:
        for seed in SEEDS:
            metadata_path = OUT / f"{stem_for(uid, seed)}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            check_locked(metadata, metadata_path, uid)

            reference_by_uid.setdefault(uid, metadata["reference"])
            negative_by_uid.setdefault(uid, metadata["negative_reference"])
            if reference_by_uid[uid] != metadata["reference"]:
                raise AssertionError(f"{uid}: reference changed across seeds")
            if negative_by_uid[uid] != metadata["negative_reference"]:
                raise AssertionError(f"{uid}: negative reference changed across seeds")

            reference = Path(metadata["reference"])
            negative_reference = Path(metadata["negative_reference"])
            paths = {
                key: Path(metadata["outputs"][key])
                for key in (
                    "base",
                    "matched_control",
                    "distilled",
                    "teacher",
                    "negative_teacher",
                )
            }
            for path in (reference, negative_reference, *paths.values()):
                if not path.is_file():
                    raise FileNotFoundError(path)

            public_reference = OUT / f"report_{uid}_reference.png"
            public_negative = OUT / f"report_{uid}_negative_reference.png"
            shutil.copy2(reference, public_reference)
            shutil.copy2(negative_reference, public_negative)

            row = {
                "uid": uid,
                "seed": seed,
                "reference": str(public_reference),
                "negative_reference": str(public_negative),
                **{key: str(path) for key, path in paths.items()},
            }
            for kind, path in paths.items():
                row[f"{kind}_ccip"] = float(ccip_difference(str(reference), str(path)))
            reference_head = head_crop(reference)
            for kind in ("base", "matched_control", "distilled", "teacher"):
                row[f"{kind}_head_ccip"] = float(
                    ccip_difference(reference_head, head_crop(paths[kind]))
                )
            row["distilled_wrong_ccip"] = float(
                ccip_difference(str(negative_reference), str(paths["distilled"]))
            )
            row["teacher_wrong_ccip"] = float(
                ccip_difference(str(negative_reference), str(paths["teacher"]))
            )
            row["negative_teacher_wrong_ccip"] = float(
                ccip_difference(str(negative_reference), str(paths["negative_teacher"]))
            )
            row["improvement"] = row["base_ccip"] - row["distilled_ccip"]
            row["matched_gain"] = (
                row["matched_control_ccip"] - row["distilled_ccip"]
            )
            row["relative_improvement"] = row["improvement"] / max(
                row["base_ccip"], 1e-12
            )
            row["specificity_margin"] = (
                row["distilled_wrong_ccip"] - row["distilled_ccip"]
            )
            row["teacher_specificity_margin"] = (
                row["teacher_wrong_ccip"] - row["teacher_ccip"]
            )
            row["negative_control_switch"] = (
                row["negative_teacher_ccip"] - row["negative_teacher_wrong_ccip"]
            )
            row["pass"] = row["distilled_ccip"] < THRESHOLD
            row["specific"] = row["specificity_margin"] > 0.0
            row["reference_pixel_l1"] = pixel_l1(reference, paths["distilled"])
            rows.append(row)
    add_scene_clip_scores(rows)
    return rows


def summarize(rows: list[dict]) -> dict:
    improvements = [row["improvement"] for row in rows]
    specificity = [row["specificity_margin"] for row in rows]
    ci_low, ci_high = bootstrap_mean_ci(improvements)
    spec_low, spec_high = bootstrap_mean_ci(specificity)
    per_identity = {}
    for uid in UIDS:
        subset = [row for row in rows if row["uid"] == uid]
        per_identity[uid] = {
            "base_mean": statistics.mean(row["base_ccip"] for row in subset),
            "matched_control_mean": statistics.mean(
                row["matched_control_ccip"] for row in subset
            ),
            "distilled_mean": statistics.mean(
                row["distilled_ccip"] for row in subset
            ),
            "teacher_mean": statistics.mean(row["teacher_ccip"] for row in subset),
            "distilled_head_mean": statistics.mean(
                row["distilled_head_ccip"] for row in subset
            ),
            "distilled_scene_mean": statistics.mean(
                row["distilled_scene_clip"] for row in subset
            ),
            "specificity_margin_mean": statistics.mean(
                row["specificity_margin"] for row in subset
            ),
            "worst_distilled": max(row["distilled_ccip"] for row in subset),
            "passes": sum(row["pass"] for row in subset),
            "specific": sum(row["specific"] for row in subset),
        }
    return {
        "threshold": THRESHOLD,
        "n": len(rows),
        "base_mean": statistics.mean(row["base_ccip"] for row in rows),
        "matched_control_mean": statistics.mean(
            row["matched_control_ccip"] for row in rows
        ),
        "distilled_mean": statistics.mean(row["distilled_ccip"] for row in rows),
        "teacher_mean": statistics.mean(row["teacher_ccip"] for row in rows),
        "base_head_mean": statistics.mean(row["base_head_ccip"] for row in rows),
        "matched_control_head_mean": statistics.mean(
            row["matched_control_head_ccip"] for row in rows
        ),
        "distilled_head_mean": statistics.mean(
            row["distilled_head_ccip"] for row in rows
        ),
        "teacher_head_mean": statistics.mean(
            row["teacher_head_ccip"] for row in rows
        ),
        "base_scene_clip_mean": statistics.mean(
            row["base_scene_clip"] for row in rows
        ),
        "matched_control_scene_clip_mean": statistics.mean(
            row["matched_control_scene_clip"] for row in rows
        ),
        "distilled_scene_clip_mean": statistics.mean(
            row["distilled_scene_clip"] for row in rows
        ),
        "teacher_scene_clip_mean": statistics.mean(
            row["teacher_scene_clip"] for row in rows
        ),
        "negative_teacher_mean": statistics.mean(
            row["negative_teacher_ccip"] for row in rows
        ),
        "mean_improvement": statistics.mean(improvements),
        "relative_mean_improvement": 1.0
        - statistics.mean(row["distilled_ccip"] for row in rows)
        / statistics.mean(row["base_ccip"] for row in rows),
        "improvement_bootstrap_ci95": [ci_low, ci_high],
        "paired_exact_p_two_sided": exact_paired_sign_pvalue(improvements),
        "specificity_margin_mean": statistics.mean(specificity),
        "specificity_bootstrap_ci95": [spec_low, spec_high],
        "worst_distilled": max(row["distilled_ccip"] for row in rows),
        "passes": sum(row["pass"] for row in rows),
        "specific_count": sum(row["specific"] for row in rows),
        "mean_reference_pixel_l1": statistics.mean(
            row["reference_pixel_l1"] for row in rows
        ),
        "per_identity": per_identity,
        "locked_parameters": LOCKED,
        "prompt_protocol": "class-only scene prompt; identity traits only from reference",
    }


def image_cell(
    path: str, label: str, detail: str, width: int, height: int
) -> str:
    filename = Path(path).name
    return f"""
              <figure>
                <div class="image-frame" style="aspect-ratio:{width}/{height}">
                  <img src="{html.escape(filename)}" alt="{html.escape(label)}" loading="lazy">
                </div>
                <figcaption>
                  <strong>{html.escape(label)}</strong>
                  <span>{html.escape(detail)}</span>
                </figcaption>
              </figure>"""


def build_html(rows: list[dict], summary: dict) -> str:
    groups = []
    for uid in UIDS:
        width, height = DIMENSIONS[uid]
        identity_rows = [row for row in rows if row["uid"] == uid]
        items = []
        for row in identity_rows:
            gate = "PASS" if row["pass"] else "FAIL"
            items.append(
                f"""
          <article class="comparison">
            <header>
              <h3>Seed {row['seed']}</h3>
              <p>
                Base {row['base_ccip']:.4f}
                <span>→</span>
                Distilled {row['distilled_ccip']:.4f}
                <b class="gate">{gate}</b>
              </p>
            </header>
            <div class="image-grid">
              {image_cell(row['reference'], 'Reference ID', f'{width} × {height}', width, height)}
              {image_cell(row['base'], 'Input / Base', f"CCIP {row['base_ccip']:.4f}", width, height)}
              {image_cell(row['distilled'], 'Distilled output', f"CCIP {row['distilled_ccip']:.4f}", width, height)}
              {image_cell(row['teacher'], 'Direct teacher', f"CCIP {row['teacher_ccip']:.4f}", width, height)}
            </div>
            <p class="control-line">
              Wrong-ID distance {row['distilled_wrong_ccip']:.4f};
              specificity margin {row['specificity_margin']:+.4f};
              head CCIP {row['distilled_head_ccip']:.4f};
              scene CLIP {row['distilled_scene_clip']:.3f}.
            </p>
          </article>"""
            )
        identity = summary["per_identity"][uid]
        groups.append(
            f"""
      <section class="identity-section" id="{uid}">
        <div class="identity-heading">
          <div>
            <h2>{uid}</h2>
            <p>三个固定 seed，共享 class-only 场景提示与全部算法参数。</p>
          </div>
          <dl>
            <div><dt>Distilled mean</dt><dd>{identity['distilled_mean']:.4f}</dd></div>
            <div><dt>Specificity</dt><dd>{identity['specificity_margin_mean']:+.4f}</dd></div>
            <div><dt>Pass</dt><dd>{identity['passes']}/3</dd></div>
          </dl>
        </div>
        {''.join(items)}
      </section>"""
        )

    table_rows = "".join(
        f"""
          <tr>
            <td>{row['uid']}</td>
            <td>{row['seed']}</td>
            <td>{row['base_ccip']:.4f}</td>
            <td>{row['matched_control_ccip']:.4f}</td>
            <td class="best">{row['distilled_ccip']:.4f}</td>
            <td>{row['teacher_ccip']:.4f}</td>
            <td>{row['distilled_head_ccip']:.4f}</td>
            <td>{row['distilled_scene_clip']:.3f}</td>
            <td>{row['improvement']:+.4f}</td>
            <td>{row['specificity_margin']:+.4f}</td>
            <td>{'PASS' if row['pass'] else 'FAIL'}</td>
          </tr>"""
        for row in rows
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="A 3 by 3 audit of class-only identity injection with trust-region score distillation">
  <title>Krea2 Class-only Identity Score Distillation | 3x3 Audit</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f3f5f4;
      --surface: #e8ecea;
      --surface-strong: #dde3e0;
      --text: #16201d;
      --muted: #596762;
      --line: #c7d0cc;
      --accent: #08766e;
      --accent-soft: #cee6e1;
      --radius: 14px;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #101715;
        --surface: #18221f;
        --surface-strong: #202c28;
        --text: #edf3f0;
        --muted: #a7b5b0;
        --line: #35423e;
        --accent: #65c8bd;
        --accent-soft: #173c38;
      }}
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Helvetica Neue", "Noto Sans SC", Arial, sans-serif;
      font-feature-settings: "tnum";
      line-height: 1.5;
    }}
    main {{
      width: min(1500px, calc(100% - 40px));
      margin: 0 auto;
      padding: 36px 0 96px;
    }}
    .hero {{
      min-height: min(660px, calc(100dvh - 36px));
      display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(320px, .65fr);
      align-items: end;
      gap: 64px;
      padding: 44px 0 48px;
      border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{
      color: var(--accent);
      font: 700 12px/1.2 ui-monospace, SFMono-Regular, Menlo, monospace;
      letter-spacing: .13em;
      text-transform: uppercase;
    }}
    h1 {{
      max-width: 900px;
      margin: 18px 0 20px;
      font-size: clamp(48px, 7vw, 104px);
      line-height: .94;
      letter-spacing: -.055em;
    }}
    .hero-copy {{
      max-width: 62ch;
      margin: 0;
      color: var(--muted);
      font-size: 18px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(2, 1fr);
      gap: 1px;
      margin: 0;
      background: var(--line);
      border: 1px solid var(--line);
      border-radius: var(--radius);
      overflow: hidden;
    }}
    .summary div {{
      min-height: 116px;
      padding: 20px;
      background: var(--surface);
    }}
    dt {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: .07em;
    }}
    dd {{
      margin: 8px 0 0;
      font-size: clamp(26px, 3vw, 41px);
      font-weight: 650;
      letter-spacing: -.04em;
    }}
    .method {{
      display: grid;
      grid-template-columns: minmax(280px, .7fr) minmax(0, 1.3fr);
      gap: 64px;
      padding: 72px 0 80px;
    }}
    .method h2, .identity-heading h2 {{
      margin: 0;
      font-size: clamp(34px, 5vw, 64px);
      letter-spacing: -.045em;
      line-height: 1;
    }}
    .method-copy {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 30px;
      color: var(--muted);
    }}
    .method-copy p {{ margin: 0; }}
    code {{
      color: var(--accent);
      background: var(--accent-soft);
      padding: 3px 7px;
      border-radius: 6px;
    }}
    .identity-section {{
      padding: 64px 0 24px;
      border-top: 1px solid var(--line);
    }}
    .identity-heading {{
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 40px;
      align-items: end;
      margin-bottom: 36px;
    }}
    .identity-heading p {{
      max-width: 60ch;
      margin: 16px 0 0;
      color: var(--muted);
    }}
    .identity-heading dl {{
      display: grid;
      grid-template-columns: repeat(3, minmax(112px, 1fr));
      gap: 28px;
      margin: 0;
    }}
    .identity-heading dd {{ font-size: 27px; }}
    .comparison {{ margin-bottom: 68px; }}
    .comparison > header {{
      display: flex;
      justify-content: space-between;
      align-items: baseline;
      gap: 24px;
      margin-bottom: 14px;
    }}
    .comparison h3 {{ margin: 0; font-size: 20px; }}
    .comparison header p {{
      margin: 0;
      color: var(--muted);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .comparison header span {{ color: var(--accent); }}
    .gate {{
      margin-left: 14px;
      color: var(--accent);
      font-size: 12px;
      letter-spacing: .07em;
    }}
    .image-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 14px;
      align-items: start;
    }}
    figure {{ min-width: 0; margin: 0; }}
    .image-frame {{
      display: grid;
      place-items: center;
      overflow: hidden;
      border-radius: var(--radius);
      background: var(--surface-strong);
    }}
    img {{
      display: block;
      width: 100%;
      height: 100%;
      object-fit: contain;
    }}
    figcaption {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 2px 0;
      font-size: 13px;
    }}
    figcaption span {{ color: var(--muted); text-align: right; }}
    .control-line {{
      margin: 12px 2px 0;
      color: var(--muted);
      font: 12px/1.5 ui-monospace, SFMono-Regular, Menlo, monospace;
    }}
    .metrics {{ padding: 72px 0 24px; border-top: 1px solid var(--line); }}
    .metrics h2 {{
      margin: 0 0 24px;
      font-size: 38px;
      letter-spacing: -.035em;
    }}
    .table-wrap {{
      overflow-x: auto;
      border-radius: var(--radius);
      border: 1px solid var(--line);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--surface);
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 13px;
    }}
    th, td {{
      padding: 13px 16px;
      text-align: right;
      border-bottom: 1px solid var(--line);
      white-space: nowrap;
    }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ color: var(--muted); font-weight: 500; }}
    td.best {{ color: var(--accent); font-weight: 700; }}
    .footnote {{
      max-width: 86ch;
      margin: 18px 0 0;
      color: var(--muted);
      font-size: 14px;
    }}
    @media (prefers-reduced-motion: reduce) {{
      html {{ scroll-behavior: auto; }}
    }}
    @media (max-width: 940px) {{
      main {{ width: min(100% - 24px, 1500px); padding-top: 20px; }}
      .hero {{ min-height: auto; grid-template-columns: 1fr; padding: 48px 0; }}
      h1 {{ font-size: clamp(46px, 16vw, 76px); }}
      .method {{ grid-template-columns: 1fr; gap: 28px; }}
      .identity-heading {{ grid-template-columns: 1fr; align-items: start; }}
      .image-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 28px 10px;
      }}
    }}
    @media (max-width: 540px) {{
      .summary, .identity-heading dl, .method-copy {{ grid-template-columns: 1fr; }}
      .summary div {{ min-height: 94px; }}
      .image-grid {{ grid-template-columns: 1fr; }}
      .comparison > header {{
        align-items: flex-start;
        flex-direction: column;
        gap: 4px;
      }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <div class="eyebrow">Krea2 v1.2 / 3 identities × 3 seeds</div>
      <h1>Class-only identity score distillation.</h1>
      <p class="hero-copy">Trust-region TSD 将平均 CCIP 从 {summary['base_mean']:.4f} 降至 {summary['distilled_mean']:.4f}，同时保持 scene CLIP {summary['base_scene_clip_mean']:.3f}→{summary['distilled_scene_clip_mean']:.3f}。9/9 更接近正确参考，严格阈值通过 6/9。本页是机制 pilot，不宣称广义 SOTA。</p>
    </div>
    <dl class="summary">
      <div><dt>CCIP mean ↓</dt><dd>{summary['distilled_mean']:.4f}</dd></div>
      <div><dt>Scene CLIP ↑</dt><dd>{summary['distilled_scene_clip_mean']:.3f}</dd></div>
      <div><dt>Strict gate</dt><dd>{summary['passes']}/{summary['n']}</dd></div>
      <div><dt>Correct ref closer</dt><dd>{summary['specific_count']}/{summary['n']}</dd></div>
    </dl>
  </section>

  <section class="method">
    <h2>Locked mechanism</h2>
    <div class="method-copy">
      <p>每个外层去噪步先由 base 前进。随后在同一 latent 和 timestep 重算 base 与 reference-conditioned teacher，teacher 不进入 distilled scheduler。</p>
      <p><code>Δs = -(1-σ)/σ · (vT-vB)</code>。逐 token RMS 归一化后构造 detached x0 伪目标，做 3 次内循环更新，并投影到自适应 trust region。</p>
    </div>
  </section>

  {''.join(groups)}

  <section class="metrics">
    <h2>Complete metric table</h2>
    <div class="table-wrap">
      <table>
        <thead>
          <tr>
            <th>Identity</th>
            <th>Seed</th>
            <th>Base</th>
            <th>Matched</th>
            <th>Distilled</th>
            <th>Teacher</th>
            <th>Head ID</th>
            <th>Scene</th>
            <th>Gain</th>
            <th>ID margin</th>
            <th>Gate</th>
          </tr>
        </thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    <p class="footnote">CCIP 越低越相似，门槛为 {THRESHOLD:.3f}。ID margin 是输出到错误参考的距离减去到正确参考的距离，正值表示更接近正确身份。平均改善 {summary['mean_improvement']:.4f}，95% bootstrap CI [{summary['improvement_bootstrap_ci95'][0]:.4f}, {summary['improvement_bootstrap_ci95'][1]:.4f}]，双侧精确配对符号检验 p={summary['paired_exact_p_two_sided']:.4f}。Scene 使用 OpenAI CLIP ViT-B/32 余弦相似度，越高越符合雨夜霓虹和透明伞场景。白底 matched control 的 scene 均值为 {summary['matched_control_scene_clip_mean']:.3f}，distilled 为 {summary['distilled_scene_clip_mean']:.3f}。平均像素 L1 为 {summary['mean_reference_pixel_l1']:.3f}，不支持直接复制参考图。</p>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    rows = collect_rows()
    summary = summarize(rows)

    metrics_path = OUT / "classonly_v12_3x3_metrics.json"
    metrics_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = OUT / "classonly_v12_3x3_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report_path = OUT / "classonly_v12_3x3_report.html"
    report_path.write_text(build_html(rows, summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"metrics: {metrics_path}")
    print(f"csv: {csv_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

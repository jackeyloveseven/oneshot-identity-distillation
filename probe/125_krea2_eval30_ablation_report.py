#!/usr/bin/env python3
"""Score and visualize the Krea2 v3 trust-region mechanism ablation."""

from __future__ import annotations

import html
import importlib.util
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np
import open_clip
import torch
from imgutils.metrics import (
    anime_dbaesthetic,
    ccip_batch_differences,
    ccip_batch_extract_features,
)
from PIL import Image


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
PROTOCOL_PATH = ROOT / "data" / "krea2_eval30_protocol.json"
MAIN_OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
OUT = ROOT / "out" / "krea2_tsd_eval30_ablations_v3"
METRICS = OUT / "ablation_metrics.json"
REPORT = OUT / "ablation_report.html"
CCIP_CACHE = OUT / "ablation_ccip_features.npz"
AESTHETIC_CACHE = OUT / "ablation_aesthetic_cache.json"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


MAIN_RUNNER = load_module("krea_main_runner", ROOT / "probe" / "121_run_krea2_eval30.py")
ABLATION_RUNNER = load_module(
    "krea_ablation_runner",
    ROOT / "probe" / "124_run_krea2_eval30_ablations.py",
)


def bootstrap_subject_mean(rows: list[dict], key: str) -> list[float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(float(row[key]))
    values = np.asarray(
        [statistics.mean(grouped[uid]) for uid in sorted(grouped)],
        dtype=np.float64,
    )
    rng = np.random.default_rng(20260727)
    draws = rng.choice(values, (20000, len(values)), replace=True).mean(axis=1)
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def collect(protocol: dict) -> list[dict]:
    rows = []
    missing = []
    jobs = [
        job
        for job in protocol["jobs"]
        if job["uid"] in ABLATION_RUNNER.ABLATION_UIDS
    ]
    for job in jobs:
        main_path = MAIN_RUNNER.expected_metadata(
            job, ABLATION_RUNNER.SEED, MAIN_OUT
        )
        if not main_path.is_file():
            missing.append(str(main_path))
            continue
        main = json.loads(main_path.read_text(encoding="utf-8"))
        reference = Path(job["reference"])
        negative_reference = Path(job["negative_reference"])
        caption = Path(main["outputs"]["caption_base"])
        full = Path(main["outputs"]["distilled"])
        row = {
            "uid": job["uid"],
            "subject_class": job["subject_class"],
            "scene": job["scene"],
            "reference": str(reference),
            "negative_reference": str(negative_reference),
            "caption": str(caption),
            "full": str(full),
        }
        for variant in ABLATION_RUNNER.VARIANTS:
            path = ABLATION_RUNNER.expected_metadata(job, variant, protocol)
            if not path.is_file():
                missing.append(str(path))
                continue
            metadata = json.loads(path.read_text(encoding="utf-8"))
            if set(metadata["outputs"]) != {"distilled"}:
                raise AssertionError(f"{path}: unexpected controls")
            if metadata.get("controls") != "none":
                raise AssertionError(f"{path}: controls flag is not none")
            output = Path(metadata["outputs"]["distilled"])
            row[variant] = str(output)
            row[f"{variant}_elapsed"] = float(metadata["elapsed_seconds"])
        rows.append(row)
    if missing:
        raise RuntimeError(
            f"ablation is incomplete: {len(missing)} files missing; "
            f"first is {missing[0]}"
        )
    return rows


def add_ccip_and_aesthetic_metrics(rows: list[dict]) -> None:
    kinds = ("caption", "full", *ABLATION_RUNNER.VARIANTS)
    image_paths = sorted(
        {
            row[key]
            for row in rows
            for key in ("reference", "negative_reference", *kinds)
        }
    )
    if CCIP_CACHE.is_file():
        cached = np.load(CCIP_CACHE)
        cached_paths = cached["paths"].tolist()
        features = cached["features"]
        if cached_paths != image_paths:
            raise RuntimeError("CCIP cache path set does not match this report")
        print(f"[ccip] loaded {len(image_paths)} cached features", flush=True)
    else:
        feature_batches = []
        for start in range(0, len(image_paths), 32):
            batch = image_paths[start : start + 32]
            feature_batches.append(ccip_batch_extract_features(batch))
            print(
                f"[ccip] extracted {min(start + 32, len(image_paths))}/"
                f"{len(image_paths)}",
                flush=True,
            )
        features = np.concatenate(feature_batches, axis=0)
        np.savez_compressed(
            CCIP_CACHE,
            paths=np.asarray(image_paths),
            features=features,
        )
    distances = ccip_batch_differences(list(features))
    index = {path: position for position, path in enumerate(image_paths)}
    for row in rows:
        ref_index = index[row["reference"]]
        wrong_index = index[row["negative_reference"]]
        for kind in kinds:
            output_index = index[row[kind]]
            row[f"{kind}_ccip"] = float(distances[ref_index, output_index])
            row[f"{kind}_wrong_ccip"] = float(
                distances[wrong_index, output_index]
            )

    aesthetic_paths = sorted({row[kind] for row in rows for kind in kinds})
    if AESTHETIC_CACHE.is_file():
        aesthetic_cache = json.loads(
            AESTHETIC_CACHE.read_text(encoding="utf-8")
        )
    else:
        aesthetic_cache = {}
    missing = [path for path in aesthetic_paths if path not in aesthetic_cache]
    for position, path in enumerate(missing, 1):
        label, percentile = anime_dbaesthetic(path)
        aesthetic_cache[path] = {
            "label": label,
            "percentile": float(percentile),
        }
        if position % 20 == 0 or position == len(missing):
            AESTHETIC_CACHE.write_text(
                json.dumps(aesthetic_cache, indent=2), encoding="utf-8"
            )
            print(
                f"[aesthetic] scored {position}/{len(missing)} new images",
                flush=True,
            )
    for row in rows:
        for kind in kinds:
            result = aesthetic_cache[row[kind]]
            row[f"{kind}_aesthetic_label"] = result["label"]
            row[f"{kind}_aesthetic_percentile"] = result["percentile"]


def add_clip_metrics(rows: list[dict], protocol: dict) -> None:
    prompt_for = {
        (job["uid"], job["scene"]): job["scene_prompt"]
        for job in protocol["jobs"]
    }
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32",
        pretrained="openai",
        device=device,
        force_quick_gelu=True,
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    kinds = ("caption", "full", *ABLATION_RUNNER.VARIANTS)
    entries = [(row, kind) for row in rows for kind in kinds]
    for start in range(0, len(entries), 64):
        batch = entries[start : start + 64]
        images = torch.stack(
            [preprocess(Image.open(row[kind]).convert("RGB")) for row, kind in batch]
        ).to(device)
        references = torch.stack(
            [preprocess(Image.open(row["reference"]).convert("RGB")) for row, _ in batch]
        ).to(device)
        texts = tokenizer(
            [prompt_for[(row["uid"], row["scene"])] for row, _ in batch]
        ).to(device)
        with torch.inference_mode():
            image_features = model.encode_image(images)
            reference_features = model.encode_image(references)
            text_features = model.encode_text(texts)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            reference_features /= reference_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            scene_scores = (image_features * text_features).sum(dim=-1).cpu()
            reference_scores = (
                (image_features * reference_features).sum(dim=-1).cpu()
            )
        for (row, kind), scene_value, reference_value in zip(
            batch, scene_scores.tolist(), reference_scores.tolist()
        ):
            row[f"{kind}_scene_clip"] = float(scene_value)
            row[f"{kind}_reference_clip"] = float(reference_value)


def summarize(rows: list[dict]) -> dict:
    variants = ("full", *ABLATION_RUNNER.VARIANTS)
    summary = {
        "n": len(rows),
        "identities": len({row["uid"] for row in rows}),
        "scenes": len({row["scene"] for row in rows}),
        "seed": ABLATION_RUNNER.SEED,
        "caption_ccip": statistics.mean(row["caption_ccip"] for row in rows),
        "variants": {},
    }
    for variant in variants:
        for row in rows:
            row[f"{variant}_gain_vs_caption"] = (
                row["caption_ccip"] - row[f"{variant}_ccip"]
            )
            row[f"{variant}_delta_vs_full"] = (
                row[f"{variant}_ccip"] - row["full_ccip"]
            )
            row[f"{variant}_specificity"] = (
                row[f"{variant}_wrong_ccip"] - row[f"{variant}_ccip"]
            )
        values = {
            "ccip": statistics.mean(row[f"{variant}_ccip"] for row in rows),
            "gain_vs_caption": statistics.mean(
                row[f"{variant}_gain_vs_caption"] for row in rows
            ),
            "gain_vs_caption_ci95": bootstrap_subject_mean(
                rows, f"{variant}_gain_vs_caption"
            ),
            "delta_vs_full": statistics.mean(
                row[f"{variant}_delta_vs_full"] for row in rows
            ),
            "delta_vs_full_ci95": bootstrap_subject_mean(
                rows, f"{variant}_delta_vs_full"
            ),
            "specificity": statistics.mean(
                row[f"{variant}_specificity"] for row in rows
            ),
            "scene_clip": statistics.mean(
                row[f"{variant}_scene_clip"] for row in rows
            ),
            "reference_clip": statistics.mean(
                row[f"{variant}_reference_clip"] for row in rows
            ),
            "aesthetic_percentile": statistics.mean(
                row[f"{variant}_aesthetic_percentile"] for row in rows
            ),
        }
        if variant != "full":
            values["elapsed_seconds"] = statistics.mean(
                row[f"{variant}_elapsed"] for row in rows
            )
        summary["variants"][variant] = values
    return summary


def rel(path: str) -> str:
    return os.path.relpath(path, REPORT.parent)


def figure(path: str, label: str, ccip: float) -> str:
    return f"""
      <figure>
        <img src="{html.escape(rel(path))}" alt="{html.escape(label)}" loading="lazy">
        <figcaption><strong>{html.escape(label)}</strong><span>{ccip:.3f}</span></figcaption>
      </figure>"""


def build_html(rows: list[dict], summary: dict) -> str:
    names = {
        "full": "Full",
        "fixed_trust": "Fixed trust",
        "global_only": "Global only",
        "loose_trust": "Loose trust",
        "one_inner": "One inner",
        "global_rms": "Global RMS",
        "raw_score": "Raw score",
        "no_ref_boost": "No ref boost",
        "early_window": "Early only",
        "late_window": "Late only",
        "matched_lora": "Matched LoRA",
        "identity_anchor_const": "Identity anchor (constant)",
        "identity_anchor_late": "Identity anchor (late)",
        "identity_stat_const": "Channel statistics (constant)",
        "identity_stat_late": "Channel statistics (late)",
        "wrong_ref_same_text": "Wrong ref",
    }
    table_rows = []
    for variant, values in summary["variants"].items():
        ci = values["gain_vs_caption_ci95"]
        table_rows.append(
            f"""
        <tr>
          <th>{names[variant]}</th>
          <td>{values['ccip']:.4f}</td>
          <td>{values['gain_vs_caption']:+.4f}</td>
          <td>[{ci[0]:+.4f}, {ci[1]:+.4f}]</td>
          <td>{values['delta_vs_full']:+.4f}</td>
          <td>{values['specificity']:+.4f}</td>
          <td>{values['scene_clip']:.4f}</td>
          <td>{values['reference_clip']:.4f}</td>
          <td>{values['aesthetic_percentile']:.4f}</td>
        </tr>"""
        )
    observations = []
    for row in rows:
        variants = "".join(
            figure(row[variant], names[variant], row[f"{variant}_ccip"])
            for variant in names
        )
        observations.append(
            f"""
      <article>
        <header>
          <h2>u{row['uid']} / {row['scene']}</h2>
          <p>{row['subject_class']} | seed {ABLATION_RUNNER.SEED}</p>
        </header>
        <div class="strip">
          {figure(row['reference'], 'Reference', 0.0)}
          {figure(row['caption'], 'Text-only input', row['caption_ccip'])}
          {variants}
        </div>
      </article>"""
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Krea2 TSD v3 Mechanism Ablation</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #eef1f2;
      --surface: #f9fbfa;
      --text: #172022;
      --muted: #607074;
      --line: #c8d1d2;
      --accent: #087b78;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #101719;
        --surface: #192225;
        --text: #eff4f4;
        --muted: #a8b5b7;
        --line: #344246;
        --accent: #72cbc6;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Helvetica Neue", "Noto Sans", Arial, sans-serif;
      line-height: 1.45;
    }}
    main {{ width: min(1800px, calc(100% - 36px)); margin: auto; padding: 64px 0 100px; }}
    .hero {{ max-width: 1050px; padding-bottom: 56px; }}
    .eyebrow {{ color: var(--accent); font: 700 12px/1 ui-monospace, monospace; letter-spacing: .12em; text-transform: uppercase; }}
    h1 {{ margin: 18px 0; font-size: clamp(48px, 7vw, 96px); line-height: .94; letter-spacing: -.055em; }}
    .hero p {{ max-width: 66ch; color: var(--muted); font-size: 18px; }}
    .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: var(--surface); }}
    table {{ width: 100%; border-collapse: collapse; min-width: 980px; }}
    th, td {{ padding: 14px 16px; text-align: right; border-bottom: 1px solid var(--line); font-variant-numeric: tabular-nums; }}
    th:first-child {{ text-align: left; }}
    thead th {{ color: var(--muted); font-size: 11px; letter-spacing: .06em; text-transform: uppercase; }}
    article {{ padding: 64px 0 24px; border-bottom: 1px solid var(--line); }}
    article header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 24px; }}
    article h2 {{ margin: 0 0 14px; font-size: 32px; letter-spacing: -.035em; }}
    article p {{ margin: 0; color: var(--muted); }}
    .strip {{ display: grid; grid-template-columns: repeat(18, minmax(170px, 1fr)); gap: 10px; overflow-x: auto; padding-bottom: 14px; }}
    figure {{ min-width: 0; margin: 0; }}
    img {{ display: block; width: 100%; aspect-ratio: 1; object-fit: contain; border-radius: 10px; background: var(--surface); }}
    figcaption {{ display: flex; justify-content: space-between; gap: 8px; padding-top: 8px; font-size: 12px; }}
    figcaption span {{ color: var(--muted); font-family: ui-monospace, monospace; }}
    @media (max-width: 700px) {{
      main {{ width: min(100% - 20px, 1800px); padding-top: 36px; }}
      h1 {{ font-size: 46px; }}
      article header {{ display: block; }}
      .strip {{ grid-template-columns: repeat(18, minmax(76vw, 1fr)); }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div class="eyebrow">10 identities | 3 scenes | 1 fixed seed</div>
    <h1>What makes identity distillation stable?</h1>
    <p>The full method is compared against trust-region, optimization, normalization, time-window, reconstruction-anchor, channel-statistic, and wrong-reference controls. Lower CCIP is better. Every ablation output is shown; no output was selected by its score.</p>
  </section>
  <section class="table-wrap">
    <table>
      <thead><tr><th>Variant</th><th>CCIP</th><th>Gain vs text</th><th>95% subject CI</th><th>Delta vs full</th><th>ID margin</th><th>Scene CLIP</th><th>Ref CLIP</th><th>Aesthetic</th></tr></thead>
      <tbody>{''.join(table_rows)}</tbody>
    </table>
  </section>
  {''.join(observations)}
</main>
</body>
</html>
"""


def main() -> int:
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    rows = collect(protocol)
    add_ccip_and_aesthetic_metrics(rows)
    add_clip_metrics(rows, protocol)
    summary = summarize(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    METRICS.write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2),
        encoding="utf-8",
    )
    REPORT.write_text(build_html(rows, summary), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"metrics: {METRICS}")
    print(f"report: {REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

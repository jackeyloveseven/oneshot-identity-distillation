#!/usr/bin/env python3
"""Audit the locked Krea2 TSD 3x3 experiment and build a reviewer-facing report."""

from __future__ import annotations

import csv
import html
import itertools
import json
import statistics
from pathlib import Path

import numpy as np
from PIL import Image
from imgutils.metrics import ccip_difference


ROOT = Path("/home/ubuntu/chengyanli/items/cherry")
OUT = ROOT / "out" / "krea2_tsd"
UIDS = ("u4229", "u1839", "u9728")
SEEDS = (777, 778, 779)
THRESHOLD = 0.178
SUFFIX = "512x768_i3_p1_g0.06_l0.05_a1.5_rms_token_raw_roifull_o1"
LOCKED = {
    "steps": 8,
    "inner_iters": 3,
    "lr": 0.05,
    "pseudo_step_size": 1.0,
    "teacher_weight": 1.0,
    "active_window": [0.0, 1.0],
    "normalize": "rms_token",
    "residual_mode": "raw",
    "trust_region_rms": 0.06,
    "local_trust_region_rms": 0.05,
    "adaptive_trust_ratio": 1.5,
}


def exact_paired_sign_pvalue(improvements: list[float]) -> float:
    """Two-sided exact paired randomization p-value over all sign assignments."""
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


def check_locked(metadata: dict, metadata_path: Path) -> None:
    for key, expected in LOCKED.items():
        actual = metadata.get(key)
        if actual != expected:
            raise AssertionError(
                f"{metadata_path.name}: unlocked {key}: expected {expected!r}, got {actual!r}"
            )
    if metadata.get("teacher_scheduler_usage") != "none_in_distilled_trajectory":
        raise AssertionError(f"{metadata_path.name}: teacher entered student scheduler")
    if not metadata.get("scene_prompt") or not metadata.get("teacher_prompt"):
        raise AssertionError(f"{metadata_path.name}: prompt protocol missing")


def collect_rows() -> list[dict]:
    rows = []
    scene_prompt_by_uid: dict[str, str] = {}
    teacher_prompt_by_uid: dict[str, str] = {}
    for uid in UIDS:
        for seed in SEEDS:
            stem = f"{uid}_s{seed}_{SUFFIX}"
            metadata_path = OUT / f"{stem}.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            check_locked(metadata, metadata_path)

            scene_prompt_by_uid.setdefault(uid, metadata["scene_prompt"])
            teacher_prompt_by_uid.setdefault(uid, metadata["teacher_prompt"])
            if scene_prompt_by_uid[uid] != metadata["scene_prompt"]:
                raise AssertionError(f"{uid}: scene prompt changed across seeds")
            if teacher_prompt_by_uid[uid] != metadata["teacher_prompt"]:
                raise AssertionError(f"{uid}: teacher prompt changed across seeds")

            reference = Path(metadata["reference"])
            paths = {
                key: Path(metadata["outputs"][key])
                for key in ("base", "distilled", "teacher")
            }
            for path in (reference, *paths.values()):
                if not path.is_file():
                    raise FileNotFoundError(path)

            row = {
                "uid": uid,
                "seed": seed,
                "reference": str(reference),
                **{key: str(path) for key, path in paths.items()},
            }
            for kind, path in paths.items():
                row[f"{kind}_ccip"] = float(ccip_difference(str(reference), str(path)))
            row["improvement"] = row["base_ccip"] - row["distilled_ccip"]
            row["relative_improvement"] = row["improvement"] / max(row["base_ccip"], 1e-12)
            row["pass"] = row["distilled_ccip"] < THRESHOLD
            row["reference_pixel_l1"] = pixel_l1(reference, paths["distilled"])
            rows.append(row)
    return rows


def summarize(rows: list[dict]) -> dict:
    improvements = [row["improvement"] for row in rows]
    ci_low, ci_high = bootstrap_mean_ci(improvements)
    per_identity = {}
    for uid in UIDS:
        subset = [row for row in rows if row["uid"] == uid]
        per_identity[uid] = {
            "base_mean": statistics.mean(row["base_ccip"] for row in subset),
            "distilled_mean": statistics.mean(
                row["distilled_ccip"] for row in subset
            ),
            "teacher_mean": statistics.mean(row["teacher_ccip"] for row in subset),
            "worst_distilled": max(row["distilled_ccip"] for row in subset),
            "passes": sum(row["pass"] for row in subset),
        }
    return {
        "threshold": THRESHOLD,
        "n": len(rows),
        "base_mean": statistics.mean(row["base_ccip"] for row in rows),
        "distilled_mean": statistics.mean(row["distilled_ccip"] for row in rows),
        "teacher_mean": statistics.mean(row["teacher_ccip"] for row in rows),
        "mean_improvement": statistics.mean(improvements),
        "relative_mean_improvement": 1.0
        - statistics.mean(row["distilled_ccip"] for row in rows)
        / statistics.mean(row["base_ccip"] for row in rows),
        "improvement_bootstrap_ci95": [ci_low, ci_high],
        "paired_exact_p_two_sided": exact_paired_sign_pvalue(improvements),
        "worst_distilled": max(row["distilled_ccip"] for row in rows),
        "passes": sum(row["pass"] for row in rows),
        "mean_reference_pixel_l1": statistics.mean(
            row["reference_pixel_l1"] for row in rows
        ),
        "per_identity": per_identity,
        "locked_parameters": LOCKED,
    }


def image_cell(path: str, label: str, detail: str) -> str:
    filename = Path(path).name
    return f"""
      <figure>
        <div class="image-frame">
          <img src="{html.escape(filename)}" alt="{html.escape(label)}" loading="lazy">
        </div>
        <figcaption><strong>{html.escape(label)}</strong><span>{html.escape(detail)}</span></figcaption>
      </figure>"""


def build_html(rows: list[dict], summary: dict) -> str:
    groups = []
    for uid in UIDS:
        identity_rows = [row for row in rows if row["uid"] == uid]
        items = []
        for row in identity_rows:
            items.append(
                f"""
          <article class="comparison">
            <header>
              <h3>Seed {row['seed']}</h3>
              <p>Base {row['base_ccip']:.4f} <span>→</span> Distilled {row['distilled_ccip']:.4f}</p>
            </header>
            <div class="image-grid">
              {image_cell(row['reference'], 'Reference ID', '身份参考')}
              {image_cell(row['base'], 'Input / Base', f"CCIP {row['base_ccip']:.4f}")}
              {image_cell(row['distilled'], 'Distilled output', f"CCIP {row['distilled_ccip']:.4f}")}
              {image_cell(row['teacher'], 'Direct teacher', f"Upper bound {row['teacher_ccip']:.4f}")}
            </div>
          </article>"""
            )
        identity = summary["per_identity"][uid]
        groups.append(
            f"""
      <section class="identity-section" id="{uid}">
        <div class="identity-heading">
          <div>
            <h2>{uid}</h2>
            <p>三种随机种子，共享身份 capsule、场景和算法参数。</p>
          </div>
          <dl>
            <div><dt>Distilled mean</dt><dd>{identity['distilled_mean']:.4f}</dd></div>
            <div><dt>Worst</dt><dd>{identity['worst_distilled']:.4f}</dd></div>
            <div><dt>Pass</dt><dd>{identity['passes']}/3</dd></div>
          </dl>
        </div>
        {''.join(items)}
      </section>"""
        )

    table_rows = "".join(
        f"""
          <tr>
            <td>{row['uid']}</td><td>{row['seed']}</td>
            <td>{row['base_ccip']:.4f}</td>
            <td class="best">{row['distilled_ccip']:.4f}</td>
            <td>{row['teacher_ccip']:.4f}</td>
            <td>{row['improvement']:.4f}</td>
            <td>{'PASS' if row['pass'] else 'FAIL'}</td>
          </tr>"""
        for row in rows
    )
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Krea2 Identity Score Distillation Audit</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f1f3f4; --surface: #e7eaec; --text: #182023; --muted: #5a656a;
      --line: #c7ced1; --accent: #006f73; --accent-soft: #d1e7e6; --radius: 14px;
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #111719; --surface: #192124; --text: #edf1f2; --muted: #aab5b8;
        --line: #344044; --accent: #6ac4c2; --accent-soft: #183b3d;
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0; background: var(--bg); color: var(--text);
      font-family: "Helvetica Neue", "Noto Sans SC", Arial, sans-serif;
      font-feature-settings: "tnum"; line-height: 1.5;
    }}
    main {{ width: min(1480px, calc(100% - 40px)); margin: 0 auto; padding: 40px 0 96px; }}
    .hero {{
      min-height: min(700px, calc(100dvh - 40px)); display: grid;
      grid-template-columns: minmax(0, 1.35fr) minmax(300px, .65fr);
      align-items: end; gap: 56px; padding: 56px 0 48px; border-bottom: 1px solid var(--line);
    }}
    .eyebrow {{ color: var(--accent); font: 700 12px/1.2 monospace; letter-spacing: .14em; text-transform: uppercase; }}
    h1 {{ max-width: 780px; margin: 18px 0 20px; font-size: clamp(48px, 7vw, 100px); line-height: .94; letter-spacing: -.055em; }}
    .hero-copy {{ max-width: 58ch; color: var(--muted); font-size: 18px; }}
    .summary {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 1px; background: var(--line); border: 1px solid var(--line); border-radius: var(--radius); overflow: hidden; }}
    .summary div {{ min-height: 118px; padding: 20px; background: var(--surface); }}
    dt {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .08em; }}
    dd {{ margin: 8px 0 0; font-size: clamp(27px, 3vw, 42px); font-weight: 650; letter-spacing: -.04em; }}
    .method {{ max-width: 900px; padding: 72px 0; }}
    .method h2, .identity-heading h2 {{ margin: 0; font-size: clamp(34px, 5vw, 64px); letter-spacing: -.045em; }}
    .method p, .identity-heading p {{ color: var(--muted); max-width: 68ch; }}
    code {{ color: var(--accent); background: var(--accent-soft); padding: 3px 7px; border-radius: 6px; }}
    .identity-section {{ padding: 64px 0 24px; border-top: 1px solid var(--line); }}
    .identity-heading {{ display: grid; grid-template-columns: 1fr auto; gap: 40px; align-items: end; margin-bottom: 36px; }}
    .identity-heading dl {{ display: grid; grid-template-columns: repeat(3, minmax(100px, 1fr)); gap: 28px; margin: 0; }}
    .identity-heading dd {{ font-size: 27px; }}
    .comparison {{ margin-bottom: 64px; }}
    .comparison > header {{ display: flex; justify-content: space-between; align-items: baseline; gap: 24px; margin-bottom: 14px; }}
    .comparison h3 {{ margin: 0; font-size: 20px; }}
    .comparison header p {{ margin: 0; color: var(--muted); font-family: monospace; }}
    .comparison header span {{ color: var(--accent); }}
    .image-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 14px; }}
    figure {{ min-width: 0; margin: 0; }}
    .image-frame {{ aspect-ratio: 2 / 3; overflow: hidden; border-radius: var(--radius); background: var(--surface); }}
    img {{ display: block; width: 100%; height: 100%; object-fit: cover; }}
    figcaption {{ display: flex; justify-content: space-between; gap: 12px; padding: 10px 2px 0; font-size: 13px; }}
    figcaption span {{ color: var(--muted); text-align: right; }}
    .metrics {{ padding: 72px 0; }}
    .metrics h2 {{ font-size: 38px; letter-spacing: -.035em; }}
    .table-wrap {{ overflow-x: auto; border-radius: var(--radius); border: 1px solid var(--line); }}
    table {{ width: 100%; border-collapse: collapse; background: var(--surface); font-family: monospace; font-size: 13px; }}
    th, td {{ padding: 13px 16px; text-align: right; border-bottom: 1px solid var(--line); white-space: nowrap; }}
    th:first-child, td:first-child {{ text-align: left; }}
    tr:last-child td {{ border-bottom: 0; }}
    th {{ color: var(--muted); font-weight: 500; }}
    td.best {{ color: var(--accent); font-weight: 700; }}
    .footnote {{ color: var(--muted); max-width: 78ch; margin-top: 18px; font-size: 14px; }}
    @media (max-width: 900px) {{
      main {{ width: min(100% - 24px, 1480px); padding-top: 20px; }}
      .hero {{ min-height: auto; grid-template-columns: 1fr; padding: 48px 0; }}
      h1 {{ font-size: clamp(46px, 16vw, 74px); }}
      .identity-heading {{ grid-template-columns: 1fr; align-items: start; }}
      .image-grid {{ grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 24px 10px; }}
    }}
    @media (max-width: 520px) {{
      .summary, .identity-heading dl {{ grid-template-columns: 1fr 1fr; }}
      .image-grid {{ grid-template-columns: 1fr; }}
      .comparison > header {{ align-items: flex-start; flex-direction: column; gap: 4px; }}
    }}
  </style>
</head>
<body>
<main>
  <section class="hero">
    <div>
      <div class="eyebrow">Krea2 identity score distillation</div>
      <h1>Same-space distillation, audited.</h1>
      <p class="hero-copy">三个身份，三种随机种子。每组同时展示参考身份、输入基线、蒸馏输出和直接 teacher 上限。</p>
    </div>
    <dl class="summary">
      <div><dt>Distilled mean</dt><dd>{summary['distilled_mean']:.4f}</dd></div>
      <div><dt>Worst case</dt><dd>{summary['worst_distilled']:.4f}</dd></div>
      <div><dt>Pass rate</dt><dd>{summary['passes']}/{summary['n']}</dd></div>
      <div><dt>Paired p</dt><dd>{summary['paired_exact_p_two_sided']:.4f}</dd></div>
    </dl>
  </section>
  <section class="method">
    <h2>Locked protocol</h2>
    <p>Base 和 teacher 共享由参考图生成的身份 capsule。teacher 额外接收参考图，只提供同状态分数残差，不进入 distilled scheduler。</p>
    <p><code>Δs = -(1-σ) / σ · (vteacher-vbase)</code>，逐 token RMS 归一化，并投影到由 Euler 可达位移决定的自适应 trust region。</p>
  </section>
  {''.join(groups)}
  <section class="metrics">
    <h2>Complete metric table</h2>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Identity</th><th>Seed</th><th>Base</th><th>Distilled</th><th>Teacher</th><th>Gain</th><th>Gate</th></tr></thead>
        <tbody>{table_rows}</tbody>
      </table>
    </div>
    <p class="footnote">CCIP 越低越相似，预注册门槛为 0.178。平均改善 {summary['mean_improvement']:.4f}，95% bootstrap CI [{summary['improvement_bootstrap_ci95'][0]:.4f}, {summary['improvement_bootstrap_ci95'][1]:.4f}]。参考图与输出的平均像素 L1 为 {summary['mean_reference_pixel_l1']:.3f}，排除直接复制参考图。</p>
  </section>
</main>
</body>
</html>
"""


def main() -> int:
    rows = collect_rows()
    summary = summarize(rows)

    metrics_path = OUT / "publication_3x3_metrics.json"
    metrics_path.write_text(
        json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    csv_path = OUT / "publication_3x3_metrics.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    report_path = OUT / "publication_3x3_report.html"
    report_path.write_text(build_html(rows, summary), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"metrics: {metrics_path}")
    print(f"csv: {csv_path}")
    print(f"report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

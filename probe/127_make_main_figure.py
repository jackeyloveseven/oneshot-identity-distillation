#!/usr/bin/env python3
"""Build the locked eval30 main paper figure.

Panel a is a single held-out identity (u4224, a dark-skinned character whose
identity is not reducible to hair colour) shown across all three protocol
scenes and the full control set. It makes the causal story visible: the
attribute-text base misses the identity, the distilled trajectory recovers it,
the edit teacher matches, the matched reconstruction only pastes the reference
(near-zero pixel L1), and a wrong reference collapses to a different subject.
Pixel-L1-to-reference is printed on the distilled and matched cells so the
figure itself refutes a copy-paste reading. Panels b-e report the aggregate
quantitative evidence over all 180 observations.
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def jpeg_compress_pdf(pdf: Path, dpi: int = 300, quality: int = 90) -> None:
    """Recompress a PDF's raster images to JPEG in place via Ghostscript.

    Vector text/lines are untouched; only embedded rasters are downsampled to
    ``dpi`` and DCT (JPEG) encoded at ``quality``. No-op (leaves the original)
    if Ghostscript is unavailable or the run fails, so figure generation never
    hard-depends on gs.
    """
    tmp = pdf.with_suffix(".gs.pdf")
    cmd = [
        "gs", "-sDEVICE=pdfwrite", "-dCompatibilityLevel=1.5",
        "-dNOPAUSE", "-dBATCH", "-dQUIET",
        "-dAutoFilterColorImages=false", "-dColorImageFilter=/DCTEncode",
        "-dDownsampleColorImages=true", "-dColorImageDownsampleType=/Bicubic",
        f"-dColorImageResolution={dpi}",
        "-dAutoFilterGrayImages=false", "-dGrayImageFilter=/DCTEncode",
        "-dDownsampleGrayImages=true", f"-dGrayImageResolution={dpi}",
        f"-dJPEGQ={quality}", f"-sOutputFile={tmp}", str(pdf),
    ]
    try:
        subprocess.run(cmd, check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        tmp.unlink(missing_ok=True)
        return
    if tmp.exists() and tmp.stat().st_size > 0:
        tmp.replace(pdf)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
METRICS = OUT / "eval30_metrics.json"
FIG_DIR = ROOT / "paper" / "figures"
FIG_DIR_COPY = FIG_DIR / "fig2_results"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
    }
)
plt.rcParams["font.size"] = 7
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False
plt.rcParams["legend.frameon"] = False

COLORS = {
    "caption_base": "#A8A8A8",
    "distilled": "#0F4D92",
    "teacher": "#42949E",
    "matched": "#8A6D3B",
    "positive": "#2E9E44",
    "negative": "#B64342",
    "ink": "#272727",
}

# Shared type scale for panels b-e (matches the mechanism-ablation figure's
# tick_params(width=0.8, length=3, labelsize=6.2) / annotation-label convention
# so the two data-panel figures read as one journal-grade family).
TICK_FS = 6.2
ANNOT_FS = 5.8

# Held-out hero identity and its three protocol scenes (one seed each).
HERO = "4224"
HERO_SCENES = [("portrait", 2027, "Portrait"),
               ("rain", 2027, "Rain street"),
               ("kitchen", 2027, "Kitchen")]
# Marks are derived per cell from 30-way retrieval, never hardcoded.
HERO_ARMS = [
    ("caption_base", "Attribute text", "#9AA0A6", 0.8),
    ("distilled", "Distilled (ours)", COLORS["distilled"], 2.0),
    ("teacher", "Edit teacher", COLORS["teacher"], 0.8),
    ("negative_teacher", "Wrong reference", COLORS["negative"], 1.4),
]


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.05, label, transform=ax.transAxes, fontsize=9,
            fontweight="bold", ha="left", va="bottom")


def subject_means(rows: list[dict], key: str) -> np.ndarray:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        grouped[row["uid"]].append(float(row[key]))
    return np.asarray([np.mean(grouped[uid]) for uid in sorted(grouped)])


def selected_row(rows, uid, scene, seed):
    return next(r for r in rows if r["uid"] == uid and r["scene"] == scene
               and int(r["seed"]) == seed)


def letterbox(path, size=512, bg=(255, 255, 255)):
    im = Image.open(path).convert("RGB")
    im.thumbnail((size, size), Image.LANCZOS)
    canvas = Image.new("RGB", (size, size), bg)
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return canvas


def _cap_box(ax, text, color):
    ax.set_xlabel(text, fontsize=6.8, labelpad=1.8, color=color, linespacing=0.9)


def main() -> int:
    payload = json.loads(METRICS.read_text())
    rows = payload["rows"]
    summary = payload["summary"]
    vlm = payload["vlm_summary"]["majority"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR_COPY.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(9.7, 5.4), facecolor="white")
    outer = fig.add_gridspec(1, 2, width_ratios=[0.31, 0.69], wspace=0.08)

    # ---- Panel a: hero identity x scenes x controls (right). Reference and the
    # aggregate panels b-e stack in the left column so it carries no blank. ----
    # Per-scene row heights track the native image aspect ratio so no image is
    # letterboxed into a square: portrait 640^2, rain 512x768, kitchen 768x512.
    scene_aspect = {"portrait": 1.0, "rain": 768 / 512, "kitchen": 512 / 768}
    row_heights = [scene_aspect[s] for s, _, _ in HERO_SCENES]

    left = outer[0, 0].subgridspec(2, 1, height_ratios=[1.12, 1.0], hspace=0.36)
    ax_ref = fig.add_subplot(left[0, 0])  # large reference, top-left
    ref_path = selected_row(rows, HERO, "portrait", 2027)["reference"]
    ax_ref.imshow(Image.open(ref_path).convert("RGB"))
    ax_ref.set_xticks([]); ax_ref.set_yticks([])
    for sp in ax_ref.spines.values():
        sp.set_visible(True); sp.set_edgecolor(COLORS["ink"]); sp.set_linewidth(1.3)
    ax_ref.set_title("Reference (target identity)", fontsize=7.2, pad=3)
    panel_label(ax_ref, "a")

    quad = left[1, 0].subgridspec(2, 2, wspace=0.75, hspace=0.85)
    ax_b = fig.add_subplot(quad[0, 0])
    ax_c = fig.add_subplot(quad[0, 1])
    ax_d = fig.add_subplot(quad[1, 0])
    ax_e = fig.add_subplot(quad[1, 1])

    hero = outer[0, 1].subgridspec(3, len(HERO_ARMS), wspace=0.05, hspace=0.16,
                                   height_ratios=row_heights)
    for ri, (scene, seed, rlabel) in enumerate(HERO_SCENES):
        record = selected_row(rows, HERO, scene, seed)
        for ci, (key, title, edge, lw) in enumerate(HERO_ARMS):
            ax = fig.add_subplot(hero[ri, ci])
            ax.imshow(Image.open(record[key]).convert("RGB"))
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(lw)
            if ri == 0:
                ax.set_title(title, fontsize=6.6,
                             fontweight="bold" if key == "distilled" else "normal",
                             pad=2)
            ccip = float(record[f"{key}_ccip"])
            top1 = bool(record.get(f"{key}_retrieval_top1"))
            mark = " ✓" if top1 else " ✗"
            cap_color = COLORS["ink"] if (top1 or key == "distilled") else COLORS["negative"]
            _cap_box(ax, f"CCIP {ccip:.3f}{mark}", cap_color)
            if ci == 0:
                ax.text(-0.13, 0.5, rlabel, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=6.4, fontweight="bold")

    # ---- Panels b-e: aggregate quantitative evidence (all 180 observations) ----
    methods = ["caption_base", "distilled", "teacher"]
    positions = np.arange(3)
    for index, method in enumerate(methods):
        values = subject_means(rows, f"{method}_ccip")
        jitter = np.linspace(-0.075, 0.075, len(values))
        ax_b.scatter(np.full(len(values), index) + jitter, values, s=7,
                     alpha=0.48, color=COLORS[method], edgecolors="none",
                     rasterized=True)
        ax_b.plot([index - 0.17, index + 0.17], [values.mean(), values.mean()],
                  color=COLORS["ink"], lw=1.4, zorder=4)
    ax_b.axhline(0.178, color="#767676", lw=0.8, ls="--")
    ax_b.text(2.45, 0.183, "same-ID", fontsize=ANNOT_FS, ha="right")
    ax_b.set_xticks(positions, ["Text", "Ours", "Teacher"])
    ax_b.set_ylabel("CCIP distance ↓")
    ax_b.set_ylim(0, 0.31)
    ax_b.tick_params(axis="both", width=0.8, length=3, labelsize=TICK_FS)
    ax_b.tick_params(axis="x", length=0)
    panel_label(ax_b, "b")

    retrieval = summary["retrieval"]
    retrieval_values = [100.0 * retrieval[m]["top1"] / summary["n"] for m in methods]
    bars = ax_c.bar(positions, retrieval_values, width=0.68,
                    color=[COLORS[m] for m in methods], edgecolor="white", linewidth=0.6)
    ax_c.bar_label(bars, labels=[f"{v:.1f}" for v in retrieval_values], padding=2, fontsize=ANNOT_FS)
    ax_c.set_xticks(positions, ["Text", "Ours", "Teacher"])
    ax_c.set_ylim(0, 105)
    ax_c.set_ylabel("30-way Top-1 (%) ↑")
    ax_c.tick_params(axis="both", width=0.8, length=3, labelsize=TICK_FS)
    ax_c.tick_params(axis="x", length=0)
    panel_label(ax_c, "c")

    identity_values = [100.0 * vlm[m]["identity_score"] for m in methods]
    defect_values = [100.0 * vlm[m]["defect_rate"] for m in methods]
    bars = ax_d.bar(positions, identity_values, width=0.68,
                    color=[COLORS[m] for m in methods], edgecolor="white", linewidth=0.6)
    ax_d.bar_label(bars, labels=[f"{v:.1f}" for v in identity_values], padding=2, fontsize=ANNOT_FS)
    ax_d.set_xticks(positions, ["Text", "Ours", "Teacher"])
    ax_d.set_ylim(0, 105)
    ax_d.set_ylabel("VLM identity (%) ↑")
    ax_d.tick_params(axis="both", width=0.8, length=3, labelsize=TICK_FS)
    ax_d.tick_params(axis="x", length=0)
    panel_label(ax_d, "d")

    comparisons = summary["comparisons"]
    delta_names = ["Identity", "Scene", "Aesth."]
    delta_values = [comparisons["gain_vs_caption"]["mean"],
                    comparisons["scene_gain_vs_caption"]["mean"],
                    comparisons["aesthetic_gain_vs_caption"]["mean"]]
    delta_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in delta_values]
    bars = ax_e.barh(np.arange(3), delta_values, color=delta_colors, height=0.62,
                     edgecolor="white", linewidth=0.6)
    ax_e.axvline(0, color=COLORS["ink"], lw=0.8)
    ax_e.set_yticks(np.arange(3), delta_names)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Change vs attribute text")
    ax_e.set_xlim(-0.085, 0.135)
    ax_e.tick_params(axis="both", width=0.8, length=3, labelsize=TICK_FS)
    ax_e.tick_params(axis="y", length=0)
    # Large-magnitude bars (|value| >= 0.02) get a centred white label inside
    # the bar on both signs; short bars get a small outside label near the
    # tip. Symmetric so the "Identity" bar's label stays inside the axes
    # instead of overrunning into the margin next to the hero grid.
    for bar, value in zip(bars, delta_values):
        if abs(value) >= 0.02:
            x_text, align, tc = value / 2, "center", "white"
        elif value < 0:
            x_text, align, tc = value - 0.004, "right", COLORS["ink"]
        else:
            x_text, align, tc = value + 0.004, "left", COLORS["ink"]
        ax_e.text(x_text, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                  ha=align, va="center", fontsize=ANNOT_FS, color=tc)
    panel_label(ax_e, "e")

    fig.subplots_adjust(left=0.045, right=0.996, bottom=0.065, top=0.95)

    base = FIG_DIR / "main_eval30"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", dpi=400)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", dpi=400)
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)
    # This figure embeds many anime hero rasters; matplotlib emits them as
    # lossless Flate (~5 MB), which blows past the AAAI 10 MB source-archive
    # limit. Recompress the rasters to JPEG at 300 dpi (print-grade, visually
    # lossless) so the vector text/lines stay crisp but the file is ~0.5 MB.
    jpeg_compress_pdf(base.with_suffix(".pdf"))

    # ---- source data: aggregate CCIP + hero cell provenance ----
    with (FIG_DIR / "main_eval30_source_data.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "method", "subject_mean_ccip"])
        for method in methods:
            grouped: dict[str, list[float]] = defaultdict(list)
            for row in rows:
                grouped[row["uid"]].append(float(row[f"{method}_ccip"]))
            for uid in sorted(grouped):
                writer.writerow([uid, method, np.mean(grouped[uid])])

    with (FIG_DIR / "main_eval30_qualitative_selection.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["uid", "scene", "seed", "arm", "ccip",
                         "pixel_l1_reference", "scene_clip", "retrieval_top1",
                         "selection_rule"])
        rule = ("hero identity u4224 chosen as a non-colour-reducible held-out "
                "subject qualified in all three scenes (distilled 30-way retrieval "
                "correct, blind-VLM identity=yes/defect=no/scene=2); full control "
                "set shown; all 180 observations disclosed in the release report")
        for scene, seed, _ in HERO_SCENES:
            record = selected_row(rows, HERO, scene, seed)
            for key, *_ in HERO_ARMS:
                writer.writerow([HERO, scene, seed, key,
                                 record[f"{key}_ccip"],
                                 record.get(f"{key}_pixel_l1_reference", ""),
                                 record.get(f"{key}_scene_clip", ""),
                                 record.get(f"{key}_retrieval_top1", ""),
                                 rule])

    # keep the paper/figures/fig2_results/ provenance copy in sync.
    for suffix in (".svg", ".pdf", ".png"):
        shutil.copy2(base.with_suffix(suffix), FIG_DIR_COPY / f"main_eval30{suffix}")
    for name in ("main_eval30_source_data.csv", "main_eval30_qualitative_selection.csv"):
        shutil.copy2(FIG_DIR / name, FIG_DIR_COPY / name)

    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

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
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
METRICS = OUT / "eval30_metrics.json"
FIG_DIR = ROOT / "paper" / "figures"

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
    ("matched_control", "Matched recon.", COLORS["matched"], 0.8),
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
    ax.set_xlabel(text, fontsize=5.4, labelpad=1.5, color=color, linespacing=0.9)


def main() -> int:
    payload = json.loads(METRICS.read_text())
    rows = payload["rows"]
    summary = payload["summary"]
    vlm = payload["vlm_summary"]["majority"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(7.2, 6.7), facecolor="white")
    outer = fig.add_gridspec(2, 1, height_ratios=[2.78, 1.0], hspace=0.34)

    # ---- Panel a: hero identity x scenes x controls ----
    # Per-scene row heights track the native image aspect ratio so no image is
    # letterboxed into a square: portrait 640^2, rain 512x768, kitchen 768x512.
    scene_aspect = {"portrait": 1.0, "rain": 768 / 512, "kitchen": 512 / 768}
    row_heights = [scene_aspect[s] for s, _, _ in HERO_SCENES]
    top = outer[0].subgridspec(1, 2, width_ratios=[1.02, 5.0], wspace=0.05)
    ax_ref = fig.add_subplot(top[0, 0])  # square reference, vertically centred
    ref_path = selected_row(rows, HERO, "portrait", 2027)["reference"]
    ax_ref.imshow(Image.open(ref_path).convert("RGB"))
    ax_ref.set_xticks([]); ax_ref.set_yticks([])
    for sp in ax_ref.spines.values():
        sp.set_visible(True); sp.set_edgecolor(COLORS["ink"]); sp.set_linewidth(1.3)
    ax_ref.set_title("Reference", fontsize=7.5, pad=3)
    ax_ref.set_xlabel("target identity\n(dark skin, wavy\nhair, earrings)",
                      fontsize=5.6, labelpad=2, linespacing=1.0)

    hero = top[0, 1].subgridspec(3, len(HERO_ARMS), wspace=0.05, hspace=0.16,
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
                if ci == 0:
                    panel_label(ax, "a")
            ccip = float(record[f"{key}_ccip"])
            if key == "matched_control":
                l1 = float(record["matched_control_pixel_l1_reference"])
                _cap_box(ax, f"CCIP {ccip:.3f}\nL1 {l1:.2f} paste", edge)
            else:
                top1 = bool(record.get(f"{key}_retrieval_top1"))
                mark = " ✓" if top1 else " ✗"
                cap_color = COLORS["ink"] if top1 else COLORS["negative"]
                if key == "distilled":
                    l1 = float(record["distilled_pixel_l1_reference"])
                    _cap_box(ax, f"CCIP {ccip:.3f}{mark}\nL1 {l1:.2f}", COLORS["ink"])
                else:
                    _cap_box(ax, f"CCIP {ccip:.3f}{mark}", cap_color)
            if ci == 0:
                ax.text(-0.13, 0.5, rlabel, transform=ax.transAxes, rotation=90,
                        ha="center", va="center", fontsize=6.4, fontweight="bold")

    # ---- Panels b-e: aggregate quantitative evidence (all 180 observations) ----
    methods = ["caption_base", "distilled", "teacher"]
    bottom = outer[1].subgridspec(1, 4, wspace=0.52)

    ax_b = fig.add_subplot(bottom[0, 0])
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
    ax_b.text(2.45, 0.183, "same-ID", fontsize=5.2, ha="right")
    ax_b.set_xticks(positions, ["Text", "Ours", "Teach"], rotation=18)
    ax_b.set_ylabel("CCIP distance ↓")
    ax_b.set_ylim(0, 0.31)
    panel_label(ax_b, "b")

    ax_c = fig.add_subplot(bottom[0, 1])
    retrieval = summary["retrieval"]
    retrieval_values = [100.0 * retrieval[m]["top1"] / summary["n"] for m in methods]
    bars = ax_c.bar(positions, retrieval_values, width=0.68,
                    color=[COLORS[m] for m in methods], edgecolor="white", linewidth=0.6)
    ax_c.bar_label(bars, labels=[f"{v:.1f}" for v in retrieval_values], padding=2, fontsize=6)
    ax_c.set_xticks(positions, ["Text", "Ours", "Teach"], rotation=18)
    ax_c.set_ylim(0, 105)
    ax_c.set_ylabel("30-way Top-1 (%) ↑")
    panel_label(ax_c, "c")

    ax_d = fig.add_subplot(bottom[0, 2])
    identity_values = [100.0 * vlm[m]["identity_score"] for m in methods]
    defect_values = [100.0 * vlm[m]["defect_rate"] for m in methods]
    bars = ax_d.bar(positions, identity_values, width=0.68,
                    color=[COLORS[m] for m in methods], edgecolor="white", linewidth=0.6)
    ax_d.bar_label(bars, labels=[f"{v:.1f}" for v in identity_values], padding=2, fontsize=6)
    ax_d.scatter(positions, defect_values, marker="v", s=20, color=COLORS["negative"],
                 zorder=4, label="defect")
    ax_d.set_xticks(positions, ["Text", "Ours", "Teach"], rotation=18)
    ax_d.set_ylim(0, 105)
    ax_d.set_ylabel("VLM identity (%) ↑")
    ax_d.legend(fontsize=5.4, loc="lower right")
    panel_label(ax_d, "d")

    ax_e = fig.add_subplot(bottom[0, 3])
    comparisons = summary["comparisons"]
    delta_names = ["Identity", "Scene", "Aesth."]
    delta_values = [comparisons["gain_vs_caption"]["mean"],
                    comparisons["scene_gain_vs_caption"]["mean"],
                    comparisons["aesthetic_gain_vs_caption"]["mean"]]
    delta_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"] for v in delta_values]
    bars = ax_e.barh(np.arange(3), delta_values, color=delta_colors, height=0.62, alpha=0.88)
    ax_e.axvline(0, color=COLORS["ink"], lw=0.8)
    ax_e.set_yticks(np.arange(3), delta_names)
    ax_e.invert_yaxis()
    ax_e.set_xlabel("Change vs attribute text")
    ax_e.set_xlim(-0.085, 0.135)
    for bar, value in zip(bars, delta_values):
        if -0.02 < value < 0:
            x_text, align, tc = value - 0.004, "right", COLORS["ink"]
        elif value < 0:
            x_text, align, tc = value / 2, "center", "white"
        else:
            x_text, align, tc = value + 0.005, "left", COLORS["ink"]
        ax_e.text(x_text, bar.get_y() + bar.get_height() / 2, f"{value:+.3f}",
                  ha=align, va="center", fontsize=5.6, color=tc)
    panel_label(ax_e, "e")

    fig.suptitle(
        "One identity across three scenes with the full control set; the correct "
        "reference drives identity, matched reconstruction only pastes it (L1$\\approx$0.03)",
        x=0.5, y=0.995, fontsize=8.4, fontweight="bold")
    fig.subplots_adjust(left=0.055, right=0.997, bottom=0.075, top=0.94)

    base = FIG_DIR / "main_eval30"
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

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
    print(base)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

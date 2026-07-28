#!/usr/bin/env python3
"""Supplementary extended qualitative gallery: 12 held-out identities across the
three scenes, each with reference, attribute-text baseline, distilled output,
and wrong-reference control. Scenes are ordered rain/kitchen/portrait so the
re-posed compositions lead. Distilled recovers identity; wrong reference
collapses to a visibly different subject."""
from __future__ import annotations
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "out" / "krea2_tsd_eval30_tagged_v3"
FIG = ROOT / "paper" / "figures"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
})
d = json.loads((OUT / "eval30_metrics.json").read_text())
rows = {(r["uid"], r["scene"], int(r["seed"])): r for r in d["rows"]}
INK, BLUE, RED, GREY = "#272727", "#0F4D92", "#B64342", "#9AA0A6"

GAL = [
    ("1317", "rain", 2027, "blue / red-eye"), ("1391", "rain", 2027, "pale blonde"),
    ("1620", "rain", 2027, "black + eyepatch"), ("2364", "rain", 2027, "blonde hairband"),
    ("9993", "kitchen", 2027, "black + glasses"), ("3090", "kitchen", 2027, "silver + gem"),
    ("5319", "kitchen", 2028, "braided crown"), ("4224", "kitchen", 2027, "dark skin"),
    ("68", "portrait", 2027, "vivid red"), ("943", "portrait", 2027, "silver androgyn."),
    ("2542", "portrait", 2027, "brown bun"), ("1070", "portrait", 2027, "flower pin"),
]
COLS = [("reference", "Reference"), ("caption_base", "Attribute text"),
        ("distilled", "Distilled (ours)"), ("negative_teacher", "Wrong reference")]


def lb(path, size=420):
    im = Image.open(path).convert("RGB"); im.thumbnail((size, size), Image.LANCZOS)
    c = Image.new("RGB", (size, size), (255, 255, 255)); c.paste(im, ((size-im.width)//2, (size-im.height)//2)); return c


nrow = len(GAL)
fig = plt.figure(figsize=(7.0, nrow * 1.72), facecolor="white")
gs = fig.add_gridspec(nrow, len(COLS), wspace=0.03, hspace=0.34)
for ri, (uid, scene, seed, note) in enumerate(GAL):
    r = rows[(uid, scene, seed)]
    for ci, (key, title) in enumerate(COLS):
        ax = fig.add_subplot(gs[ri, ci])
        ax.imshow(lb(r[key])); ax.set_xticks([]); ax.set_yticks([])
        if key == "distilled":
            cap, edge, lw, cc = f"CCIP {float(r['distilled_ccip']):.3f} ✓", BLUE, 1.8, INK
        elif key == "caption_base":
            cap, edge, lw, cc = f"CCIP {float(r['caption_base_ccip']):.3f}", GREY, 0.6, INK
        elif key == "negative_teacher":
            cap, edge, lw, cc = f"CCIP {float(r['negative_teacher_ccip']):.3f} ✗", RED, 1.2, RED
        else:
            cap, edge, lw, cc = None, INK, 1.0, INK
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(lw)
        if ri == 0:
            ax.set_title(title, fontsize=8, fontweight="bold", pad=4)
        if cap:
            ax.set_xlabel(cap, fontsize=6.4, labelpad=2, color=cc)
        if ci == 0:
            ax.set_ylabel(f"u{uid}\n{scene}\n{note}", fontsize=6.6, rotation=0,
                          ha="right", va="center", labelpad=15)
fig.suptitle("Extended identity-transfer gallery: 12 held-out subjects across three scenes",
             fontsize=10, fontweight="bold", y=0.997)
fig.subplots_adjust(left=0.10, right=0.997, top=0.965, bottom=0.006)
base = FIG / "supp_gallery"
for ext, kw in [(".pdf", {}), (".svg", {}), (".png", {"dpi": 200}), (".tiff", {"dpi": 300})]:
    fig.savefig(base.with_suffix(ext), bbox_inches="tight", **kw)
plt.close(fig)

with (FIG / "supp_gallery_source_data.csv").open("w", newline="") as h:
    wtr = csv.writer(h)
    wtr.writerow(["uid", "scene", "seed", "note", "caption_base_ccip", "distilled_ccip",
                  "negative_teacher_ccip", "distilled_retrieval_top1"])
    for uid, scene, seed, note in GAL:
        r = rows[(uid, scene, seed)]
        wtr.writerow([uid, scene, seed, note, r["caption_base_ccip"], r["distilled_ccip"],
                      r["negative_teacher_ccip"], r["distilled_retrieval_top1"]])
print("wrote", base)

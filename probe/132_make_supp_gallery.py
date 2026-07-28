#!/usr/bin/env python3
"""Supplementary qualitative gallery on high-quality datafactory character
references (out/df_showcase_pilot/). These are a QUALITATIVE showcase of the
locked TSD method on out-of-distribution, higher-fidelity anime identities;
they are NOT part of the quantitative 30-identity evaluation and carry no
numeric claim. We show the reliably-clean re-posed scenes (rain three-quarter
body, kitchen waist-up); the close portrait is the method's weakest scene
(quantified in the main paper) and is intentionally not curated here. One
reference image drives an identity-preserving generation with a frozen
text-to-image model and no per-subject training."""
from __future__ import annotations
import glob
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHOW = ROOT / "out" / "df_showcase_pilot"
FIG = ROOT / "paper" / "figures"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8,
})
INK, BLUE = "#272727", "#0F4D92"

# best-of-n clean identities (label, id, note)
GAL = [
    ("Blue knight", "saber", "blond bun, blue armor"),
    ("Thunder boy", "zenitsu", "blond, yellow haori"),
    ("Flame haori", "rengoku", "flame-tipped hair"),
    ("Water swordsman", "tanjiro", "red hair, green checkers"),
    ("Crimson noble", "scarlet", "red-black hair, gothic"),
]
# columns: reference + two reliably-clean re-posed scenes
COLS = [("reference", "Reference"), ("rain", "Rain street"), ("kitchen", "Kitchen")]


def load(ident, key):
    if key == "reference":
        return Image.open(SHOW / "refs" / f"{ident}.png").convert("RGB")
    hit = glob.glob(str(SHOW / f"{ident}_{key}_*_distilled.png"))
    return Image.open(hit[0]).convert("RGB") if hit else None


nrow = len(GAL)
# reference square (1:1), rain tall (2:3), kitchen wide (3:2); size columns so
# each image sits at native aspect with no stretch (imshow keeps aspect='equal').
fig = plt.figure(figsize=(7.0, nrow * 1.5), facecolor="white")
gs = fig.add_gridspec(nrow, len(COLS), wspace=0.04, hspace=0.16,
                      width_ratios=[1.0, 0.85, 1.3])
for ri, (label, ident, note) in enumerate(GAL):
    for ci, (key, title) in enumerate(COLS):
        ax = fig.add_subplot(gs[ri, ci])
        im = load(ident, key)
        if im is not None:
            ax.imshow(im, aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        edge = BLUE if key != "reference" else INK
        lw = 1.8 if key != "reference" else 1.2
        for sp in ax.spines.values():
            sp.set_visible(True); sp.set_edgecolor(edge); sp.set_linewidth(lw)
        if ri == 0:
            ax.set_title(title, fontsize=9, fontweight="bold", pad=4)
        if ci == 0:
            ax.set_ylabel(f"{label}\n({note})", fontsize=7.2, rotation=0,
                          ha="right", va="center", labelpad=14)
fig.suptitle("One reference image → identity-preserving generation in new scenes, "
             "training-free (distilled, ours)", fontsize=10, fontweight="bold", y=0.995)
fig.text(0.5, 0.005, "Qualitative showcase on out-of-distribution high-fidelity "
         "anime identities; not part of the 30-identity quantitative evaluation.",
         ha="center", fontsize=6.6, style="italic", color="#555555")
fig.subplots_adjust(left=0.13, right=0.997, top=0.955, bottom=0.03)
base = FIG / "supp_gallery"
for ext, kw in [(".pdf", {"dpi": 300}), (".svg", {"dpi": 300}), (".png", {"dpi": 200})]:
    fig.savefig(base.with_suffix(ext), bbox_inches="tight", **kw)
plt.close(fig)
print("wrote", base)

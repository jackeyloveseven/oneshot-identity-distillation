#!/usr/bin/env python
"""
94_final_figure.py -- assemble the decisive cross-model comparison figures.
Pure post-processing (no GPU). Two outputs:
  (1) attn_quant_comparison.png -- per-layer ref-attention (Z vs Lumina strip/bound)
      + causal ref_bias sweep (Lumina) showing mass rises but identity never does.
  (2) montage_decisive_ZvsLumina.png -- paired visual: same u2 ref, Z-strip WORKS,
      Lumina-strip FAILS, Lumina-forced washes, Lumina-rope copy-pastes.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ZDIR = "/home/ubuntu/chengyanli/items/cherry/out/zimage_causal"
LDIR = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
OUT_QUANT = f"{LDIR}/attn_quant_comparison.png"
OUT_MONT = f"{LDIR}/montage_decisive_ZvsLumina.png"

z = json.load(open(f"{ZDIR}/summary.json"))
lA = json.load(open(f"{LDIR}/partA_summary.json"))
lC = json.load(open(f"{LDIR}/partC_w2_summary.json"))

def series(d):
    layers = sorted(int(k) for k in d)
    return layers, [d[str(l)]["mean_ref_mass"] for l in layers]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
ax = axes[0]
for cond, lbl, c, m in [
    ("Z_strip_b16-25_w4", "Z-Image strip (pos-free key) -- IDENTITY WORKS", "tab:blue", "o-"),
    ("Z_keep_b16-25_w4",  "Z-Image keep (pos-bound key)", "tab:cyan", "s--"),
]:
    xl, yl = series(z[cond]); ax.plot(xl, yl, m, label=lbl, color=c)
for cond, lbl, c, m in [
    ("A1_strip_b16-23_w2", "Lumina strip (pos-free key) -- IDENTITY FAILS", "tab:red", "o-"),
    ("A1_rope_b16-23_w2",  "Lumina rope (pos-bound key) -- copy-paste", "tab:orange", "s--"),
]:
    xl, yl = series(lA[cond]); ax.plot(xl, yl, m, label=lbl, color=c)
ax.set_xlabel("layer id"); ax.set_ylabel("mean softmax attention mass\n(target image query -> injected ref-key columns)")
ax.set_title("(a) Per-layer ref-attention: comparable mass, opposite outcome\n"
             "Lumina attends to pos-free keys as much as Z does -- yet no identity")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.25)

ax = axes[1]
tags = sorted(lC.keys(), key=lambda t: lC[t]["ref_bias"])
biases = [lC[t]["ref_bias"] for t in tags]
masses = [sum(v["mean_ref_mass"] for v in lC[t]["per_layer"].values()) / len(lC[t]["per_layer"]) for t in tags]
ax.plot(biases, masses, "o-", color="tab:purple", label="attention mass on ref key")
ax.set_xlabel("ref_bias  (forced additive pre-softmax logit on ref columns, all heads)")
ax.set_ylabel("mean ref-attention mass (Lumina band 8-15)")
ax.set_title("(b) Causal knob FAILS: forcing attention 0.09->0.98\nrouted 0 identity (image washes to blank/noise)")
ax.axhline(0.070, ls=":", color="tab:blue", lw=1.5, label="Z-Image working point (0.07)")
ax.set_ylim(0, 1.05); ax.legend(fontsize=8, loc="center right"); ax.grid(alpha=0.25)
for x, y in zip(biases, masses):
    ax.annotate("wash" if x > 0 else "clean\n(no id)", (x, y), textcoords="offset points",
                xytext=(4, -12 if x > 0 else 6), fontsize=7, color="gray")
plt.tight_layout(); plt.savefig(OUT_QUANT, dpi=150); print("saved:", OUT_QUANT)

# ---- paired visual montage ----
def lab(path_or_im, txt, cell=512, sub=""):
    im = (Image.open(path_or_im).convert("RGB") if isinstance(path_or_im, str) else path_or_im).resize((cell, cell)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, cell, 42], fill="black")
    d.text((6, 4), txt, fill="white")
    if sub: d.text((6, 24), sub, fill="#9fe")
    return im

cell = 512
ref = "/mnt/local/smoke_anime/anime_260623/img/u2_ref.png"
panels = [
    (ref, "REF u2", "pink hair, purple eyes, red pendant"),
    (f"{ZDIR}/Z_strip_b16-25_w4.png", "Z-Image  strip", "IDENTITY TRANSFERS (content-addressed)"),
    ("/home/ubuntu/chengyanli/items/cherry/out/lumina/u2_depth_w2_b16-23.png", "Lumina  strip", "NO identity (grey hair) despite mass"),
    (f"{LDIR}/C_w2_biassweep_rb8.png", "Lumina  strip + forced attn", "forced mass 0.69 -> WASH, still no id"),
    ("/home/ubuntu/chengyanli/items/cherry/out/lumina/u2_refsizesweep_rope_refsize1024_w2_b21-25.png",
     "Lumina  rope (pos-bound)", "copy-paste: V carries id, pos-locked"),
]
imgs = [lab(p, t, cell, s) for p, t, s in panels]
canvas = Image.new("RGB", (cell * len(imgs), cell), "white")
for c, im in enumerate(imgs):
    canvas.paste(im, (c * cell, 0))
canvas.save(OUT_MONT)
print("saved:", OUT_MONT)

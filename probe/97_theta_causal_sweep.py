#!/usr/bin/env python
"""
97_theta_causal_sweep.py -- SINGLE-MODEL (Z-Image only) causal isolation of
"RoPE phase geometry" as the determinant behind the Z-vs-Lumina dissociation
found in out/lumina/causal/FINDINGS.json (item2/item3): Z's position-free
(strip) ref-key gives PEAKED/content-selective ref-attention (structured
identity transfers); Lumina's gives DIFFUSE ref-attention (mean-color bleed
only). Candidate cause = RoPE theta (Z=256 vs Lumina=10000), but N=2 could not
isolate it from GQA/depth/text-encoder/data confounds.

This script does NOT compare across models. It uses ONLY Z-Image (frozen pure
T2I base, no LoRA) with a NEW refpos="thetasweep" surgery added to
30_kv_inject.py: the injected ref key is rotated with an actual RoPE at a
SWEEPABLE theta (same center-aligned ref-grid position at every theta, so
theta is the ONLY thing that changes across conditions -- no model/GQA/depth/
data confound). Theta grid: 256 (Z native) -> 1000 -> 4000 -> 10000 (Lumina's
baked theta), on the SAME identity band (16-25), weight (w=4), subject (u2:
pink hair / purple eyes / red pendant), seed, steps as the established
Z-strip-works reference point.

Prediction if phase geometry is causal: theta up -> per-query ref-attention
peak DOWN, normalized entropy UP (drifting from Z's 0.70 anchor toward
Lumina's 0.81 anchor) -> AND visual identity degrades from structured
(hair+eyes+pendant) to diffuse mean-color bleed (own face + pink tint only),
reproducing Lumina's failure mode inside one frozen model.

Outputs:
  out/zimage_causal/theta_sweep/theta{T}.png            (per-theta generation)
  out/zimage_causal/theta_sweep/theta_sweep_summary.json (per-theta selectivity)
  out/lumina/causal/montage_theta_causal_sweep.png       (ref | t256 | t1000 | t4000 | t10000 | strip-anchor)
  out/lumina/causal/theta_causal_doseresponse.png        (peak/entropy vs theta, with Z/Lumina anchor lines)
"""
import os
os.environ["REF"] = "/mnt/local/smoke_anime/anime_260623/img"  # override BEFORE importing 30 (module-level read)

import importlib.util, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
ZOUT = "/home/ubuntu/chengyanli/items/cherry/out/zimage_causal"
TSOUT = f"{ZOUT}/theta_sweep"
LDIR = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
os.makedirs(TSOUT, exist_ok=True)
os.makedirs(LDIR, exist_ok=True)

spec = importlib.util.spec_from_file_location("zimg30_thetasweep", os.path.join(HERE, "30_kv_inject.py"))
Z = importlib.util.module_from_spec(spec)
sys.argv = ["30_kv_inject.py"]
spec.loader.exec_module(Z)

SUBJECT = "2"
BAND = set(range(16, 26))   # Z's established identity band (matches FINDINGS.json / selectivity.json)
W = 4.0                       # Z's established working operating point (Z_strip_band16-25_w4)
STEPS = 30
CFG = 5.0
SEED = 0
THETAS = [256.0, 1000.0, 4000.0, 10000.0]  # Z native -> Lumina's baked theta

# established cross-model anchors (out/lumina/causal/selectivity.json), for calibration only
Z_STRIP_ANCHOR = {"peak": 0.10246, "entropy": 0.7027, "mass": 0.06925}
LUMINA_STRIP_ANCHOR = {"peak": 0.04254, "entropy": 0.8060, "mass": 0.03831}


def agg(atl, band):
    out = {}
    for lid, recs in atl.items():
        if lid not in band or not recs:
            continue
        n = len(recs)
        out[lid] = {
            "ref_mass": sum(r["ref_mass"] for r in recs) / n,
            "ref_entropy_norm": sum(r["ref_entropy_norm"] for r in recs) / n,
            "ref_peak": sum(r["ref_peak"] for r in recs) / n,
        }
    layers = sorted(out)
    return {
        "per_layer": {str(l): out[l] for l in layers},
        "mean_entropy_norm": sum(out[l]["ref_entropy_norm"] for l in layers) / len(layers),
        "mean_peak": sum(out[l]["ref_peak"] for l in layers) / len(layers),
        "mean_ref_mass": sum(out[l]["ref_mass"] for l in layers) / len(layers),
    }


def run(tag, **kw):
    t0 = time.time()
    d = dict(uid=SUBJECT, steps=STEPS, cfg=CFG, seed=SEED, band=None)
    d.update(kw)
    d.pop("band", None)
    Z.INJ["band"] = BAND
    im = Z.generate(**d)
    im.save(f"{TSOUT}/{tag}.png")
    dt = time.time() - t0
    print(f"[{tag}] done in {dt:.1f}s", flush=True)
    return im


results = {}
imgs = []
print("=== Z-Image single-model theta-causal sweep (band16-25, w4, u2) ===", flush=True)
for theta in THETAS:
    tag = f"theta{theta:g}"
    im = run(tag, w=W, refpos="thetasweep", ref_theta=theta, measure=True)
    sel = agg(Z.ATTN_LOG, BAND)
    results[tag] = {"theta": theta, **sel}
    print(f"  theta={theta:g}  mean_peak={sel['mean_peak']:.5f}  mean_entropy_norm={sel['mean_entropy_norm']:.4f}  "
          f"mean_ref_mass={sel['mean_ref_mass']:.4f}", flush=True)
    imgs.append((theta, im))

with open(f"{TSOUT}/theta_sweep_summary.json", "w") as f:
    json.dump({
        "subject": SUBJECT, "band": sorted(BAND), "w": W, "steps": STEPS, "cfg": CFG, "seed": SEED,
        "thetas": THETAS, "results": results,
        "anchors": {"Z_strip_band16-25_w4": Z_STRIP_ANCHOR, "Lumina_strip_band16-23_w2": LUMINA_STRIP_ANCHOR},
    }, f, indent=1)
print("saved:", f"{TSOUT}/theta_sweep_summary.json", flush=True)

# ---- montage: ref | theta256 | theta1000 | theta4000 | theta10000 | strip-anchor(established) ----
from PIL import Image, ImageDraw
def lab(path_or_im, txt, sub="", cell=512):
    im = (Image.open(path_or_im).convert("RGB") if isinstance(path_or_im, str) else path_or_im).resize((cell, cell)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, cell, 44], fill="black")
    d.text((6, 4), txt, fill="white")
    if sub:
        d.text((6, 24), sub, fill="#9fe")
    return im

cell = 512
ref_path = f"{os.environ['REF']}/u{SUBJECT}_ref.png"
panels = [(ref_path, "REF u2", "pink hair, purple eyes, red pendant")]
for theta, im in imgs:
    r = results[f"theta{theta:g}"]
    panels.append((im, f"thetasweep theta={theta:g}", f"peak={r['mean_peak']:.3f} ent={r['mean_entropy_norm']:.3f}"))
strip_anchor_path = f"{ZOUT}/Z_strip_b16-25_w4.png"
if os.path.exists(strip_anchor_path):
    panels.append((strip_anchor_path, "strip (established, pos-free)",
                    f"peak={Z_STRIP_ANCHOR['peak']:.3f} ent={Z_STRIP_ANCHOR['entropy']:.3f}"))

imglabs = [lab(p, t, s, cell) for p, t, s in panels]
canvas = Image.new("RGB", (cell * len(imglabs), cell), "white")
for c, im in enumerate(imglabs):
    canvas.paste(im, (c * cell, 0))
mont_path = f"{LDIR}/montage_theta_causal_sweep.png"
canvas.save(mont_path)
print("saved:", mont_path, flush=True)

# ---- dose-response curve ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

thetas_x = THETAS
peaks = [results[f"theta{t:g}"]["mean_peak"] for t in thetas_x]
ents = [results[f"theta{t:g}"]["mean_entropy_norm"] for t in thetas_x]

fig, ax1 = plt.subplots(figsize=(7.5, 5.2))
ax2 = ax1.twinx()
ax1.plot(thetas_x, peaks, "o-", color="tab:blue", label="mean per-query ref-attn peak")
ax2.plot(thetas_x, ents, "s-", color="tab:red", label="mean normalized entropy")
ax1.set_xscale("log")
ax1.set_xlabel("ref_theta (RoPE theta used to rotate the injected ref key, log scale)")
ax1.set_ylabel("mean ref-attn peak (content-selectivity)", color="tab:blue")
ax2.set_ylabel("mean normalized entropy (diffuseness)", color="tab:red")
ax1.tick_params(axis="y", labelcolor="tab:blue")
ax2.tick_params(axis="y", labelcolor="tab:red")

ax1.axhline(Z_STRIP_ANCHOR["peak"], ls=":", color="tab:blue", lw=1.2)
ax1.annotate("Z-strip anchor (peak)", (thetas_x[0], Z_STRIP_ANCHOR["peak"]), fontsize=7.5, color="tab:blue",
             textcoords="offset points", xytext=(0, 6))
ax1.axhline(LUMINA_STRIP_ANCHOR["peak"], ls="--", color="tab:blue", lw=1.2)
ax1.annotate("Lumina-strip anchor (peak)", (thetas_x[0], LUMINA_STRIP_ANCHOR["peak"]), fontsize=7.5, color="tab:blue",
             textcoords="offset points", xytext=(0, -12))
ax2.axhline(Z_STRIP_ANCHOR["entropy"], ls=":", color="tab:red", lw=1.2)
ax2.annotate("Z-strip anchor (entropy)", (thetas_x[-1], Z_STRIP_ANCHOR["entropy"]), fontsize=7.5, color="tab:red",
             textcoords="offset points", xytext=(-90, -12))
ax2.axhline(LUMINA_STRIP_ANCHOR["entropy"], ls="--", color="tab:red", lw=1.2)
ax2.annotate("Lumina-strip anchor (entropy)", (thetas_x[-1], LUMINA_STRIP_ANCHOR["entropy"]), fontsize=7.5, color="tab:red",
             textcoords="offset points", xytext=(-95, 6))

ax1.set_title("Z-Image single-model theta-causal sweep (thetasweep refpos, band16-25, w4, u2)\n"
              "dotted=Z-strip anchor  dashed=Lumina-strip anchor (from FINDINGS.json)")
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="center left")
ax1.grid(alpha=0.25)
plt.tight_layout()
dr_path = f"{LDIR}/theta_causal_doseresponse.png"
plt.savefig(dr_path, dpi=150)
print("saved:", dr_path, flush=True)
print("ALL DONE", flush=True)

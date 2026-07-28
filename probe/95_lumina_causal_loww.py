#!/usr/bin/env python
"""
95_lumina_causal_loww.py -- DECISIVE causal control for item 3.

Removes the value-magnitude confound from the ref_bias sweep:
  * low injection weight w in {1.0, 0.5} (no 2x-4x value amplification that
    blows Lumina's cond_norm/noise_norm renorm -> the earlier blank-wash);
  * deep band 16-23, MATCHED to Z-Image's identity band 16-25 (the earlier
    sweep used mid band 8-15, an unfair depth mismatch);
  * uniform additive ref_bias PRESERVES within-ref content-selection (adding a
    constant to all ref logits leaves their relative softmax unchanged; it only
    shifts the ref-vs-context split) -> a clean "more ref mass, same content
    addressing" knob.

Decision rule:
  * at Z-level mass (~0.07-0.2) with a COHERENT image and STILL no ref identity
    (pink hair) -> routing is not the barrier -> clean verdict (2).
  * if ref identity (pink) appears at coherent mass -> routing WAS the barrier
    -> one-knob causal fix -> verdict (1), AAAI-worthy.
"""
import importlib.util, os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
os.makedirs(OUTD, exist_ok=True)

spec = importlib.util.spec_from_file_location("lumina80", os.path.join(HERE, "80_lumina_inject.py"))
L = importlib.util.module_from_spec(spec)
sys.argv = ["80_lumina_inject.py"]
spec.loader.exec_module(L)

PROMPT = ("solo, single character, centered close-up anime portrait, upper body, "
          "detailed face, clean anime style, simple plain white background")
NEG = ""
ref_path = L.resolve_ref("2")


def run(tag, **kw):
    t0 = time.time()
    d = dict(prompt=PROMPT, neg=NEG, size=1024, refsize=512, steps=30, cfg=4.0, seed=0, ref_path=ref_path)
    d.update(kw)
    im = L.generate(**d)
    im.save(f"{OUTD}/{tag}.png")
    s = None
    if kw.get("measure"):
        s = {lid: sum(r["ref_mass"] for r in recs) / len(recs) for lid, recs in L.ATTN_LOG.items()}
        avg = sum(s.values()) / len(s)
        print(f"[{tag}] {time.time()-t0:.1f}s  avg_mass={avg:.4f}", flush=True)
    else:
        print(f"[{tag}] {time.time()-t0:.1f}s", flush=True)
    return im, s


band = set(range(16, 24))  # matched to Z identity band 16-25
res = {}
from PIL import Image, ImageDraw
def lab(im, txt, cell=512):
    im = im.resize((cell, cell)).copy(); d = ImageDraw.Draw(im)
    d.rectangle([0, 0, cell, 30], fill="black"); d.text((6, 7), txt, fill="white"); return im

for w in [1.0, 0.5]:
    imgs = []
    # finer low-end grid: w=1 washes later, so push mass into the 0.07-0.5 band
    for rb in [0.0, 1.0, 2.0, 3.0, 4.0, 6.0]:
        tag = f"D_deep16-23_w{w:g}_rb{rb:g}"
        im, s = run(tag, w=w, band=band, refpos="strip", measure=True, ref_bias=rb)
        res[tag] = {"w": w, "ref_bias": rb, "avg_mass": (sum(s.values())/len(s)) if s else None, "per_layer": s}
        imgs.append((f"w{w:g} rb{rb:g} m={res[tag]['avg_mass']:.2f}", im))
    cell = 448
    rimg = Image.open(ref_path).convert("RGB")
    panels = [("ref (pink)", rimg)] + imgs
    canvas = Image.new("RGB", (cell * len(panels), cell), "white")
    for c, (t, im) in enumerate(panels):
        canvas.paste(lab(im, t, cell), (c * cell, 0))
    canvas.save(f"{OUTD}/montage_D_deep_w{w:g}.png")
    print("saved montage:", f"{OUTD}/montage_D_deep_w{w:g}.png", flush=True)

with open(f"{OUTD}/partD_loww_summary.json", "w") as f:
    json.dump(res, f, indent=1)
print("saved partD_loww_summary.json", flush=True)
print("ALL DONE", flush=True)

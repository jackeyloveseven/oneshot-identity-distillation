#!/usr/bin/env python
"""
90_lumina_causal_driver.py -- decisive cross-model investigation driver.

Imports probe/80_lumina_inject.py as a module (single model load) and runs a
battery of measurement + causal configs using its instrumented InjectProcessor
(measure / ref_bias, both default-off in 80 itself). Does NOT modify 80's
existing behavior -- only calls its public generate()/dump_attn_log() with
explicit params.

Outputs -> /home/ubuntu/chengyanli/items/cherry/out/lumina/causal/
"""
import importlib.util, os, sys, json, time

HERE = os.path.dirname(os.path.abspath(__file__))
OUTD = "/home/ubuntu/chengyanli/items/cherry/out/lumina/causal"
os.makedirs(OUTD, exist_ok=True)

spec = importlib.util.spec_from_file_location("lumina80", os.path.join(HERE, "80_lumina_inject.py"))
L = importlib.util.module_from_spec(spec)
sys.argv = ["80_lumina_inject.py"]  # avoid argparse choking on our own argv
spec.loader.exec_module(L)

PROMPT = ("solo, single character, centered close-up anime portrait, upper body, "
          "detailed face, clean anime style, simple plain white background")
NEG = ""
SUBJECT = "2"
ref_path = L.resolve_ref(SUBJECT)

log = {}


def run(tag, **kw):
    t0 = time.time()
    defaults = dict(prompt=PROMPT, neg=NEG, size=1024, refsize=512, steps=20, cfg=4.0, seed=0,
                     ref_path=ref_path)
    defaults.update(kw)
    im = L.generate(**defaults)
    im.save(f"{OUTD}/{tag}.png")
    dt = time.time() - t0
    summary = None
    if kw.get("measure"):
        summary = {lid: {
            "mean_ref_mass": sum(r["ref_mass"] for r in recs) / len(recs),
            "mean_k_ref_norm": sum(r["k_ref_norm"] for r in recs) / len(recs),
            "mean_k_ctx_norm": sum(r["k_ctx_norm"] for r in recs) / len(recs),
        } for lid, recs in L.ATTN_LOG.items()}
    print(f"[{tag}] done in {dt:.1f}s" + (f"  layers_measured={len(summary)}" if summary else ""), flush=True)
    return im, summary


print("=== PART A: ref-attention quantification (item 1) ===", flush=True)

# A1: coherent-rendering band (16-23), strip (position-free) vs rope (post-RoPE,
# position-bound) at matched w -- tightest controlled comparison.
band_a = set(range(16, 24))
_, s_strip_16_23 = run("A1_strip_b16-23_w2", w=2.0, band=band_a, refpos="strip", measure=True)
log["A1_strip_b16-23_w2"] = s_strip_16_23
_, s_rope_16_23 = run("A1_rope_b16-23_w2", w=2.0, band=band_a, refpos="rope", measure=True)
log["A1_rope_b16-23_w2"] = s_rope_16_23

# A2: full-depth (0-25) strip at w=4 -- matches the ALREADY-KNOWN "full inject
# strip -> collapse" condition; record the raw ref-mass even though the image
# itself is expected to be degenerate (measurement is still valid: it reflects
# what queries in a collapsing forward pass actually did).
band_full = set(range(26))
_, s_strip_full = run("A2_strip_bfull_w4", w=4.0, band=band_full, refpos="strip", measure=True)
log["A2_strip_bfull_w4"] = s_strip_full

with open(f"{OUTD}/partA_summary.json", "w") as f:
    json.dump(log, f, indent=1)
print("saved partA_summary.json", flush=True)

print("=== PART B: causal ref_bias sweep on a coherent-rendering band (item 3) ===", flush=True)
# band 8-15 confirmed (out/lumina/u2_depth_w2_b8-15.png) to render coherently
# under strip w=2 with ZERO identity leak (grey hair, not the ref's pink hair).
band_b = set(range(8, 16))
biases = [0.0, 8.0, 16.0, 24.0, 32.0, 48.0]
partB = {}
imgs = []
for rb in biases:
    tag = f"B_biassweep_rb{rb:g}"
    im, s = run(tag, w=4.0, band=band_b, refpos="strip", measure=True, ref_bias=rb, steps=30)
    partB[tag] = {"ref_bias": rb, "per_layer": s}
    imgs.append((f"rb={rb:g}", im))

with open(f"{OUTD}/partB_summary.json", "w") as f:
    json.dump(partB, f, indent=1)
print("saved partB_summary.json", flush=True)

# montage: ref | off | rb sweep
from PIL import Image, ImageDraw
def label(im, txt, cell):
    im = im.resize((cell, cell)).copy()
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, min(cell, 340), 30], fill="black")
    d.text((6, 7), txt, fill="white")
    return im

cell = 512
rimg = Image.open(ref_path).convert("RGB")
im_off, _ = run("B_off", w="off", ref_path=None, band=band_b, refpos="strip")
panels = [("ref (pink hair)", rimg), ("off (no inject)", im_off)] + imgs
canvas = Image.new("RGB", (cell * len(panels), cell), "white")
for c, (txt, im) in enumerate(panels):
    canvas.paste(label(im, txt, cell), (c * cell, 0))
canvas.save(f"{OUTD}/montage_partB_biassweep.png")
print("saved montage:", f"{OUTD}/montage_partB_biassweep.png", flush=True)

print("ALL DONE", flush=True)
